"""Deterministic pre-gates and session labels for Phase 3.

Pure module: no network, no vendor SDK, no LLM. Evaluates a FeaturePacket and
decides whether downstream reasoning may run, or whether the cycle must HOLD
for a documented reason.

Labels are always computed, even when a gate fails, so a consumer can log what
the session looked like at the moment of rejection.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from regimepilot.features import FeaturePacket
from regimepilot.models import Observation

# Do not open new trades in the last half hour of the regular session.
MIN_MINUTES_TO_CLOSE = 30

# A completed bar older than this is too stale for intraday momentum features.
MAX_BAR_AGE_SECONDS = 120

# Realized-vol thresholds on the raw 30-minute measurement from features.py.
# These are decimal values, not annualized percentages.
VOL_REGIME_LOW_MAX = 0.003
VOL_REGIME_MID_MAX = 0.008

MomentumAlign = Literal["aligned_up", "aligned_down", "mixed", "unknown"]
VolRegime = Literal["low", "mid", "high", "unknown"]
SessionPhase = Literal["open", "midday", "late", "unknown"]

HoldReason = Literal[
    "market_closed",
    "too_close_to_close",
    "stale_data",
    "missing_momentum",
    "already_in_position",
]


class SessionLabels(Observation):
    """Human-readable tags derived from a FeaturePacket."""

    momentum_align: MomentumAlign = "unknown"
    vol_regime: VolRegime = "unknown"
    session_phase: SessionPhase = "unknown"


class GateResult(Observation):
    """Whether pre-gates allow LLM reasoning to run."""

    passed: bool
    hold_reason: HoldReason | None = None
    labels: SessionLabels = Field(default_factory=SessionLabels)


def _sign(value: float | None) -> int | None:
    if value is None:
        return None
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def derive_momentum_align(return_15m: float | None, return_60m: float | None) -> MomentumAlign:
    """Classify short-horizon momentum agreement."""
    sign_15 = _sign(return_15m)
    sign_60 = _sign(return_60m)
    if sign_15 is None or sign_60 is None:
        return "unknown"
    if sign_15 == 0 or sign_60 == 0:
        return "mixed"
    if sign_15 == sign_60:
        return "aligned_up" if sign_15 > 0 else "aligned_down"
    return "mixed"


def derive_vol_regime(realized_vol_30m: float | None) -> VolRegime:
    """Bucket the raw realized-vol measurement."""
    if realized_vol_30m is None:
        return "unknown"
    if realized_vol_30m <= VOL_REGIME_LOW_MAX:
        return "low"
    if realized_vol_30m <= VOL_REGIME_MID_MAX:
        return "mid"
    return "high"


def derive_session_phase(minutes_since_open: float | None) -> SessionPhase:
    """Rough intraday phase from minutes elapsed since the regular open."""
    if minutes_since_open is None:
        return "unknown"
    if minutes_since_open < 60:
        return "open"
    if minutes_since_open < 300:
        return "midday"
    return "late"


def derive_labels(packet: FeaturePacket) -> SessionLabels:
    """Compute all session labels from one FeaturePacket."""
    return SessionLabels(
        momentum_align=derive_momentum_align(packet.return_15m, packet.return_60m),
        vol_regime=derive_vol_regime(packet.realized_vol_30m),
        session_phase=derive_session_phase(packet.minutes_since_open),
    )


def evaluate_gates(
    packet: FeaturePacket,
    *,
    has_open_option_position: bool = False,
) -> GateResult:
    """Return pass/fail plus labels. First failing rule wins."""
    labels = derive_labels(packet)

    if has_open_option_position:
        return GateResult(passed=False, hold_reason="already_in_position", labels=labels)

    if packet.market_is_open is not True:
        return GateResult(passed=False, hold_reason="market_closed", labels=labels)

    if packet.minutes_to_close is not None and packet.minutes_to_close < MIN_MINUTES_TO_CLOSE:
        return GateResult(passed=False, hold_reason="too_close_to_close", labels=labels)

    if packet.bar_age_seconds is not None and packet.bar_age_seconds > MAX_BAR_AGE_SECONDS:
        return GateResult(passed=False, hold_reason="stale_data", labels=labels)

    if packet.return_15m is None or packet.return_60m is None:
        return GateResult(passed=False, hold_reason="missing_momentum", labels=labels)

    return GateResult(passed=True, hold_reason=None, labels=labels)
