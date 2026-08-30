"""Deterministic regime classification: bars and IV history in, one RegimeReading out.

Pure module, mirroring ``features.py`` and ``gates.py``: no network, no vendor
SDK, no LLM. ``evidence.py`` calls this alongside ``features.build_feature_packet``
and attaches the result to the same evidence bundle the model already reads,
so a regime label is one more grounded fact in the briefing, not a second
source of truth.

Two rules mirror ``features.py``'s null policy:

* A missing or insufficient input makes its statistic ``None``, and a
  statistic that is ``None`` can only ever produce ``"unknown"``, never a
  guessed regime.
* Thresholds below are proposed defaults, not yet approved the way
  ``gates.MIN_MINUTES_TO_CLOSE`` was: tune them against the backtest scorecard
  in ``score.py`` before trusting them on a live cycle, and record the
  approval the same way the README already records other threshold changes.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal

from regimepilot.models import Observation, OhlcvBar

# Proposed defaults (unapproved). SPY's long-run realized vol sits roughly in
# the 12-20% annualized range; above this is treated as a "high vol" day.
HIGH_VOL_ANNUALIZED_THRESHOLD = 0.20

# A one-period directional index above this is conventionally read as
# "trending" (the common ADX convention); below it, price action is read as
# range-bound regardless of its net sign over the window.
TREND_STRENGTH_THRESHOLD = 20.0

# features.realized_vol_30m is the *raw* root-sum-of-squared 1-minute log
# returns over a 30-minute window -- unscaled, as features.py documents. To
# compare it against an annualized threshold it must be scaled by the ratio of
# trading minutes in a year to minutes in the sample window. A regular US
# session is 390 minutes; 252 sessions/year gives 98,280 trading minutes/year.
_TRADING_MINUTES_PER_YEAR = 252 * 390
_ANNUALIZATION_FACTOR = math.sqrt(_TRADING_MINUTES_PER_YEAR / 30)

RegimeLabel = Literal["trending_up", "trending_down", "high_vol_chop", "low_vol_drift", "unknown"]


class RegimeReading(Observation):
    """One regime classification, with the raw statistics it was built from.

    The raw fields are carried alongside ``label`` so a consumer -- the LLM
    prompt, the journal, a backtest report -- can see the evidence behind the
    label rather than trusting a bare enum value.
    """

    label: RegimeLabel = "unknown"
    realized_vol_annualized: float | None = None
    trend_strength: float | None = None
    iv_rank: float | None = None


def annualize_realized_vol_30m(realized_vol_30m: float | None) -> float | None:
    """Scale ``features.realized_vol_30m`` to an annualized decimal volatility.

    ``None`` in, ``None`` out: a period too short to measure is not a zero-vol
    day, it is an unmeasured one, and must not be classified as calm.
    """
    if realized_vol_30m is None:
        return None
    return realized_vol_30m * _ANNUALIZATION_FACTOR


def _true_range(high: float, low: float, previous_close: float) -> float:
    return max(high - low, abs(high - previous_close), abs(low - previous_close))


def trend_strength_adx(bars: Sequence[OhlcvBar], *, period: int = 14) -> float | None:
    """A one-period directional index (DX), used as an "ADX-like" trend-strength proxy.

    This is deliberately simpler than Wilder's smoothed ADX: DX is computed
    once over the trailing ``period`` bars using a plain average rather than
    Wilder's running smoothing. It answers the same question -- "how directional
    has price action been, independent of which direction" -- with less state
    to carry between calls, which matters more for a backtest replayed bar by
    bar than a textbook-exact indicator would. Callers wanting true ADX should
    smooth a series of these DX readings themselves.

    ``bars`` must carry ``high``, ``low`` and ``close`` on every bar in the
    window and must already be sorted oldest to newest; a bar missing any of
    the three, or too few bars for ``period``, yields ``None`` rather than a
    value computed on a gap.
    """
    usable = [b for b in bars if b.high is not None and b.low is not None and b.close is not None]
    if len(usable) < period + 1:
        return None
    window = usable[-(period + 1):]

    plus_dm: list[float] = []
    minus_dm: list[float] = []
    true_ranges: list[float] = []
    for earlier, later in zip(window, window[1:]):
        up_move = later.high - earlier.high
        down_move = earlier.low - later.low
        plus_dm.append(up_move if (up_move > down_move and up_move > 0) else 0.0)
        minus_dm.append(down_move if (down_move > up_move and down_move > 0) else 0.0)
        true_ranges.append(_true_range(later.high, later.low, earlier.close))

    atr = sum(true_ranges) / period
    if atr == 0:
        return 0.0

    plus_di = 100 * (sum(plus_dm) / period) / atr
    minus_di = 100 * (sum(minus_dm) / period) / atr
    denominator = plus_di + minus_di
    if denominator == 0:
        return 0.0
    return 100 * abs(plus_di - minus_di) / denominator


def iv_rank(current_iv: float | None, historical_ivs: Sequence[float]) -> float | None:
    """Where ``current_iv`` sits within ``historical_ivs``, as a 0-100 percentile-style score.

    Returns ``None`` when there is no current reading or no history to rank it
    against -- an empty or single-point history cannot support a rank, and a
    caller should treat that as "not yet available" rather than a midpoint
    guess. A degenerate history (all one value) reports 50.0: the reading is
    neither high nor low relative to a history with no spread.
    """
    if current_iv is None or not historical_ivs:
        return None
    lowest, highest = min(historical_ivs), max(historical_ivs)
    if highest == lowest:
        return 50.0
    rank = 100 * (current_iv - lowest) / (highest - lowest)
    return max(0.0, min(100.0, rank))


def _trend_direction(return_60m: float | None) -> Literal["up", "down", "flat", "unknown"]:
    if return_60m is None:
        return "unknown"
    if return_60m > 0:
        return "up"
    if return_60m < 0:
        return "down"
    return "flat"


def classify_regime(
    *,
    realized_vol_30m: float | None,
    return_60m: float | None,
    trend_bars: Sequence[OhlcvBar],
    current_iv: float | None = None,
    iv_history: Sequence[float] = (),
    trend_period: int = 14,
) -> RegimeReading:
    """Combine volatility, trend strength and (optionally) IV rank into one label.

    Decision order, first match wins:

    1. Either the annualized vol or the trend strength is unmeasurable ->
       ``"unknown"``. A regime label must never be guessed from a partial
       reading; downstream code that branches on regime falls back to its
       most conservative behavior on ``"unknown"``, the same way a failed gate
       does.
    2. Trend strength at or above ``TREND_STRENGTH_THRESHOLD`` -> a trending
       regime, direction taken from the sign of ``return_60m``. A trend
       reading with a flat or unknown direction falls through to the
       volatility-based labels below, because "strongly directional but no
       net sign" describes the vol-based labels better than a trend label
       with no direction would.
    3. Otherwise, split on volatility: at or above
       ``HIGH_VOL_ANNUALIZED_THRESHOLD`` -> ``"high_vol_chop"``, else
       ``"low_vol_drift"``.

    ``iv_rank`` is carried on the result for the evidence bundle and the
    backtest report, but never changes the label itself: IV rank speaks to
    whether options are expensive right now, not to which direction or how
    choppy the underlying is, so it stays a separate fact the LLM (or a human)
    weighs alongside the label rather than a fourth classification axis.
    """
    vol = annualize_realized_vol_30m(realized_vol_30m)
    trend_strength = trend_strength_adx(trend_bars, period=trend_period)
    rank = iv_rank(current_iv, iv_history)

    if vol is None or trend_strength is None:
        return RegimeReading(
            label="unknown", realized_vol_annualized=vol, trend_strength=trend_strength, iv_rank=rank
        )

    direction = _trend_direction(return_60m)
    if trend_strength >= TREND_STRENGTH_THRESHOLD and direction in ("up", "down"):
        label: RegimeLabel = "trending_up" if direction == "up" else "trending_down"
    elif vol >= HIGH_VOL_ANNUALIZED_THRESHOLD:
        label = "high_vol_chop"
    else:
        label = "low_vol_drift"

    return RegimeReading(
        label=label, realized_vol_annualized=vol, trend_strength=trend_strength, iv_rank=rank
    )


def format_summary(reading: RegimeReading) -> str:
    """A compact, honest one-block summary, matching features.format_summary's style."""

    def fmt(value: float | None, digits: int = 4) -> str:
        return "null" if value is None else f"{value:.{digits}f}"

    return "\n".join(
        [
            f"RegimePilot regime  label={reading.label}",
            f"  {'realized_vol_annualized':<26} {fmt(reading.realized_vol_annualized)}",
            f"  {'trend_strength (DX)':<26} {fmt(reading.trend_strength, 2)}",
            f"  {'iv_rank':<26} {fmt(reading.iv_rank, 1)}",
        ]
    )
