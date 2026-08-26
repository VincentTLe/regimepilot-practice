"""Evidence assembly for Phase 3C.

Combines FeaturePacket, NewsPacket and GateResult into one briefing object
suitable for LLM direction reasoning. Read-only: no orders, no LLM calls here.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Any

from regimepilot.config import ConfigError, load_settings
from regimepilot.features import FeaturePacket, to_utc
from regimepilot.gates import GateResult, evaluate_gates
from regimepilot.history import HistoryError, observe_features
from regimepilot.models import (
    UNDERLYING_SYMBOL,
    AccountHint,
    EvidencePacket,
    GatesEvidence,
    NewsEvidence,
    NewsPacket,
    UnderlyingEvidence,
)
from regimepilot.news import NewsError, format_summary as format_news_summary, observe_news, unavailable_news_packet
from regimepilot.smoke_test import build_clients

UNDERLYING = UNDERLYING_SYMBOL

__all__ = [
    "EvidenceError",
    "build_evidence",
    "format_summary",
    "observe_evidence",
    "main",
]


class EvidenceError(RuntimeError):
    """An evidence observation could not be completed."""


def _underlying_from_features(features: FeaturePacket) -> UnderlyingEvidence:
    return UnderlyingEvidence(
        market_is_open=features.market_is_open,
        data_feed=features.data_feed,
        minutes_since_open=features.minutes_since_open,
        minutes_to_close=features.minutes_to_close,
        spread_bps=features.spread_bps,
        bar_age_seconds=features.bar_age_seconds,
        return_15m=features.return_15m,
        return_60m=features.return_60m,
        return_since_open=features.return_since_open,
        overnight_gap_pct=features.overnight_gap_pct,
        realized_vol_30m=features.realized_vol_30m,
    )


def _gates_from_result(gates: GateResult) -> GatesEvidence:
    return GatesEvidence(
        passed=gates.passed,
        hold_reason=gates.hold_reason,
        momentum_align=gates.labels.momentum_align,
    )


def _news_from_packet(news: NewsPacket) -> NewsEvidence:
    return NewsEvidence(
        available=news.available,
        item_count=news.item_count,
        items=news.items,
    )


def build_evidence(
    features: FeaturePacket,
    news: NewsPacket,
    gates: GateResult,
    *,
    has_open_option_position: bool = False,
) -> EvidencePacket:
    """Pure assembly: same inputs, same packet, no I/O."""
    return EvidencePacket(
        observed_at=features.observed_at,
        symbol=features.symbol,
        gates=_gates_from_result(gates),
        underlying=_underlying_from_features(features),
        news=_news_from_packet(news),
        account=AccountHint(has_open_option_position=has_open_option_position),
    )


def observe_evidence(
    trading_client: Any,
    data_client: Any,
    news_client: Any,
    *,
    now: datetime | None = None,
    symbol: str = UNDERLYING,
    has_open_option_position: bool = False,
) -> EvidencePacket:
    """Read features and news, evaluate gates, and return one EvidencePacket.

    Feature reads raise ``EvidenceError`` wrapping ``HistoryError``. News
    failures degrade to an unavailable news packet rather than aborting the
    whole briefing.
    """
    observed_at = to_utc(now) if now else datetime.now(timezone.utc)

    try:
        features = observe_features(
            trading_client,
            data_client,
            now=observed_at,
            symbol=symbol,
        )
    except HistoryError as error:
        raise EvidenceError(str(error)) from None

    try:
        news = observe_news(news_client, now=observed_at, symbol=symbol)
    except NewsError:
        news = unavailable_news_packet(observed_at=observed_at)

    gates = evaluate_gates(
        features,
        has_open_option_position=has_open_option_position,
    )
    return build_evidence(
        features,
        news,
        gates,
        has_open_option_position=has_open_option_position,
    )


def format_summary(packet: EvidencePacket) -> str:
    """Compact multi-section summary for terminal inspection."""
    gate_state = "PASS" if packet.gates.passed else f"HOLD ({packet.gates.hold_reason})"
    lines = [
        f"RegimePilot evidence  {packet.symbol}  @ "
        f"{packet.observed_at.strftime('%Y-%m-%d %H:%M:%SZ')}",
        f"  {'gates':<16} {gate_state}",
        f"  {'labels':<16} momentum={packet.gates.momentum_align}",
        f"  {'returns':<16} 15m={packet.underlying.return_15m!r}"
        f"  60m={packet.underlying.return_60m!r}",
        f"  {'news':<16} available={packet.news.available}  count={packet.news.item_count}",
    ]
    for item in packet.news.items:
        lines.append(f"    - [{item.age_minutes:.0f}m] {item.headline}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Print an EvidencePacket summary, or the packet itself with ``--json``."""
    arguments = list(sys.argv[1:] if argv is None else argv)

    try:
        settings = load_settings()
        trading_client, data_client = build_clients(settings)
    except ConfigError as error:
        print(f"configuration error: {error}", file=sys.stderr)
        return 1

    from regimepilot.news import build_news_client

    news_client = build_news_client(settings)

    try:
        packet = observe_evidence(trading_client, data_client, news_client)
    except EvidenceError as error:
        print(f"evidence read failed: {error}", file=sys.stderr)
        return 1

    if "--json" in arguments:
        print(json.dumps(json.loads(packet.model_dump_json()), indent=2))
    else:
        print(format_summary(packet))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
