"""Evidence assembly: one briefing for the LLM portfolio manager.

Combines FeaturePacket, NewsPacket, GateResult and the paper account into one
``EvidencePacket``. Read-only: no orders, no LLM calls here.

Since the portfolio agent (approved 2026-08-27) the account is not one flag
but a ``PortfolioContext``: every held SPY option with its marks and journal
memo, every pending SPY option order by exact symbol, and a deterministic
entry pre-check. The entry gates (market open, time to close, fresh bars,
momentum) decide only whether a NEW entry is eligible; a held position is
managed whatever they say, and closing one is judged later by the exit rules
in ``risk`` against a fresh account read.

A failed features or account read aborts the briefing: "unknown" must never
reach the model as "flat".
"""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

from regimepilot.account import AccountError, observe_account, parse_occ_symbol
from regimepilot.config import ConfigError, load_settings
from regimepilot.console import tolerant_console
from regimepilot.features import FeaturePacket, session_date_of, to_utc
from regimepilot.gates import GateResult, evaluate_gates
from regimepilot.history import HistoryError, observe_features
from regimepilot.models import (
    UNDERLYING_SYMBOL,
    AccountHint,
    AccountState,
    EvidencePacket,
    GatesEvidence,
    NewsEvidence,
    NewsPacket,
    OpenPositionContext,
    PendingOrder,
    PortfolioContext,
    PortfolioLimits,
    PositionMemo,
    UnderlyingEvidence,
)
from regimepilot.news import NewsError, format_summary as format_news_summary, observe_news, unavailable_news_packet
from regimepilot.risk import DEFAULT_LIMITS
from regimepilot.smoke_test import build_clients

UNDERLYING = UNDERLYING_SYMBOL

# A multi-leg SPY option order has no symbol of its own. It is still a pending
# order, and while it is open the portfolio is not fully known, so it counts
# as a pending buy for the entry pre-check.
MULTI_LEG_SYMBOL = "(multi-leg)"

__all__ = [
    "EvidenceError",
    "MULTI_LEG_SYMBOL",
    "build_evidence",
    "build_portfolio_context",
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


def _pending_orders(account: AccountState) -> tuple[PendingOrder, ...]:
    return tuple(
        PendingOrder(
            order_id=order.order_id,
            symbol=order.symbol or MULTI_LEG_SYMBOL,
            side=order.side,
            qty=order.qty,
            status=order.status,
        )
        for order in account.open_orders
        if order.is_spy_option
    )


def _is_pending_buy(order: PendingOrder) -> bool:
    return order.symbol == MULTI_LEG_SYMBOL or (order.side or "").lower() == "buy"


def build_portfolio_context(
    account: AccountState,
    gates: GateResult,
    *,
    observed_at: datetime,
    memory: Mapping[str, PositionMemo] | None = None,
    limits: PortfolioLimits = DEFAULT_LIMITS,
) -> PortfolioContext:
    """Pure assembly of what the manager sees: positions, pending orders, entry pre-check.

    Raises ``EvidenceError`` for a SPY option position this agent cannot
    manage (a short one, or one without a quantity): such an account must be
    handled by hand, never reasoned about as if it were understood.
    """
    memos = memory or {}
    session = session_date_of(observed_at)
    pending = _pending_orders(account)
    pending_by_symbol: dict[str, PendingOrder] = {}
    for order in pending:
        pending_by_symbol.setdefault(order.symbol, order)

    held = sorted((p for p in account.positions if p.is_spy_option), key=lambda p: p.symbol)
    positions: list[OpenPositionContext] = []
    for position in held:
        parsed = parse_occ_symbol(position.symbol)
        if parsed is None:
            raise EvidenceError(f"cannot decode the option symbol of a held position: {position.symbol}")
        if position.side is not None and position.side.lower() != "long":
            raise EvidenceError(f"unsupported {position.side} SPY option position {position.symbol}")
        if position.qty is None or position.qty < 1:
            raise EvidenceError(f"held SPY option position {position.symbol} has no usable quantity")
        memo = memos.get(position.symbol)
        entered_at = None if memo is None else memo.entered_at
        hours_held = (
            None
            if entered_at is None
            else round((to_utc(observed_at) - entered_at).total_seconds() / 3600, 2)
        )
        pending_here = pending_by_symbol.get(position.symbol)
        positions.append(
            OpenPositionContext(
                symbol=position.symbol,
                option_type=parsed.option_type,
                strike_price=parsed.strike_price,
                expiration_date=parsed.expiration_date,
                days_to_expiration=(parsed.expiration_date - session).days,
                qty=position.qty,
                avg_entry_price=position.avg_entry_price,
                cost_basis=position.cost_basis,
                current_price=position.current_price,
                unrealized_pl=position.unrealized_pl,
                unrealized_plpc=position.unrealized_plpc,
                pending_order_side=None if pending_here is None else pending_here.side,
                entered_at=entered_at,
                hours_held=hours_held,
                entry_thesis=None if memo is None else memo.entry_thesis,
                previous_decision=None if memo is None else memo.previous_decision,
            )
        )

    cost_bases = [p.cost_basis for p in held]
    total_cost_basis = None if any(c is None for c in cost_bases) else round(sum(cost_bases), 2)
    held_symbols = {p.symbol for p in positions}
    pending_buys = [o for o in pending if _is_pending_buy(o)]
    committed = len(positions) + len({o.symbol for o in pending_buys if o.symbol not in held_symbols})

    # Deterministic entry pre-check, in the order the risk layer re-checks it.
    # None means a new entry may be proposed; the risk layer has the last word.
    blocked: str | None = None
    if not gates.passed:
        blocked = gates.hold_reason or "entry_gate"
    elif pending_buys:
        blocked = "pending_buy_order"
    elif committed >= limits.max_open_positions:
        blocked = "max_positions"
    elif total_cost_basis is None:
        blocked = "unknown_cost_basis"
    elif total_cost_basis >= limits.max_total_premium_usd:
        blocked = "total_premium_cap"
    elif account.options_buying_power is None:
        blocked = "unknown_buying_power"

    return PortfolioContext(
        positions=tuple(positions),
        open_position_count=len(positions),
        total_cost_basis=total_cost_basis,
        options_buying_power=account.options_buying_power,
        equity=account.equity,
        pending_orders=pending,
        entry_allowed=blocked is None,
        entry_blocked_reason=blocked,
        limits=limits,
    )


def build_evidence(
    features: FeaturePacket,
    news: NewsPacket,
    gates: GateResult,
    *,
    portfolio: PortfolioContext | None = None,
) -> EvidencePacket:
    """Pure assembly: same inputs, same packet, no I/O."""
    has_position = portfolio is not None and portfolio.open_position_count > 0
    return EvidencePacket(
        observed_at=features.observed_at,
        symbol=features.symbol,
        gates=_gates_from_result(gates),
        underlying=_underlying_from_features(features),
        news=_news_from_packet(news),
        account=AccountHint(has_open_option_position=has_position),
        portfolio=portfolio,
    )


def observe_evidence(
    trading_client: Any,
    data_client: Any,
    news_client: Any,
    *,
    now: datetime | None = None,
    symbol: str = UNDERLYING,
    memory: Mapping[str, PositionMemo] | None = None,
    limits: PortfolioLimits = DEFAULT_LIMITS,
) -> EvidencePacket:
    """Read features, the account and news, evaluate the entry gates, build the portfolio.

    Feature reads raise ``EvidenceError`` wrapping ``HistoryError``, and the
    account read raises it wrapping ``AccountError``: without a trustworthy
    answer to "what does the account hold?" there is no briefing. News
    failures degrade to an unavailable news packet rather than aborting.
    ``memory`` is the journal's memo per held symbol (see ``memory``).
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
        account = observe_account(trading_client, now=observed_at)
    except AccountError as error:
        raise EvidenceError(str(error)) from None

    try:
        news = observe_news(news_client, now=observed_at, symbol=symbol)
    except NewsError:
        news = unavailable_news_packet(observed_at=observed_at)

    gates = evaluate_gates(features)
    portfolio = build_portfolio_context(
        account, gates, observed_at=observed_at, memory=memory, limits=limits
    )
    return build_evidence(features, news, gates, portfolio=portfolio)


def _number(value: float | None, digits: int = 2) -> str:
    return "-" if value is None else f"{value:,.{digits}f}"


def format_summary(packet: EvidencePacket) -> str:
    """Compact multi-section summary for terminal inspection."""
    gate_state = "PASS" if packet.gates.passed else f"HOLD ({packet.gates.hold_reason})"
    lines = [
        f"RegimePilot evidence  {packet.symbol}  @ "
        f"{packet.observed_at.strftime('%Y-%m-%d %H:%M:%SZ')}",
        f"  {'entry gates':<16} {gate_state}",
        f"  {'labels':<16} momentum={packet.gates.momentum_align}",
        f"  {'returns':<16} 15m={packet.underlying.return_15m!r}"
        f"  60m={packet.underlying.return_60m!r}",
        f"  {'news':<16} available={packet.news.available}  count={packet.news.item_count}",
    ]
    for item in packet.news.items:
        lines.append(f"    - [{item.age_minutes:.0f}m] {item.headline}")

    portfolio = packet.portfolio
    if portfolio is not None:
        entry = "allowed" if portfolio.entry_allowed else f"blocked ({portfolio.entry_blocked_reason})"
        lines.append(
            f"  {'portfolio':<16} {portfolio.open_position_count} position(s)"
            f"   cost basis {_number(portfolio.total_cost_basis)}"
            f"   options bp {_number(portfolio.options_buying_power)}   new entry {entry}"
        )
        for position in portfolio.positions:
            lines.append(
                f"    {position.symbol:<22} {position.option_type:<4} strike {_number(position.strike_price)}"
                f"  {position.days_to_expiration} DTE  qty {position.qty:g}"
                f"  entry {_number(position.avg_entry_price)}  mark {_number(position.current_price)}"
                f"  upl {_number(position.unrealized_pl)}"
                + (f"  pending {position.pending_order_side}" if position.pending_order_side else "")
            )
            if position.entry_thesis:
                lines.append(f"      thesis: {position.entry_thesis}")
            if position.previous_decision:
                lines.append(f"      last: {position.previous_decision}")
        for order in portfolio.pending_orders:
            if order.symbol not in {p.symbol for p in portfolio.positions}:
                lines.append(f"    pending {order.side or '-'} {order.symbol}  qty {order.qty}  {order.status or '-'}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Print an EvidencePacket summary, or the packet itself with ``--json``."""
    tolerant_console()
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
