"""Position manager + risk manager: pure money math.

Pairs raw option legs into vertical spreads, decides mechanical exits
(stop / take-profit / expiry), and sizes new entries against the
equity-relative caps. The LLM is never consulted here.
"""

from __future__ import annotations

import math
import re
from datetime import date, datetime

import settings
from data_models import Event, ExitDecision, LegPosition, LegQuote, OpenSpread

# Exit thresholds and risk caps live in settings.yaml (approved 2026-08-31).

_OCC = re.compile(r"^([A-Z]{1,6})(\d{6})([CP])(\d{8})$")


def parse_occ(symbol: str) -> tuple[str, date, str, float] | None:
    """OCC option symbol -> (underlying, expiration, type, strike), or None."""
    match = _OCC.match(symbol.strip().upper())
    if match is None:
        return None
    root, yymmdd, option_type, strike_raw = match.groups()
    try:
        expiration = datetime.strptime(yymmdd, "%y%m%d").date()
    except ValueError:
        return None
    return root, expiration, option_type, int(strike_raw) / 1000.0


def pair_spreads(
    legs: tuple[LegPosition, ...],
) -> tuple[list[OpenSpread], list[str]]:
    """Group option legs into debit verticals; anything unrecognized is a warning.

    A spread is exactly one long and one short leg with equal absolute quantity on
    the same underlying, expiration and type. Unpaired legs are reported and never
    touched by this agent.
    """
    groups: dict[tuple[str, date, str], list[LegPosition]] = {}
    for leg in legs:
        groups.setdefault((leg.underlying, leg.expiration, leg.option_type), []).append(leg)

    spreads: list[OpenSpread] = []
    warnings: list[str] = []
    for (underlying, expiration, option_type), members in sorted(groups.items()):
        longs = [leg for leg in members if leg.qty > 0]
        shorts = [leg for leg in members if leg.qty < 0]
        if len(longs) != 1 or len(shorts) != 1 or longs[0].qty != -shorts[0].qty:
            warnings.append(
                f"unpaired legs on {underlying} {expiration} {option_type}: "
                + ", ".join(f"{leg.symbol} qty={leg.qty}" for leg in members)
            )
            continue
        long_leg, short_leg = longs[0], shorts[0]
        net_entry_debit = None
        if long_leg.avg_entry_price is not None and short_leg.avg_entry_price is not None:
            debit = long_leg.avg_entry_price - short_leg.avg_entry_price
            net_entry_debit = debit if debit > 0 else None  # non-debit pair: unknown
        spreads.append(
            OpenSpread(
                underlying=underlying,
                expiration=expiration,
                option_type=option_type,
                long_symbol=long_leg.symbol,
                short_symbol=short_leg.symbol,
                qty=long_leg.qty,
                net_entry_debit=net_entry_debit,
            )
        )
    return spreads, warnings


def opposing_event_fired(spread: OpenSpread, events: tuple[Event, ...]) -> bool:
    """True when any entry event points against the spread's direction.

    A call spread ("C") is bullish, so any PUT-direction event opposes it;
    a put spread ("P") is bearish, so any CALL-direction event opposes it.
    """
    against = "PUT" if spread.option_type == "C" else "CALL"
    return any(event.direction == against for event in events)


def exit_decision(
    spread: OpenSpread,
    long_quote: LegQuote | None,
    short_quote: LegQuote | None,
    today: date,
    opposing_event: bool = False,
) -> ExitDecision | None:
    """Mechanical exit verdict for one open spread, or None to keep holding.

    Precedence: expiry, reversal, stop, take-profit. Expiry (DTE <=
    settings.EXIT_DTE) and reversal (an entry event against the spread, if
    settings.REVERSAL_EXIT) exit even on missing marks or unknown entry debit —
    they are signal-based. Stop and take-profit need both a known entry debit
    and fresh two-sided marks; when either is unknown we hold and let the
    caller log the gap rather than guess.
    """
    dte = (spread.expiration - today).days
    if dte <= settings.EXIT_DTE:
        net_mark = _net_mark(long_quote, short_quote)
        return ExitDecision(spread=spread, reason="expiry", net_mark=net_mark)
    if opposing_event and settings.REVERSAL_EXIT:
        net_mark = _net_mark(long_quote, short_quote)
        return ExitDecision(spread=spread, reason="reversal", net_mark=net_mark)
    if spread.net_entry_debit is None:
        return None
    net_mark = _net_mark(long_quote, short_quote)
    if net_mark is None:
        return None
    if net_mark <= settings.STOP_FRACTION * spread.net_entry_debit:
        return ExitDecision(spread=spread, reason="stop", net_mark=net_mark)
    if net_mark >= settings.TAKE_PROFIT_MULT * spread.net_entry_debit:
        return ExitDecision(spread=spread, reason="take_profit", net_mark=net_mark)
    return None


def _net_mark(long_quote: LegQuote | None, short_quote: LegQuote | None) -> float | None:
    if long_quote is None or short_quote is None:
        return None
    if long_quote.mid is None or short_quote.mid is None:
        return None
    return long_quote.mid - short_quote.mid


def open_premium_at_risk(spreads: list[OpenSpread]) -> float | None:
    """Total entry debit of all open spreads in dollars; None if any is unknown.

    An unknown component makes the whole figure unknown so the risk manager
    refuses new entries instead of undercounting exposure.
    """
    total = 0.0
    for spread in spreads:
        if spread.net_entry_debit is None:
            return None
        total += spread.net_entry_debit * spread.qty * 100.0
    return total


def size_entry(
    net_debit: float,
    equity: float | None,
    open_risk: float | None,
    cycle_spent: float,
) -> tuple[int, str | None]:
    """Contracts to buy for one spread entry, or (0, reason) when refused."""
    if equity is None or equity <= 0:
        return 0, "unknown_equity"
    if open_risk is None:
        return 0, "unknown_open_risk"
    if net_debit <= 0:
        return 0, "bad_debit"
    per_entry_cap = settings.PER_ENTRY_FRACTION * equity
    cycle_room = settings.PER_CYCLE_FRACTION * equity - cycle_spent
    total_room = settings.TOTAL_FRACTION * equity - open_risk - cycle_spent
    cap = min(per_entry_cap, cycle_room, total_room)
    qty = math.floor(cap / (net_debit * 100.0))
    if qty < 1:
        return 0, "risk_caps"
    return qty, None
