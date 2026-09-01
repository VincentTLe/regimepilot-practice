"""Option screener: pick the expiry, enumerate debit verticals, filter, rank, plan.

Pure functions over pre-fetched contracts and snapshots. Selection rule
(approved 2026-08-31): nearest listed expiry (weeklies included) with DTE >= 5;
candidate strike pairs within +/-10% of spot at widths of 1-3 strike steps;
liquidity filter per leg (open interest floor + quote quality); rank survivors
by IV skew, flattest first, ties to higher combined open interest.

Order plans are pure data. Only broker.submit_paper_order acts on one.
"""

from __future__ import annotations

from datetime import date, datetime

from models import LegPlan, LegQuote, OpenSpread, OrderPlan, SpreadQuote

MIN_DTE = 5
STRIKE_BAND_PCT = 0.10  # candidate strikes within +/-10% of spot
MAX_WIDTH_STEPS = 3  # spread width of 1..3 strike steps
MIN_OPEN_INTEREST = 100
MAX_QUOTE_AGE_SECONDS = 10.0  # vs the broker's server clock
MAX_LEG_SPREAD_BPS = 350.0
MIN_NET_DEBIT = 0.05


def pick_expiration(expirations: set[date], today: date) -> date | None:
    """Nearest listed expiry at least MIN_DTE days out; None when there is none."""
    eligible = [exp for exp in expirations if (exp - today).days >= MIN_DTE]
    return min(eligible) if eligible else None


def check_leg(leg: LegQuote, server_time: datetime) -> str | None:
    """First failing liquidity/quality rule for one leg, or None when acceptable."""
    if leg.open_interest is None or leg.open_interest < MIN_OPEN_INTEREST:
        return "low_open_interest"
    if leg.bid is None or leg.ask is None or leg.quote_time is None:
        return "no_quote"
    if leg.bid <= 0 or leg.ask <= 0 or leg.bid > leg.ask:
        return "crossed_quote"
    age = server_time.timestamp() - leg.quote_time.timestamp()
    if age < 0:
        return "future_quote"
    if age > MAX_QUOTE_AGE_SECONDS:
        return "stale_quote"
    mid = (leg.bid + leg.ask) / 2
    if mid <= 0 or (leg.ask - leg.bid) / mid * 10_000 > MAX_LEG_SPREAD_BPS:
        return "wide_spread"
    if leg.implied_vol is None or leg.implied_vol <= 0:
        return "missing_iv"
    return None


def enumerate_spreads(
    quotes_by_strike: dict[float, LegQuote],
    direction: str,
    spot: float,
    expiration: date,
    underlying: str,
    server_time: datetime,
) -> tuple[list[SpreadQuote], dict[str, int]]:
    """All acceptable debit verticals at one expiry, plus rejection tallies.

    Bull call: long the lower strike, short the higher. Bear put: long the
    higher strike, short the lower. Both legs must sit inside the strike band
    and pass check_leg; the spread must price sanely (MIN_NET_DEBIT <= debit < width).
    """
    lo, hi = spot * (1 - STRIKE_BAND_PCT), spot * (1 + STRIKE_BAND_PCT)
    strikes = sorted(strike for strike in quotes_by_strike if lo <= strike <= hi)
    rejections: dict[str, int] = {}

    def _reject(reason: str) -> None:
        rejections[reason] = rejections.get(reason, 0) + 1

    leg_ok: dict[float, bool] = {}
    for strike in strikes:
        reason = check_leg(quotes_by_strike[strike], server_time)
        leg_ok[strike] = reason is None
        if reason is not None:
            _reject(reason)

    spreads: list[SpreadQuote] = []
    for i, lower in enumerate(strikes):
        for step in range(1, MAX_WIDTH_STEPS + 1):
            if i + step >= len(strikes):
                break
            higher = strikes[i + step]
            if not (leg_ok[lower] and leg_ok[higher]):
                continue
            if direction == "CALL":
                long_leg, short_leg = quotes_by_strike[lower], quotes_by_strike[higher]
            else:
                long_leg, short_leg = quotes_by_strike[higher], quotes_by_strike[lower]
            width = higher - lower
            net_debit = round(long_leg.ask - short_leg.bid, 2)  # type: ignore[operator]
            if not (MIN_NET_DEBIT <= net_debit < width):
                _reject("bad_debit")
                continue
            skew = abs(short_leg.implied_vol - long_leg.implied_vol)  # type: ignore[operator]
            spreads.append(
                SpreadQuote(
                    underlying=underlying,
                    direction=direction,
                    expiration=expiration,
                    long=long_leg,
                    short=short_leg,
                    width=width,
                    net_debit=net_debit,
                    skew=skew,
                )
            )
    return spreads, rejections


def rank_spreads(spreads: list[SpreadQuote]) -> list[SpreadQuote]:
    """Flattest IV skew first; ties go to higher combined open interest."""
    return sorted(
        spreads,
        key=lambda s: (s.skew, -((s.long.open_interest or 0) + (s.short.open_interest or 0))),
    )


def select_spread(
    quotes_by_strike: dict[float, LegQuote],
    direction: str,
    spot: float,
    expiration: date,
    underlying: str,
    server_time: datetime,
) -> tuple[SpreadQuote | None, dict[str, int]]:
    spreads, rejections = enumerate_spreads(
        quotes_by_strike, direction, spot, expiration, underlying, server_time
    )
    ranked = rank_spreads(spreads)
    return (ranked[0] if ranked else None), rejections


def build_entry_plan(spread: SpreadQuote, qty: int, cycle_id: str) -> OrderPlan:
    """Buy-to-open MLEG limit at the marketable net debit."""
    return OrderPlan(
        kind="enter",
        underlying=spread.underlying,
        qty=qty,
        limit_price=spread.net_debit,
        legs=(
            LegPlan(symbol=spread.long.symbol, side="buy", intent="buy_to_open"),
            LegPlan(symbol=spread.short.symbol, side="sell", intent="sell_to_open"),
        ),
        client_order_id=f"sp-{cycle_id}-enter-{spread.underlying}",
    )


def build_exit_plan(
    spread: OpenSpread,
    long_quote: LegQuote,
    short_quote: LegQuote,
    cycle_id: str,
) -> OrderPlan | None:
    """Sell-to-close MLEG limit at the marketable net price.

    Per the SDK convention the net limit is negative when the close collects a
    credit (the normal case) and positive when closing costs money.
    """
    if long_quote.bid is None or short_quote.ask is None:
        return None
    limit = round(short_quote.ask - long_quote.bid, 2)
    tag = f"{spread.expiration:%y%m%d}{spread.option_type}"
    return OrderPlan(
        kind="exit",
        underlying=spread.underlying,
        qty=spread.qty,
        limit_price=limit,
        legs=(
            LegPlan(symbol=spread.long_symbol, side="sell", intent="sell_to_close"),
            LegPlan(symbol=spread.short_symbol, side="buy", intent="buy_to_close"),
        ),
        client_order_id=f"sp-{cycle_id}-exit-{spread.underlying}-{tag}",
    )
