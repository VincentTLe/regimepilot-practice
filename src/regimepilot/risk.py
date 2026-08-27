"""Deterministic risk decisions for the portfolio MVP (methodology approved
2026-08-27, with the three corrections of the same day).

Pure module: no network, no vendor SDK, no LLM. It takes what the selector
chose (``SelectionResult``) or which held contract the agent wants closed
(``OpenPositionContext``), plus what ``execution.observe_execution_state``
re-read right before ordering (``ExecutionState``), and answers "may one
order be built, and which?" with rules a person can check by hand. The LLM
never reaches this module: quantity, limit price, order type and
time-in-force are fixed here and nowhere else.

Entry rules (``decide_order``), applied in this order, first refusal wins:

 1. ``no_contract``                        selection carries no ``SelectedContract``
 2. ``pending_order_conflict``             any pending SPY option BUY order (a multi-leg
                                           parent without a symbol counts), or any open
                                           order on the selected symbol
 3. ``duplicate_symbol``                   the selected symbol is already held
 4. ``max_positions``                      held SPY options plus pending buys on symbols
                                           not held reach ``MAX_OPEN_POSITIONS``
 5. ``market_closed``                      fresh clock says the market is not open
 6. ``too_close_to_close``                 fewer than ``MIN_MINUTES_TO_CLOSE`` minutes left
                                           (``None`` minutes is unknown and also refused)
 7. ``unacceptable_quote``                 the fresh quote failed the selector rules, is
                                           not for the selected symbol, or has no positive
                                           ask in cents
 8. ``unknown_buying_power``               ``options_buying_power`` is ``None``
 9. ``premium_over_cap``                   ``limit_price * 100 * qty > MAX_PREMIUM_USD``
10. ``total_premium_over_cap``             the cost basis of every held SPY option plus
                                           this premium exceeds ``MAX_TOTAL_PREMIUM_USD``
                                           (a held position with no cost basis is refused)
11. ``insufficient_options_buying_power``  premium exceeds ``options_buying_power``

The entry plan: buy to open ``MAX_CONTRACTS`` contract(s), limit at the fresh
ask rounded to the cent, day, client order id ``regimepilot-<cycle_id>-open``.

Exit rules (``decide_exit``), applied in this order, first refusal wins.
They read ONLY the fresh account: the packet's quantity is never trusted.
Entry gates and portfolio caps never block an exit, and neither does the
last half hour of the session (correction 1).

 1. ``no_position``             the fresh account holds no SPY option with this symbol
 2. ``position_mismatch``       the fresh position is not long, or has no quantity of
                                at least one
 3. ``pending_order_conflict``  any open order on this exact symbol (orders on other
                                symbols do not matter)
 4. ``market_closed``           fresh clock says the market is not open
 5. ``unacceptable_quote``      the fresh quote is not for this symbol, has no bid,
                                failed a selector rule other than ``wide_spread``,
                                or has no positive bid in cents

The exit plan: sell to close the whole fresh quantity (capped at
``qty_available`` when the broker reports one), limit at the fresh bid
rounded to the cent, day, client order id ``regimepilot-<cycle_id>-close<n>``.
"""

from __future__ import annotations

from regimepilot.gates import MIN_MINUTES_TO_CLOSE
from regimepilot.models import (
    AccountState,
    ExecutionState,
    OpenOrderSummary,
    OpenPositionContext,
    OrderPlan,
    PortfolioLimits,
    PositionSummary,
    RiskDecision,
    RiskReason,
    SelectionResult,
)

# Approved 2026-08-27. Always one contract; never sized from confidence or equity.
MAX_CONTRACTS = 1

# Approved 2026-08-27. ATM 7-DTE SPY quotes ran about $5.5 (~$550 a contract)
# on 2026-08-26, so this admits a normal trade and refuses anything strange.
MAX_PREMIUM_USD = 1000.0

# Portfolio limits approved 2026-08-27: at most three SPY option positions
# (a pending buy counts as one), one new entry per cycle, and at most $3,000
# of premium at risk across every held contract plus the new one.
MAX_OPEN_POSITIONS = 3
MAX_NEW_ENTRIES_PER_CYCLE = 1
MAX_TOTAL_PREMIUM_USD = 3000.0

DEFAULT_LIMITS = PortfolioLimits(
    max_open_positions=MAX_OPEN_POSITIONS,
    max_new_entries_per_cycle=MAX_NEW_ENTRIES_PER_CYCLE,
    max_entry_premium_usd=MAX_PREMIUM_USD,
    max_total_premium_usd=MAX_TOTAL_PREMIUM_USD,
)

# Prices are sent in whole cents.
PRICE_DECIMALS = 2

__all__ = [
    "DEFAULT_LIMITS",
    "MAX_CONTRACTS",
    "MAX_NEW_ENTRIES_PER_CYCLE",
    "MAX_OPEN_POSITIONS",
    "MAX_PREMIUM_USD",
    "MAX_TOTAL_PREMIUM_USD",
    "MIN_MINUTES_TO_CLOSE",
    "PRICE_DECIMALS",
    "client_order_id_for",
    "decide_exit",
    "decide_order",
]


def client_order_id_for(cycle_id: str, tag: str = "open") -> str:
    """The client order id one action of one cycle may use: ``regimepilot-<cycle_id>-<tag>``."""
    if not cycle_id.strip():
        raise ValueError("cycle_id must not be empty")
    if not tag.strip():
        raise ValueError("tag must not be empty")
    return f"regimepilot-{cycle_id}-{tag}"


def _refuse(reason: RiskReason) -> RiskDecision:
    return RiskDecision(approved=False, reason=reason)


def _held_spy_positions(account: AccountState) -> tuple[PositionSummary, ...]:
    return tuple(p for p in account.positions if p.is_spy_option)


def _pending_spy_orders(account: AccountState) -> tuple[OpenOrderSummary, ...]:
    return tuple(o for o in account.open_orders if o.is_spy_option)


def _orders_on(account: AccountState, symbol: str) -> tuple[OpenOrderSummary, ...]:
    """Every open order on one exact symbol, whatever its side."""
    return tuple(o for o in account.open_orders if o.symbol == symbol)


def _is_pending_buy(order: OpenOrderSummary) -> bool:
    # A multi-leg parent has no symbol; while it is open the portfolio is not
    # fully known, so it is treated as a pending buy.
    return order.symbol is None or (order.side or "").lower() == "buy"


def _fresh_position(account: AccountState, symbol: str) -> PositionSummary | None:
    for position in _held_spy_positions(account):
        if position.symbol == symbol:
            return position
    return None


def decide_order(selection: SelectionResult, state: ExecutionState, *, cycle_id: str) -> RiskDecision:
    """Apply the entry rules and return an approved plan or a refusal. Never raises for a refusal."""
    selected = selection.selected
    if selected is None:
        return _refuse("no_contract")

    account = state.account
    held = _held_spy_positions(account)
    pending_buys = tuple(o for o in _pending_spy_orders(account) if _is_pending_buy(o))
    if pending_buys or _orders_on(account, selected.symbol):
        return _refuse("pending_order_conflict")

    held_symbols = {p.symbol for p in held}
    if selected.symbol in held_symbols:
        return _refuse("duplicate_symbol")

    pending_new = {o.symbol for o in pending_buys if o.symbol is not None and o.symbol not in held_symbols}
    if len(held) + len(pending_new) >= MAX_OPEN_POSITIONS:
        return _refuse("max_positions")

    if state.market_is_open is not True:
        return _refuse("market_closed")
    if state.minutes_to_close is None or state.minutes_to_close < MIN_MINUTES_TO_CLOSE:
        return _refuse("too_close_to_close")

    quote = state.quote
    if quote.reject_reason is not None or quote.symbol != selected.symbol or quote.ask is None:
        return _refuse("unacceptable_quote")
    limit_price = round(quote.ask, PRICE_DECIMALS)
    if limit_price <= 0:
        return _refuse("unacceptable_quote")

    buying_power = account.options_buying_power
    if buying_power is None:
        return _refuse("unknown_buying_power")

    qty = MAX_CONTRACTS
    premium = round(limit_price * 100 * qty, 2)
    if premium > MAX_PREMIUM_USD:
        return _refuse("premium_over_cap")

    bases = [p.cost_basis for p in held]
    if any(basis is None for basis in bases):
        return _refuse("total_premium_over_cap")
    if round(sum(bases) + premium, 2) > MAX_TOTAL_PREMIUM_USD:
        return _refuse("total_premium_over_cap")

    if premium > buying_power:
        return _refuse("insufficient_options_buying_power")

    plan = OrderPlan(
        symbol=selected.symbol,
        qty=qty,
        limit_price=limit_price,
        notional_usd=premium,
        client_order_id=client_order_id_for(cycle_id, "open"),
    )
    return RiskDecision(approved=True, plan=plan)


def decide_exit(
    position: OpenPositionContext,
    state: ExecutionState,
    *,
    cycle_id: str,
    sequence: int = 1,
) -> RiskDecision:
    """Apply the exit rules to one held contract. Reads only the fresh account; never raises for a refusal."""
    symbol = position.symbol
    account = state.account

    fresh = _fresh_position(account, symbol)
    if fresh is None:
        return _refuse("no_position")
    if (fresh.side is not None and fresh.side.lower() != "long") or fresh.qty is None or fresh.qty < 1:
        return _refuse("position_mismatch")

    if _orders_on(account, symbol):
        return _refuse("pending_order_conflict")

    if state.market_is_open is not True:
        return _refuse("market_closed")

    quote = state.quote
    if quote.symbol != symbol or quote.bid is None or quote.reject_reason not in (None, "wide_spread"):
        return _refuse("unacceptable_quote")
    limit_price = round(quote.bid, PRICE_DECIMALS)
    if limit_price <= 0:
        return _refuse("unacceptable_quote")

    qty = int(fresh.qty)
    if fresh.qty_available is not None and fresh.qty_available < qty:
        qty = int(fresh.qty_available)
    if qty < 1:
        return _refuse("position_mismatch")

    plan = OrderPlan(
        symbol=symbol,
        side="sell",
        position_intent="sell_to_close",
        qty=qty,
        limit_price=limit_price,
        notional_usd=round(limit_price * 100 * qty, 2),
        client_order_id=client_order_id_for(cycle_id, f"close{sequence}"),
    )
    return RiskDecision(approved=True, plan=plan)
