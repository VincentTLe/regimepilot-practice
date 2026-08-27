"""Deterministic risk decision for the MVP (methodology approved 2026-08-27).

Pure module: no network, no vendor SDK, no LLM. It takes what Phase 4 chose
(``SelectionResult``) and what ``execution.observe_execution_state`` re-read
right before ordering (``ExecutionState``), and answers "may one order be
built, and which?" with rules a person can check by hand. The LLM never
reaches this module: quantity, limit price, order type and time-in-force are
fixed here and nowhere else.

Approved rules, applied in this order (first refusal wins):

1. ``no_contract``                        selection carries no ``SelectedContract``
2. ``existing_spy_option_position``       account already holds a SPY option
3. ``existing_spy_option_order``          account has an open SPY option order
4. ``market_closed``                      fresh clock says the market is not open
5. ``too_close_to_close``                 fewer than ``MIN_MINUTES_TO_CLOSE`` minutes left
                                          (``None`` minutes is unknown and also refused)
6. ``unacceptable_quote``                 the fresh quote failed the selector rules
                                          (``FreshQuote.reject_reason`` is set), is not for
                                          the selected symbol, or has no positive ask in cents
7. ``unknown_buying_power``               ``options_buying_power`` is ``None``
8. ``premium_over_cap``                   ``limit_price * 100 * qty > MAX_PREMIUM_USD``
9. ``insufficient_options_buying_power``  premium exceeds ``options_buying_power``

Otherwise the plan is: buy to open ``MAX_CONTRACTS`` contracts of the selected
symbol, limit at the fresh ask rounded to the cent, day, client order id
``regimepilot-<cycle_id>``.
"""

from __future__ import annotations

from regimepilot.gates import MIN_MINUTES_TO_CLOSE
from regimepilot.models import ExecutionState, OrderPlan, RiskDecision, RiskReason, SelectionResult

# Approved 2026-08-27. Always one contract; never sized from confidence or equity.
MAX_CONTRACTS = 1

# Approved 2026-08-27. ATM 7-DTE SPY quotes ran about $5.5 (~$550 a contract)
# on 2026-08-26, so this admits a normal trade and refuses anything strange.
MAX_PREMIUM_USD = 1000.0

# Prices are sent in whole cents.
PRICE_DECIMALS = 2

__all__ = [
    "MAX_CONTRACTS",
    "MAX_PREMIUM_USD",
    "MIN_MINUTES_TO_CLOSE",
    "PRICE_DECIMALS",
    "client_order_id_for",
    "decide_order",
]


def client_order_id_for(cycle_id: str) -> str:
    """The client order id one cycle may use. One cycle, one id, at most one order."""
    if not cycle_id.strip():
        raise ValueError("cycle_id must not be empty")
    return f"regimepilot-{cycle_id}"


def _refuse(reason: RiskReason) -> RiskDecision:
    return RiskDecision(approved=False, reason=reason)


def decide_order(
    selection: SelectionResult,
    state: ExecutionState,
    *,
    cycle_id: str,
) -> RiskDecision:
    """Apply the approved rules and return an approved plan or a refusal.

    Never raises for a refusal: a refusal is a normal outcome carrying its
    reason. ``ValueError`` from the models means a programming error here.
    """
    selected = selection.selected
    if selected is None:
        return _refuse("no_contract")

    account = state.account
    if account.has_open_spy_option_position:
        return _refuse("existing_spy_option_position")
    if account.has_open_spy_option_order:
        return _refuse("existing_spy_option_order")

    if state.market_is_open is not True:
        return _refuse("market_closed")

    if state.minutes_to_close is None or state.minutes_to_close < MIN_MINUTES_TO_CLOSE:
        return _refuse("too_close_to_close")

    # The fresh quote must be for the contract that was selected, must have
    # passed the selector rules again, and must have an ask that is still a
    # positive price once rounded to the cent it will be sent as.
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
    if premium > buying_power:
        return _refuse("insufficient_options_buying_power")

    plan = OrderPlan(
        symbol=selected.symbol,
        qty=qty,
        limit_price=limit_price,
        max_premium_usd=premium,
        client_order_id=client_order_id_for(cycle_id),
    )
    return RiskDecision(approved=True, plan=plan)
