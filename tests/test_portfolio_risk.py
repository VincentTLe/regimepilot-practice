"""Portfolio risk tests: entries beside held positions, and the exit decision.

Pure: no network, no SDK. Every state is built by hand from the frozen models.
"""

from datetime import date, datetime, timedelta, timezone

import pytest

from regimepilot.models import (
    AccountState,
    ExecutionState,
    FreshQuote,
    OpenOrderSummary,
    OpenPositionContext,
    PositionSummary,
    SelectedContract,
    SelectionResult,
)
from regimepilot.risk import (
    MAX_OPEN_POSITIONS,
    MAX_TOTAL_PREMIUM_USD,
    decide_exit,
    decide_order,
)

NOW = datetime(2026, 8, 26, 14, 35, tzinfo=timezone.utc)
EXPIRATION = date(2026, 9, 2)
CYCLE_ID = "20260826-143500"

SYMBOL = "SPY260902C00765000"
OTHER = "SPY260902C00766000"
THIRD = "SPY260902C00767000"
FOURTH = "SPY260902C00768000"


def held(symbol=SYMBOL, *, qty=1.0, side="long", cost_basis=549.0, qty_available=None):
    return PositionSummary(
        symbol=symbol, asset_class="us_option", side=side, qty=qty, is_spy_option=True,
        avg_entry_price=None if cost_basis is None or qty in (None, 0) else cost_basis / (100 * qty),
        cost_basis=cost_basis, qty_available=qty_available,
    )


def pending(symbol=SYMBOL, *, side="buy", order_id="order-1"):
    return OpenOrderSummary(
        order_id=order_id, symbol=symbol, asset_class="us_option", side=side, qty=1.0, status="new",
        is_spy_option=True,
    )


def account(*, positions=(), orders=(), options_buying_power=50_000.0):
    return AccountState(
        observed_at=NOW, account_id_masked="****8888", equity=100_000.0,
        options_buying_power=options_buying_power, positions=tuple(positions), open_orders=tuple(orders),
        has_open_spy_option_position=any(p.is_spy_option for p in positions),
        has_open_spy_option_order=any(o.is_spy_option for o in orders),
    )


def quote(symbol=SYMBOL, *, bid=5.44, ask=5.49, reject_reason=None):
    return FreshQuote(symbol=symbol, bid=bid, ask=ask, quote_at=NOW - timedelta(seconds=1), server_time=NOW,
                      reject_reason=reject_reason)


def state(account_state=None, *, market_is_open=True, minutes_to_close=120.0, fresh=None):
    return ExecutionState(
        observed_at=NOW, account=account() if account_state is None else account_state,
        market_is_open=market_is_open, minutes_to_close=minutes_to_close, quote=quote() if fresh is None else fresh,
    )


def selection(symbol=SYMBOL):
    contract = SelectedContract(
        symbol=symbol, option_type="call", strike_price=765.0, expiration_date=EXPIRATION, days_to_expiration=7,
        bid=5.44, ask=5.49, mid=5.465, spread_bps=91.5, quote_at=NOW - timedelta(seconds=3), quote_age_seconds=3.0,
        underlying_mid=765.0,
    )
    return SelectionResult(observed_at=NOW, action="BUY_CALL", status="selected", selected=contract)


def position(symbol=SYMBOL, *, qty=1.0):
    return OpenPositionContext(symbol=symbol, option_type="call", strike_price=765.0, expiration_date=EXPIRATION,
                               days_to_expiration=7, qty=qty)


# --------------------------------------------------------------------------
# entries beside held positions
# --------------------------------------------------------------------------


def test_a_held_position_on_another_symbol_no_longer_blocks_an_entry():
    decision = decide_order(selection(), state(account(positions=[held(OTHER)])), cycle_id=CYCLE_ID)

    assert decision.approved is True
    assert decision.plan.client_order_id == f"regimepilot-{CYCLE_ID}-open"


def test_the_selected_symbol_already_held_is_a_duplicate():
    decision = decide_order(selection(), state(account(positions=[held(SYMBOL)])), cycle_id=CYCLE_ID)
    assert decision.reason == "duplicate_symbol"


def test_the_third_position_is_the_last_one_allowed():
    two = account(positions=[held(OTHER), held(THIRD)])
    three = account(positions=[held(OTHER), held(THIRD), held(FOURTH)])

    assert decide_order(selection(), state(two), cycle_id=CYCLE_ID).approved is True
    assert decide_order(selection(), state(three), cycle_id=CYCLE_ID).reason == "max_positions"
    assert MAX_OPEN_POSITIONS == 3


def test_a_pending_buy_blocks_an_entry_but_a_pending_sell_elsewhere_does_not():
    pending_buy = account(positions=[held(OTHER)], orders=[pending(THIRD, side="buy")])
    pending_sell = account(positions=[held(OTHER)], orders=[pending(OTHER, side="sell")])

    assert decide_order(selection(), state(pending_buy), cycle_id=CYCLE_ID).reason == "pending_order_conflict"
    assert decide_order(selection(), state(pending_sell), cycle_id=CYCLE_ID).approved is True


def test_a_multi_leg_parent_without_a_symbol_counts_as_a_pending_buy():
    parent = OpenOrderSummary(order_id="mleg", symbol=None, side="buy", qty=1.0, status="new", is_spy_option=True)
    decision = decide_order(selection(), state(account(orders=[parent])), cycle_id=CYCLE_ID)
    assert decision.reason == "pending_order_conflict"


def test_the_total_premium_cap_counts_every_held_cost_basis():
    over = account(positions=[held(OTHER, cost_basis=2500.0)])  # 2500 + 549 > 3000
    under = account(positions=[held(OTHER, cost_basis=2400.0)])  # 2949
    exact = account(positions=[held(OTHER, cost_basis=2451.0)])  # 3000.0
    unknown = account(positions=[held(OTHER, cost_basis=None)])

    assert decide_order(selection(), state(over), cycle_id=CYCLE_ID).reason == "total_premium_over_cap"
    assert decide_order(selection(), state(under), cycle_id=CYCLE_ID).approved is True
    assert decide_order(selection(), state(exact), cycle_id=CYCLE_ID).approved is True
    assert decide_order(selection(), state(unknown), cycle_id=CYCLE_ID).reason == "total_premium_over_cap"
    assert MAX_TOTAL_PREMIUM_USD == 3000.0


# --------------------------------------------------------------------------
# the exit decision
# --------------------------------------------------------------------------


def test_an_exit_sells_the_whole_fresh_quantity_at_the_fresh_bid():
    decision = decide_exit(position(), state(account(positions=[held()])), cycle_id=CYCLE_ID)

    assert decision.approved is True
    plan = decision.plan
    assert (plan.side, plan.position_intent, plan.order_type, plan.time_in_force) == (
        "sell", "sell_to_close", "limit", "day",
    )
    assert (plan.symbol, plan.qty, plan.limit_price, plan.notional_usd) == (SYMBOL, 1, 5.44, 544.0)
    assert plan.client_order_id == f"regimepilot-{CYCLE_ID}-close1"
    second = decide_exit(position(), state(account(positions=[held()])), cycle_id=CYCLE_ID, sequence=2)
    assert second.plan.client_order_id == f"regimepilot-{CYCLE_ID}-close2"


def test_the_fresh_account_decides_the_quantity_never_the_packet():
    fresh_two = account(positions=[held(qty=2.0)])
    capped = account(positions=[held(qty=2.0, qty_available=1.0)])
    nothing_available = account(positions=[held(qty=2.0, qty_available=0.0)])

    assert decide_exit(position(qty=1.0), state(fresh_two), cycle_id=CYCLE_ID).plan.qty == 2
    assert decide_exit(position(qty=5.0), state(capped), cycle_id=CYCLE_ID).plan.qty == 1
    assert decide_exit(position(), state(nothing_available), cycle_id=CYCLE_ID).reason == "position_mismatch"


@pytest.mark.parametrize(
    "fresh_account, reason",
    [
        (account(), "no_position"),
        (account(positions=[held(OTHER)]), "no_position"),
        (account(positions=[held(side="short")]), "position_mismatch"),
        (account(positions=[held(qty=0.0)]), "position_mismatch"),
        (account(positions=[held()], orders=[pending(SYMBOL, side="sell")]), "pending_order_conflict"),
        (account(positions=[held()], orders=[pending(SYMBOL, side="buy")]), "pending_order_conflict"),
    ],
    ids=["empty", "other-symbol", "short", "zero-qty", "pending-sell-here", "pending-buy-here"],
)
def test_exit_refusals_from_the_fresh_account(fresh_account, reason):
    decision = decide_exit(position(), state(fresh_account), cycle_id=CYCLE_ID)
    assert decision.approved is False and decision.reason == reason and decision.plan is None


def test_a_pending_order_on_another_symbol_does_not_block_this_exit():
    fresh = account(positions=[held(), held(OTHER)], orders=[pending(OTHER, side="sell")])
    assert decide_exit(position(), state(fresh), cycle_id=CYCLE_ID).approved is True


def test_an_exit_needs_an_open_market_but_never_thirty_minutes():
    """Correction 1: too_close_to_close blocks entries, never exits."""
    fresh = account(positions=[held()])
    assert decide_exit(position(), state(fresh, market_is_open=False), cycle_id=CYCLE_ID).reason == "market_closed"
    assert decide_exit(position(), state(fresh, market_is_open=None), cycle_id=CYCLE_ID).reason == "market_closed"
    assert decide_exit(position(), state(fresh, minutes_to_close=3.0), cycle_id=CYCLE_ID).approved is True
    assert decide_exit(position(), state(fresh, minutes_to_close=None), cycle_id=CYCLE_ID).approved is True


@pytest.mark.parametrize(
    "fresh_quote, approved",
    [
        (quote(reject_reason="stale_quote"), False),
        (quote(reject_reason="invalid_quote"), False),
        (quote(reject_reason="no_quote"), False),
        (quote(reject_reason="wide_spread"), True),
        (quote(bid=None), False),
        (quote(bid=0.004, ask=0.01), False),
        (quote(symbol=OTHER), False),
    ],
    ids=["stale", "invalid", "none", "wide-spread-allowed", "no-bid", "sub-cent", "other-symbol"],
)
def test_exit_quote_rules(fresh_quote, approved):
    decision = decide_exit(position(), state(account(positions=[held()]), fresh=fresh_quote), cycle_id=CYCLE_ID)
    assert decision.approved is approved
    if not approved:
        assert decision.reason == "unacceptable_quote"


def test_an_exit_ignores_every_entry_cap():
    crowded = account(
        positions=[held(cost_basis=3000.0), held(OTHER, cost_basis=3000.0), held(THIRD, cost_basis=3000.0)],
        options_buying_power=None,
    )
    decision = decide_exit(position(), state(crowded), cycle_id=CYCLE_ID)
    assert decision.approved is True
    assert decision.plan.model_validate_json(decision.plan.model_dump_json()) == decision.plan
