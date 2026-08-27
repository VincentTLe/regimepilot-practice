"""Risk tests: the deterministic order decision, offline.

No fake client is needed: the risk module is pure, so every test builds the
selection and execution state it wants and asserts on the decision. Boundary
values are the approved thresholds themselves (cap, buying power, minutes to
close), and every refusal reason is exercised once.
"""

import socket
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from regimepilot import risk as risk_module
from regimepilot.features import spread_bps
from regimepilot.models import (
    AccountState,
    ExecutionState,
    FreshQuote,
    OpenOrderSummary,
    OrderPlan,
    PositionSummary,
    RiskDecision,
    SelectedContract,
    SelectionResult,
)
from regimepilot.risk import (
    MAX_CONTRACTS,
    MAX_PREMIUM_USD,
    MIN_MINUTES_TO_CLOSE,
    client_order_id_for,
    decide_order,
)

# Selection at 10:35 New York on Wednesday 2026-08-26; the re-check ran a
# few seconds later, with the fresh quote stamped one second before the
# server clock was read.
NOW = datetime(2026, 8, 26, 14, 35, tzinfo=timezone.utc)
RECHECK_AT = NOW + timedelta(seconds=5)
EXPIRATION = date(2026, 9, 2)
SYMBOL = "SPY260902C00765000"
OTHER_SYMBOL = "SPY260902C00766000"
CYCLE_ID = "20260826T143500Z"


def selected_contract(*, symbol=SYMBOL, bid=5.40, ask=5.49):
    return SelectedContract(
        symbol=symbol,
        option_type="call",
        strike_price=765.0,
        expiration_date=EXPIRATION,
        days_to_expiration=7,
        bid=bid,
        ask=ask,
        mid=(bid + ask) / 2,
        spread_bps=spread_bps(bid, ask),
        quote_at=NOW - timedelta(seconds=1),
        quote_age_seconds=1.0,
        underlying_mid=765.0,
    )


def selection(*, selected=...):
    """A selection that chose one contract; ``selected=None`` models a no-contract answer."""
    if selected is ...:
        selected = selected_contract()
    if selected is None:
        return SelectionResult(
            observed_at=NOW,
            action="BUY_CALL",
            status="no_contract",
            reason="all_candidates_rejected",
            target_expiration=EXPIRATION,
        )
    return SelectionResult(
        observed_at=NOW,
        action="BUY_CALL",
        status="selected",
        target_expiration=EXPIRATION,
        selected=selected,
    )


def account(*, options_buying_power=50_000.0, position=False, order=False):
    """An empty paper account unless a SPY option position or open order is asked for."""
    positions = ()
    if position:
        positions = (
            PositionSummary(
                symbol=SYMBOL, asset_class="us_option", side="long", qty=1.0, is_spy_option=True
            ),
        )
    orders = ()
    if order:
        orders = (
            OpenOrderSummary(
                order_id="aaaabbbb-cccc-dddd-eeee-ffff00001111",
                symbol=SYMBOL,
                asset_class="us_option",
                side="buy",
                qty=1.0,
                status="new",
                is_spy_option=True,
            ),
        )
    return AccountState(
        observed_at=RECHECK_AT,
        account_id_masked="1111****8888",
        equity=100_000.0,
        options_buying_power=options_buying_power,
        positions=positions,
        open_orders=orders,
        has_open_spy_option_position=position,
        has_open_spy_option_order=order,
    )


def fresh_quote(*, symbol=SYMBOL, bid=5.40, ask=5.49, reject_reason=None):
    return FreshQuote(
        symbol=symbol,
        bid=bid,
        ask=ask,
        quote_at=RECHECK_AT - timedelta(seconds=1),
        server_time=RECHECK_AT,
        reject_reason=reject_reason,
    )


def state(*, account_state=None, market_is_open=True, minutes_to_close=120.0, quote=None):
    return ExecutionState(
        observed_at=RECHECK_AT,
        account=account() if account_state is None else account_state,
        market_is_open=market_is_open,
        minutes_to_close=minutes_to_close,
        quote=fresh_quote() if quote is None else quote,
    )


def decide(selection_result=None, execution_state=None, *, cycle_id=CYCLE_ID):
    return decide_order(
        selection() if selection_result is None else selection_result,
        state() if execution_state is None else execution_state,
        cycle_id=cycle_id,
    )


# --------------------------------------------------------------------------
# 1. the client order id ties one cycle to at most one order
# --------------------------------------------------------------------------


def test_client_order_id_is_the_cycle_id_with_a_fixed_prefix():
    assert client_order_id_for(CYCLE_ID) == f"regimepilot-{CYCLE_ID}"


@pytest.mark.parametrize("cycle_id", ["", "   "])
def test_client_order_id_refuses_an_empty_cycle_id(cycle_id):
    with pytest.raises(ValueError):
        client_order_id_for(cycle_id)


# --------------------------------------------------------------------------
# 2. the normal path: one contract, limit at the fresh ask, day, buy to open
# --------------------------------------------------------------------------


def test_normal_path_approves_one_contract_at_the_fresh_ask():
    decision = decide()

    assert decision.approved is True
    assert decision.reason is None
    plan = decision.plan
    assert plan is not None
    assert plan.symbol == SYMBOL
    assert plan.qty == MAX_CONTRACTS == 1
    assert plan.limit_price == 5.49
    assert plan.max_premium_usd == 549.0
    assert plan.client_order_id == f"regimepilot-{CYCLE_ID}"
    # Fixed by methodology, never passed in: buy to open, limit, day.
    assert plan.side == "buy"
    assert plan.order_type == "limit"
    assert plan.time_in_force == "day"
    assert plan.position_intent == "buy_to_open"


def test_the_plan_round_trips_through_json():
    plan = decide().plan
    assert OrderPlan.model_validate_json(plan.model_dump_json()) == plan


def test_the_limit_price_is_the_fresh_ask_not_the_selection_ask():
    chosen = selected_contract(bid=4.90, ask=5.00)
    fresh = state(quote=fresh_quote(bid=5.40, ask=5.49))
    plan = decide(selection(selected=chosen), fresh).plan
    assert plan.limit_price == 5.49
    assert plan.max_premium_usd == 549.0


def test_the_ask_is_rounded_to_the_cent():
    plan = decide(execution_state=state(quote=fresh_quote(ask=5.4949))).plan
    assert plan.limit_price == 5.49
    assert plan.max_premium_usd == 549.0


def test_quantity_is_always_one_whatever_the_buying_power():
    plan = decide(execution_state=state(account_state=account(options_buying_power=1_000_000.0))).plan
    assert plan.qty == 1


# --------------------------------------------------------------------------
# 3. every refusal reason, once
# --------------------------------------------------------------------------


REFUSALS = [
    ("no_contract", lambda: (selection(selected=None), state())),
    ("existing_spy_option_position", lambda: (selection(), state(account_state=account(position=True)))),
    ("existing_spy_option_order", lambda: (selection(), state(account_state=account(order=True)))),
    ("market_closed", lambda: (selection(), state(market_is_open=False))),
    ("market_closed", lambda: (selection(), state(market_is_open=None))),
    ("too_close_to_close", lambda: (selection(), state(minutes_to_close=MIN_MINUTES_TO_CLOSE - 0.1))),
    ("too_close_to_close", lambda: (selection(), state(minutes_to_close=None))),
    ("unacceptable_quote", lambda: (selection(), state(quote=fresh_quote(reject_reason="stale_quote")))),
    ("unacceptable_quote", lambda: (selection(), state(quote=fresh_quote(reject_reason="wide_spread")))),
    ("unacceptable_quote", lambda: (selection(), state(quote=fresh_quote(bid=None, ask=None)))),
    ("unacceptable_quote", lambda: (selection(), state(quote=fresh_quote(bid=0.0, ask=0.0)))),
    ("unacceptable_quote", lambda: (selection(), state(quote=fresh_quote(bid=0.001, ask=0.004)))),
    ("unacceptable_quote", lambda: (selection(), state(quote=fresh_quote(symbol=OTHER_SYMBOL)))),
    ("unknown_buying_power", lambda: (selection(), state(account_state=account(options_buying_power=None)))),
    ("premium_over_cap", lambda: (selection(), state(quote=fresh_quote(bid=9.90, ask=10.01)))),
    ("insufficient_options_buying_power", lambda: (selection(), state(account_state=account(options_buying_power=548.99)))),
]


@pytest.mark.parametrize("reason, build", REFUSALS, ids=[r for r, _ in REFUSALS])
def test_each_rule_refuses_with_its_reason_and_no_plan(reason, build):
    decision = decide(*build())
    assert decision.approved is False
    assert decision.reason == reason
    assert decision.plan is None


# --------------------------------------------------------------------------
# 4. boundaries: the thresholds themselves are allowed
# --------------------------------------------------------------------------


def test_a_premium_exactly_at_the_cap_is_allowed():
    decision = decide(execution_state=state(quote=fresh_quote(bid=9.90, ask=10.00)))
    assert decision.approved is True
    assert decision.plan.max_premium_usd == MAX_PREMIUM_USD == 1000.0


def test_a_premium_exactly_equal_to_buying_power_is_allowed():
    decision = decide(execution_state=state(account_state=account(options_buying_power=549.0)))
    assert decision.approved is True


def test_exactly_the_minimum_minutes_to_close_is_allowed():
    decision = decide(execution_state=state(minutes_to_close=float(MIN_MINUTES_TO_CLOSE)))
    assert decision.approved is True


# --------------------------------------------------------------------------
# 5. first refusal wins
# --------------------------------------------------------------------------


def test_the_first_failing_rule_is_reported():
    failing_several = state(
        account_state=account(position=True, options_buying_power=None),
        market_is_open=False,
        minutes_to_close=None,
        quote=fresh_quote(reject_reason="stale_quote", symbol=OTHER_SYMBOL),
    )
    assert decide(execution_state=failing_several).reason == "existing_spy_option_position"
    assert decide(selection(selected=None), failing_several).reason == "no_contract"


# --------------------------------------------------------------------------
# 6. a decision can never be both refused and carry a plan
# --------------------------------------------------------------------------


def test_the_decision_model_refuses_an_inconsistent_decision():
    plan = decide().plan
    with pytest.raises(ValueError):
        RiskDecision(approved=True)
    with pytest.raises(ValueError):
        RiskDecision(approved=False, reason="market_closed", plan=plan)
    with pytest.raises(ValueError):
        RiskDecision(approved=False)


# --------------------------------------------------------------------------
# 7. pure and read-only
# --------------------------------------------------------------------------


def test_the_decision_makes_no_network_call(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("the risk module must not open a socket")

    monkeypatch.setattr(socket, "socket", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)

    assert decide().approved is True


def test_the_risk_module_never_imports_the_vendor_sdk():
    source = Path(risk_module.__file__).read_text(encoding="utf-8")
    assert "alpaca" not in source.lower()

    for value in vars(risk_module).values():
        module = getattr(value, "__module__", "") or ""
        assert not module.startswith("alpaca")


def test_the_risk_module_exposes_no_submission_helper():
    forbidden = ("submit", "cancel", "replace", "close_position", "close_all", "exercise", "place_")
    offenders = [name for name in dir(risk_module) if any(word in name.lower() for word in forbidden)]
    assert offenders == []
