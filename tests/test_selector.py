"""Selector tests: Phase 4B chooses one contract from a ChainPacket, offline.

No fake client is needed: the selector is pure, so every test builds the
packet it wants and asserts on the verdicts. Boundary values are the approved
thresholds themselves.
"""

import json
import socket
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from regimepilot import selector as selector_module
from regimepilot.features import spread_bps
from regimepilot.models import (
    ChainPacket,
    ContractCandidate,
    SelectedContract,
    SelectionResult,
)
from regimepilot.selector import (
    DTE_TARGET_DAYS,
    MAX_QUOTE_AGE_SECONDS,
    MAX_SPREAD_BPS,
    choose_expiration,
    format_summary,
    judge_candidate,
    main,
    select_contract,
)

# The chain was observed at 10:35 New York on Wednesday 2026-08-26 (local
# clock); the server clock was read two seconds later, right after the quotes.
TODAY = date(2026, 8, 26)
OBSERVED_AT = datetime(2026, 8, 26, 14, 35, tzinfo=timezone.utc)
READ_AT = OBSERVED_AT + timedelta(seconds=2)
MID = 765.0

# A cent-priced pair that is exactly 3.50% wide: 0.07 on a 2.00 mid. Floating
# point puts spread_bps a few 1e-13 above 350; the rule must still read it as 350.
AT_LIMIT = (1.965, 2.035)
OVER_LIMIT = (2.525, 2.615)  # ~350.19 bps


def candidate(
    *,
    strike=765.0,
    dte=7,
    option_type="call",
    bid=4.90,
    ask=5.00,
    age=1.0,
    tradable=True,
    status="active",
    quote_at=...,
    symbol=None,
):
    """One observed contract. ``dte=None`` models a row without an expiration."""
    expiration = None if dte is None else TODAY + timedelta(days=dte)
    if quote_at is ...:
        quote_at = READ_AT - timedelta(seconds=age)
    if symbol is None:
        letter = "C" if option_type == "call" else "P"
        stamp = "000000" if expiration is None else expiration.strftime("%y%m%d")
        strike_code = 0 if strike is None else int(round(strike * 1000))
        symbol = f"SPY{stamp}{letter}{strike_code:08d}"
    return ContractCandidate(
        symbol=symbol,
        option_type=option_type,
        strike_price=strike,
        expiration_date=expiration,
        days_to_expiration=dte,
        status=status,
        tradable=tradable,
        bid=bid,
        ask=ask,
        quote_at=quote_at,
    )


def ladder(dte, strikes, option_type="call", **overrides):
    return [candidate(strike=float(k), dte=dte, option_type=option_type, **overrides) for k in strikes]


def packet(action="BUY_CALL", *, mid=MID, candidates=(), quotes_read_at=READ_AT, observed_at=OBSERVED_AT):
    return ChainPacket(
        observed_at=observed_at,
        action=action,
        option_feed="indicative",
        underlying_mid=mid,
        quotes_read_at=quotes_read_at,
        candidates=tuple(candidates),
    )


# --------------------------------------------------------------------------
# 1. outcomes that need no judging
# --------------------------------------------------------------------------


def test_hold_is_not_applicable_and_judges_nothing():
    result = select_contract(packet("HOLD", mid=None))

    assert isinstance(result, SelectionResult)
    assert result.status == "not_applicable"
    assert result.reason is None
    assert result.selected is None
    assert result.verdicts == ()
    assert result.target_expiration is None
    assert result.action == "HOLD"
    assert result.observed_at == OBSERVED_AT


def test_a_missing_underlying_mid_is_no_contract():
    result = select_contract(packet(mid=None, candidates=ladder(7, [765])))

    assert (result.status, result.reason) == ("no_contract", "no_underlying_price")
    assert result.selected is None
    assert result.verdicts == ()


def test_an_empty_chain_is_no_contract():
    result = select_contract(packet(candidates=()))

    assert (result.status, result.reason) == ("no_contract", "no_candidates")
    assert result.selected is None


# --------------------------------------------------------------------------
# 2. the happy path: nearest expiration to seven days, nearest strike to the mid
# --------------------------------------------------------------------------


@pytest.mark.parametrize("action, option_type", [("BUY_CALL", "call"), ("BUY_PUT", "put")])
def test_picks_the_strike_nearest_the_mid_at_the_expiration_nearest_seven_days(action, option_type):
    candidates = (
        ladder(6, [764, 765, 766], option_type)
        + ladder(7, [763, 764, 765, 766, 767], option_type)
        + ladder(9, [765], option_type)
    )
    result = select_contract(packet(action, mid=765.4, candidates=candidates))

    assert result.status == "selected"
    assert result.reason is None
    assert result.target_expiration == TODAY + timedelta(days=7)

    chosen = result.selected
    assert isinstance(chosen, SelectedContract)
    assert chosen.strike_price == 765.0
    assert chosen.days_to_expiration == 7
    assert chosen.expiration_date == TODAY + timedelta(days=7)
    assert chosen.option_type == option_type
    assert (chosen.bid, chosen.ask) == (4.90, 5.00)
    assert chosen.mid == pytest.approx(4.95)
    assert chosen.spread_bps == pytest.approx(spread_bps(4.90, 5.00))
    assert chosen.quote_at == READ_AT - timedelta(seconds=1)
    assert chosen.quote_age_seconds == pytest.approx(1.0)
    assert chosen.underlying_mid == 765.4

    # Verdicts cover the target expiration only, in ATM-rank order; the first
    # eligible one is the pick.
    assert [v.strike_price for v in result.verdicts] == [765.0, 766.0, 764.0, 767.0, 763.0]
    assert all(v.reject_reason is None for v in result.verdicts)
    assert result.verdicts[0].symbol == chosen.symbol
    assert {v.days_to_expiration for v in result.verdicts} == {7}


# --------------------------------------------------------------------------
# 3. ties go to the in-the-money side
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "action, option_type, expected",
    [("BUY_CALL", "call", 765.0), ("BUY_PUT", "put", 766.0)],
)
def test_an_exact_half_tie_goes_to_the_in_the_money_side(action, option_type, expected):
    # A mid computed from cent prices, the way the chain observer computes it.
    mid = (765.13 + 765.87) / 2
    result = select_contract(packet(action, mid=mid, candidates=ladder(7, [765, 766], option_type)))

    assert result.selected.strike_price == expected


@pytest.mark.parametrize(
    "action, option_type, expected",
    [("BUY_CALL", "call", 764.0), ("BUY_PUT", "put", 766.0)],
)
def test_a_tie_after_the_atm_strike_is_rejected_also_goes_in_the_money(action, option_type, expected):
    wide_atm = candidate(strike=765.0, dte=7, option_type=option_type, bid=OVER_LIMIT[0], ask=OVER_LIMIT[1])
    result = select_contract(packet(action, candidates=ladder(7, [764, 766], option_type) + [wide_atm]))

    assert result.status == "selected"
    assert result.selected.strike_price == expected
    atm = next(v for v in result.verdicts if v.strike_price == 765.0)
    assert atm.reject_reason == "wide_spread"


# --------------------------------------------------------------------------
# 4. the expiration is chosen by identity, and never abandoned
# --------------------------------------------------------------------------


def test_the_expiration_nearest_seven_days_wins():
    assert choose_expiration(ladder(5, [765]) + ladder(6, [765]) + ladder(9, [765])) == TODAY + timedelta(days=6)


def test_an_equidistant_expiration_tie_goes_later():
    assert choose_expiration(ladder(6, [765]) + ladder(8, [765])) == TODAY + timedelta(days=8)


def test_candidates_without_an_expiration_do_not_vote_and_are_not_judged():
    dated = candidate(strike=765.0, dte=9)
    result = select_contract(packet(candidates=[candidate(dte=None), dated]))

    assert result.target_expiration == TODAY + timedelta(days=9)
    assert result.selected.symbol == dated.symbol
    assert [v.symbol for v in result.verdicts] == [dated.symbol]


def test_when_no_candidate_has_an_expiration_everything_is_invalid():
    result = select_contract(packet(candidates=[candidate(dte=None), candidate(dte=None, strike=766.0)]))

    assert (result.status, result.reason) == ("no_contract", "all_candidates_rejected")
    assert result.target_expiration is None
    assert [v.reject_reason for v in result.verdicts] == ["invalid_contract", "invalid_contract"]


def test_the_target_expiration_is_never_abandoned():
    """Every 7-DTE quote is too wide while 6-DTE is fine: no contract, no jump."""
    too_wide = ladder(7, [764, 765, 766], bid=OVER_LIMIT[0], ask=OVER_LIMIT[1])
    result = select_contract(packet(candidates=too_wide + ladder(6, [765])))

    assert (result.status, result.reason) == ("no_contract", "all_candidates_rejected")
    assert result.selected is None
    assert result.target_expiration == TODAY + timedelta(days=7)
    assert {v.days_to_expiration for v in result.verdicts} == {7}
    assert {v.reject_reason for v in result.verdicts} == {"wide_spread"}


# --------------------------------------------------------------------------
# 5. every reject reason, in rule order
# --------------------------------------------------------------------------


REJECTIONS = [
    pytest.param(dict(strike=None), "invalid_contract", id="no-strike"),
    pytest.param(dict(option_type="put"), "invalid_contract", id="wrong-type"),
    pytest.param(dict(dte=0), "invalid_contract", id="zero-dte"),
    pytest.param(dict(tradable=False), "not_tradable", id="not-tradable"),
    pytest.param(dict(tradable=None), "not_tradable", id="tradable-unknown"),
    pytest.param(dict(status="inactive"), "not_tradable", id="inactive"),
    pytest.param(dict(bid=None), "no_quote", id="no-bid"),
    pytest.param(dict(ask=None), "no_quote", id="no-ask"),
    pytest.param(dict(quote_at=None), "no_quote", id="no-stamp"),
    pytest.param(dict(bid=0.0), "invalid_quote", id="zero-bid"),
    pytest.param(dict(bid=5.00, ask=4.90), "invalid_quote", id="crossed"),
    pytest.param(dict(bid=float("nan")), "invalid_quote", id="nan-bid"),
    pytest.param(dict(quote_at=READ_AT + timedelta(milliseconds=1)), "invalid_quote", id="future-stamp"),
    pytest.param(dict(age=MAX_QUOTE_AGE_SECONDS + 0.01), "stale_quote", id="stale"),
    pytest.param(dict(bid=OVER_LIMIT[0], ask=OVER_LIMIT[1]), "wide_spread", id="wide"),
]


@pytest.mark.parametrize("overrides, reason", REJECTIONS)
def test_each_reject_reason(overrides, reason):
    result = select_contract(packet(candidates=[candidate(**overrides)]))

    assert (result.status, result.reason) == ("no_contract", "all_candidates_rejected")
    assert [v.reject_reason for v in result.verdicts] == [reason]


def test_rules_are_applied_in_a_fixed_order():
    """A candidate failing several rules reports the first one only."""
    assert judge_candidate(candidate(tradable=False, bid=None, age=99.0), option_type="call", reference=READ_AT) == "not_tradable"
    assert judge_candidate(candidate(bid=None, age=99.0), option_type="call", reference=READ_AT) == "no_quote"
    # A future stamp is invalid, never "fresh": it is caught before the age rule.
    future = candidate(quote_at=READ_AT + timedelta(seconds=30))
    assert judge_candidate(future, option_type="call", reference=READ_AT) == "invalid_quote"
    assert judge_candidate(candidate(), option_type="call", reference=READ_AT) is None


# --------------------------------------------------------------------------
# 6. the thresholds are inclusive at exactly the approved value
# --------------------------------------------------------------------------


def test_a_quote_exactly_ten_seconds_old_is_accepted():
    result = select_contract(packet(candidates=[candidate(age=float(MAX_QUOTE_AGE_SECONDS))]))

    assert result.status == "selected"
    assert result.selected.quote_age_seconds == pytest.approx(10.0)


def test_a_spread_of_exactly_350_bps_is_accepted():
    bid, ask = AT_LIMIT
    assert spread_bps(bid, ask) == pytest.approx(350.0, abs=1e-9)  # precondition: 3.50% wide

    result = select_contract(packet(candidates=[candidate(bid=bid, ask=ask)]))

    assert result.status == "selected"
    assert result.selected.spread_bps == pytest.approx(350.0)


def test_a_spread_just_over_350_bps_is_rejected():
    bid, ask = OVER_LIMIT
    assert 350.0 < spread_bps(bid, ask) < 351.0  # precondition

    result = select_contract(packet(candidates=[candidate(bid=bid, ask=ask)]))

    assert result.verdicts[0].reject_reason == "wide_spread"


# --------------------------------------------------------------------------
# 7. which clock ages are measured against
# --------------------------------------------------------------------------


def test_ages_use_the_server_clock_when_present():
    # One second after the local stamp, one second before the server stamp.
    late = candidate(quote_at=OBSERVED_AT + timedelta(seconds=1))
    result = select_contract(packet(candidates=[late]))

    assert result.status == "selected"
    assert result.selected.quote_age_seconds == pytest.approx(1.0)


def test_without_a_server_clock_the_local_observed_at_is_the_reference():
    fresh = candidate(quote_at=OBSERVED_AT - timedelta(seconds=1))
    result = select_contract(packet(candidates=[fresh], quotes_read_at=None))
    assert result.status == "selected"
    assert result.selected.quote_age_seconds == pytest.approx(1.0)

    # A quote that arrived after the local stamp is then "in the future", which
    # is the safe outcome when the server clock is missing.
    late = candidate(quote_at=OBSERVED_AT + timedelta(seconds=1))
    result = select_contract(packet(candidates=[late], quotes_read_at=None))
    assert result.verdicts[0].reject_reason == "invalid_quote"


# --------------------------------------------------------------------------
# 8. constants and the status invariant
# --------------------------------------------------------------------------


def test_the_approved_thresholds_are_pinned():
    assert (DTE_TARGET_DAYS, MAX_SPREAD_BPS, MAX_QUOTE_AGE_SECONDS) == (7, 350, 10)


def selected_contract():
    return SelectedContract(
        symbol="SPY260902C00765000",
        option_type="call",
        strike_price=765.0,
        expiration_date=TODAY + timedelta(days=7),
        days_to_expiration=7,
        bid=4.9,
        ask=5.0,
        mid=4.95,
        spread_bps=202.0,
        quote_at=READ_AT,
        quote_age_seconds=1.0,
        underlying_mid=765.0,
    )


@pytest.mark.parametrize(
    "fields",
    [
        dict(status="selected", selected=None, reason=None),
        dict(status="selected", selected=selected_contract(), reason="no_candidates"),
        dict(status="no_contract", selected=None, reason=None),
        dict(status="no_contract", selected=selected_contract(), reason="no_candidates"),
        dict(status="not_applicable", selected=None, reason="no_candidates"),
        dict(status="not_applicable", selected=selected_contract(), reason=None),
    ],
    ids=[
        "selected-without-contract",
        "selected-with-reason",
        "no-contract-without-reason",
        "no-contract-with-contract",
        "not-applicable-with-reason",
        "not-applicable-with-contract",
    ],
)
def test_a_result_that_contradicts_its_status_cannot_be_built(fields):
    with pytest.raises(ValueError):
        SelectionResult(observed_at=OBSERVED_AT, action="BUY_CALL", **fields)


def test_results_that_agree_with_their_status_can_be_built():
    SelectionResult(observed_at=OBSERVED_AT, action="BUY_CALL", status="selected", selected=selected_contract())
    SelectionResult(observed_at=OBSERVED_AT, action="BUY_CALL", status="no_contract", reason="no_candidates")
    SelectionResult(observed_at=OBSERVED_AT, action="HOLD", status="not_applicable")


# --------------------------------------------------------------------------
# 9. a result is a record: deterministic, frozen, serializable
# --------------------------------------------------------------------------


def test_selection_is_deterministic_frozen_and_serializable():
    chain = packet(candidates=ladder(7, [763, 764, 765, 766, 767]) + ladder(8, [765]))
    result = select_contract(chain)

    assert select_contract(chain) == result
    serialized = result.model_dump_json()
    assert SelectionResult.model_validate_json(serialized) == result
    assert json.loads(serialized)["selected"]["symbol"] == result.selected.symbol

    with pytest.raises(Exception):
        result.status = "no_contract"
    with pytest.raises(Exception):
        result.verdicts[0].reject_reason = "wide_spread"
    with pytest.raises(Exception):
        SelectionResult(**{**result.model_dump(), "extra": True})
    assert not hasattr(result.verdicts, "append")


# --------------------------------------------------------------------------
# 10. the summary shows the pick, the rules and every rejection
# --------------------------------------------------------------------------


def test_format_summary_shows_the_pick_the_rules_and_every_rejection():
    candidates = ladder(7, [764, 765, 766]) + [
        candidate(strike=767.0, dte=7, bid=OVER_LIMIT[0], ask=OVER_LIMIT[1]),
        candidate(strike=763.0, dte=7, age=99.0),
    ]
    rendered = format_summary(select_contract(packet(candidates=candidates)))

    assert "selected" in rendered
    assert "SPY260902C00765000" in rendered
    assert "765.00" in rendered
    assert "350" in rendered and "10 s" in rendered  # the rules line
    assert "wide_spread" in rendered and "767.00" in rendered
    assert "stale_quote" in rendered and "763.00" in rendered


def test_format_summary_explains_hold_and_no_contract_without_fake_numbers():
    hold = format_summary(select_contract(packet("HOLD", mid=None)))
    assert "not_applicable" in hold
    assert "0.00" not in hold

    nothing = format_summary(select_contract(packet(candidates=())))
    assert "no_candidates" in nothing
    assert "0.00" not in nothing


# --------------------------------------------------------------------------
# 11. pure and read-only
# --------------------------------------------------------------------------


def test_selection_makes_no_network_call(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("the selector must not open a socket")

    monkeypatch.setattr(socket, "socket", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)

    assert select_contract(packet(candidates=ladder(7, [765]))).status == "selected"


def test_the_selector_module_never_imports_the_vendor_sdk():
    source = Path(selector_module.__file__).read_text(encoding="utf-8")
    assert "alpaca" not in source.lower()

    for value in vars(selector_module).values():
        module = getattr(value, "__module__", "") or ""
        assert not module.startswith("alpaca")


def test_the_selector_module_exposes_no_trading_or_execution_helper():
    forbidden = (
        "submit", "cancel", "replace", "close_position", "close_all", "exercise",
        "order", "buy_call", "buy_put", "place_", "position", "size", "risk", "decide",
    )
    offenders = [
        name for name in dir(selector_module) if any(word in name.lower() for word in forbidden)
    ]
    assert offenders == []


# --------------------------------------------------------------------------
# 12. the command never selects for a direction it was not given
# --------------------------------------------------------------------------


@pytest.mark.parametrize("argv", [["--action", "SELL"], ["--action", "HOLD"], ["--action="]])
def test_main_refuses_a_direction_it_cannot_select_for(argv, capsys):
    assert main(argv) == 1
    assert "usage" in capsys.readouterr().err
