"""Portfolio decision tests: the model's JSON becomes one validated PortfolioDecision.

Offline: the OpenRouter transport is a fake; no credential is read.
"""

import json as json_module
from datetime import date

import pytest

from regimepilot import decision as decision_module
from regimepilot.decision import decide_portfolio, parse_portfolio_decision, stub_portfolio_decision
from regimepilot.models import (
    AccountHint,
    EntryDecision,
    OpenPositionContext,
    PortfolioContext,
    PortfolioDecision,
)
from regimepilot.risk import DEFAULT_LIMITS

from test_decision import PRIMARY_MODEL, FakeResponse, build_evidence, settings_env

CALL_A = "SPY260903C00645000"
PUT_B = "SPY260903P00640000"
UNKNOWN_C = "SPY260903C00650000"

SETTINGS = decision_module.load_settings(settings_env())


def position_ctx(symbol, option_type, strike, *, qty=1.0, unrealized_plpc=0.0):
    return OpenPositionContext(
        symbol=symbol, option_type=option_type, strike_price=strike, expiration_date=date(2026, 9, 3),
        days_to_expiration=7, qty=qty, avg_entry_price=5.5, cost_basis=550.0, current_price=5.5,
        unrealized_pl=0.0, unrealized_plpc=unrealized_plpc,
    )


A = position_ctx(CALL_A, "call", 645.0)
B = position_ctx(PUT_B, "put", 640.0)


def evidence_with(*, positions=(), entry_allowed=True, blocked=None, momentum="aligned_up",
                  market_is_open=True, hold_reason=None):
    if not entry_allowed and blocked is None:
        blocked = hold_reason or "entry_gate"
    base = build_evidence(passed=hold_reason is None, hold_reason=hold_reason, momentum_align=momentum)
    underlying = base.underlying.model_copy(update={"market_is_open": market_is_open})
    portfolio = PortfolioContext(
        positions=tuple(positions), open_position_count=len(positions),
        total_cost_basis=sum(p.cost_basis for p in positions) if positions else 0.0,
        options_buying_power=10_000.0, equity=100_000.0, entry_allowed=entry_allowed,
        entry_blocked_reason=blocked, limits=DEFAULT_LIMITS,
    )
    return base.model_copy(update={
        "underlying": underlying, "portfolio": portfolio,
        "account": AccountHint(has_open_option_position=bool(positions)),
    })


def llm(payload, calls=None):
    """A transport that answers with ``payload`` (dict or raw text) in the OpenRouter chat shape."""
    content = payload if isinstance(payload, str) else json_module.dumps(payload)

    def fake_post(url, *, headers, json, timeout):
        if calls is not None:
            calls.append(json)
        return FakeResponse({"model": PRIMARY_MODEL, "choices": [{"message": {"content": content}}]})

    return fake_post


def verdicts(decision):
    return {v.symbol: v.action for v in decision.positions}


def test_hold_and_close_are_kept_per_held_symbol():
    payload = {
        "positions": [
            {"symbol": CALL_A, "action": "HOLD", "reason": "thesis intact"},
            {"symbol": PUT_B, "action": "close", "reason": "momentum flipped"},
        ],
        "new_entry": None, "confidence": "high", "portfolio_thesis": "Mixed tape.", "evidence_used": ["gates"],
    }

    decision = decide_portfolio(evidence_with(positions=(A, B)), settings=SETTINGS, transport=llm(payload))

    assert verdicts(decision) == {CALL_A: "HOLD", PUT_B: "CLOSE"}
    assert [v.reason for v in decision.positions] == ["thesis intact", "momentum flipped"]
    assert decision.new_entry is None
    assert decision.confidence == "high" and decision.model == PRIMARY_MODEL
    assert decision.gate_skipped is False


def test_an_omitted_position_holds_and_an_unknown_symbol_is_dropped():
    payload = {
        "positions": [
            {"symbol": CALL_A, "action": "HOLD", "reason": "fine"},
            {"symbol": UNKNOWN_C, "action": "CLOSE", "reason": "not ours"},
        ],
        "new_entry": None, "confidence": "medium", "portfolio_thesis": "t", "evidence_used": [],
    }

    decision = parse_portfolio_decision(json_module.dumps(payload), evidence_with(positions=(A, B)))

    assert verdicts(decision) == {CALL_A: "HOLD", PUT_B: "HOLD"}
    assert decision.positions[1].reason == "not addressed by the model"
    assert len(decision.positions) == 2


def test_malformed_output_holds_everything_and_opens_nothing():
    decision = parse_portfolio_decision("I would close B and buy calls", evidence_with(positions=(A, B)), model="m")

    assert verdicts(decision) == {CALL_A: "HOLD", PUT_B: "HOLD"}
    assert decision.new_entry is None
    assert decision.evidence_used == ("parse_error",)
    assert decision.confidence == "low" and decision.model == "m"


def test_a_new_entry_is_kept_only_when_the_pre_check_allowed_one():
    asked = {"positions": [], "new_entry": {"direction": "put", "candidate_id": "P9", "thesis": "fade"},
             "confidence": "medium", "portfolio_thesis": "t", "evidence_used": []}
    blocked = parse_portfolio_decision(json_module.dumps(asked), evidence_with(entry_allowed=False, blocked="max_positions"))
    allowed = parse_portfolio_decision(json_module.dumps(asked), evidence_with(entry_allowed=True))

    assert blocked.new_entry is None
    assert allowed.new_entry == EntryDecision(direction="PUT", candidate_id=None, thesis="fade")


def test_size_price_and_symbol_keys_from_the_model_are_ignored():
    payload = {
        "positions": [{"symbol": CALL_A, "action": "HOLD", "reason": "r", "qty": 50}],
        "new_entry": {"direction": "CALL", "thesis": "go", "qty": 10, "limit_price": 0.01, "symbol": "SPY991231C00001000"},
        "symbol": "QQQ", "orders": [{"symbol": "QQQ", "qty": 100}],
        "confidence": "high", "portfolio_thesis": "t", "evidence_used": [],
    }

    decision = parse_portfolio_decision(json_module.dumps(payload), evidence_with(positions=(A,)))

    assert decision.symbol == "SPY"
    assert decision.new_entry.direction == "CALL" and decision.new_entry.candidate_id is None
    for field in ("qty", "quantity", "limit_price", "price", "orders"):
        assert field not in PortfolioDecision.model_fields
        assert field not in EntryDecision.model_fields
    assert "symbol" not in EntryDecision.model_fields


def test_a_closed_market_never_calls_the_model():
    def never(*args, **kwargs):
        raise AssertionError("the model must not be called")

    decision = decide_portfolio(
        evidence_with(positions=(A, B), market_is_open=False, hold_reason="market_closed"),
        settings=SETTINGS, transport=never,
    )

    assert decision.gate_skipped is True and decision.model == "pre_gate"
    assert verdicts(decision) == {CALL_A: "HOLD", PUT_B: "HOLD"}
    assert decision.new_entry is None


def test_too_close_to_close_still_manages_positions_but_blocks_the_entry():
    """Correction 1: an entry gate never freezes the held positions."""
    calls = []
    payload = {"positions": [{"symbol": CALL_A, "action": "CLOSE", "reason": "late day"}],
               "new_entry": {"direction": "CALL", "thesis": "chase"},
               "confidence": "medium", "portfolio_thesis": "t", "evidence_used": []}

    decision = decide_portfolio(
        evidence_with(positions=(A,), entry_allowed=False, hold_reason="too_close_to_close"),
        settings=SETTINGS, transport=llm(payload, calls),
    )

    assert calls, "the model was called"
    assert verdicts(decision) == {CALL_A: "CLOSE"}
    assert decision.new_entry is None


def test_the_stub_closes_on_a_momentum_flip_and_enters_with_momentum():
    down = stub_portfolio_decision(evidence_with(positions=(A, B), momentum="aligned_down"))
    assert verdicts(down) == {CALL_A: "CLOSE", PUT_B: "HOLD"}
    assert down.new_entry.direction == "PUT" and down.model == "stub"

    flat_up = stub_portfolio_decision(evidence_with(momentum="aligned_up"))
    assert flat_up.positions == () and flat_up.new_entry.direction == "CALL"

    mixed = stub_portfolio_decision(evidence_with(positions=(A,), momentum="mixed"))
    assert verdicts(mixed) == {CALL_A: "HOLD"} and mixed.new_entry is None


def test_a_flat_account_with_entry_allowed_accepts_a_new_entry():
    payload = {"positions": [], "new_entry": {"direction": "CALL", "thesis": "breakout"},
               "confidence": "medium", "portfolio_thesis": "t", "evidence_used": ["underlying.return_15m"]}

    decision = decide_portfolio(evidence_with(), settings=SETTINGS, transport=llm(payload))

    assert decision.positions == ()
    assert decision.new_entry.direction == "CALL" and decision.new_entry.thesis == "breakout"


def test_nothing_to_decide_skips_the_model():
    def never(*args, **kwargs):
        raise AssertionError("the model must not be called")

    decision = decide_portfolio(evidence_with(entry_allowed=False, blocked="pending_buy_order"),
                                settings=SETTINGS, transport=never)

    assert decision.gate_skipped is True
    assert "pending_buy_order" in decision.portfolio_thesis
