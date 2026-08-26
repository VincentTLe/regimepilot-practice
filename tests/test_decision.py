"""Decision tests: stub rules, parsing and OpenRouter boundary."""

import json
from datetime import datetime, timezone

import httpx
import pytest

from regimepilot import decision as decision_module
from regimepilot.decision import (
    DecisionError,
    build_gate_hold_proposal,
    build_prompt_messages,
    call_openrouter,
    parse_trade_proposal,
    propose_trade,
    stub_proposal,
)
from regimepilot.models import (
    AccountHint,
    EvidencePacket,
    GatesEvidence,
    NewsEvidence,
    TradeProposal,
    UnderlyingEvidence,
)

OBSERVED_AT = datetime(2026, 8, 24, 14, 36, 5, tzinfo=timezone.utc)
OPENROUTER_KEY = "OPENROUTER-TEST-KEY"


def build_evidence(*, passed=True, hold_reason=None, momentum_align="aligned_up"):
    return EvidencePacket(
        observed_at=OBSERVED_AT,
        symbol="SPY",
        gates=GatesEvidence(
            passed=passed,
            hold_reason=hold_reason,
            momentum_align=momentum_align,
        ),
        underlying=UnderlyingEvidence(
            data_feed="iex",
            market_is_open=True,
            return_15m=0.05,
            return_60m=0.08,
            return_since_open=0.02,
            overnight_gap_pct=0.01,
            realized_vol_30m=0.004,
            spread_bps=2.0,
            bar_age_seconds=30.0,
            minutes_since_open=90.0,
            minutes_to_close=300.0,
        ),
        news=NewsEvidence(available=True, item_count=0, items=()),
        account=AccountHint(has_open_option_position=False),
    )


def settings_env(**overrides):
    env = {
        "ALPACA_API_KEY": "test-key",
        "ALPACA_SECRET_KEY": "test-secret",
        "ALPACA_PAPER": "true",
        "OPENROUTER_API_KEY": OPENROUTER_KEY,
    }
    env.update(overrides)
    return env


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("error", request=None, response=self)

    def json(self):
        return self._payload


def test_pre_gate_failure_returns_hold_without_calling_llm():
    evidence = build_evidence(passed=False, hold_reason="market_closed")
    proposal = propose_trade(evidence, stub=False, settings=decision_module.load_settings(settings_env()))

    assert proposal.action == "HOLD"
    assert proposal.gate_skipped is True
    assert proposal.model == "pre_gate"


def test_stub_proposal_maps_aligned_momentum_to_call_or_put():
    assert stub_proposal(build_evidence(momentum_align="aligned_up")).action == "BUY_CALL"
    assert stub_proposal(build_evidence(momentum_align="aligned_down")).action == "BUY_PUT"
    assert stub_proposal(build_evidence(momentum_align="mixed")).action == "HOLD"


def test_parse_trade_proposal_accepts_valid_json():
    evidence = build_evidence()
    raw = json.dumps(
        {
            "action": "BUY_PUT",
            "confidence": "high",
            "thesis": "Momentum turned down.",
            "evidence_used": ["underlying.return_15m", "gates.momentum_align"],
        }
    )
    proposal = parse_trade_proposal(raw, evidence, model="stealth/ox-alpha")

    assert proposal.action == "BUY_PUT"
    assert proposal.confidence == "high"
    assert proposal.evidence_used == ("underlying.return_15m", "gates.momentum_align")


def test_parse_trade_proposal_defaults_to_hold_on_invalid_json():
    evidence = build_evidence()
    proposal = parse_trade_proposal("not json at all", evidence, model="stealth/ox-alpha")

    assert proposal.action == "HOLD"
    assert proposal.confidence == "low"
    assert "invalid" in proposal.thesis.lower()


def test_call_openrouter_uses_primary_and_fallback_models():
    captured = {}

    def fake_post(url, *, headers, json, timeout):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        captured["timeout"] = timeout
        return FakeResponse(
            {
                "model": "stealth/ox-alpha",
                "choices": [{"message": {"content": '{"action":"HOLD","confidence":"low","thesis":"Wait.","evidence_used":[]}'}}],
            }
        )

    text, model = call_openrouter(
        build_prompt_messages(build_evidence()),
        api_key=OPENROUTER_KEY,
        transport=fake_post,
    )

    assert "stealth/ox-alpha" in captured["json"]["models"]
    assert "openrouter/free" in captured["json"]["models"]
    assert captured["headers"]["Authorization"] == f"Bearer {OPENROUTER_KEY}"
    assert "HOLD" in text
    assert model == "stealth/ox-alpha"


def test_call_openrouter_raises_a_credential_safe_error_on_http_failure():
    def fake_post(*args, **kwargs):
        raise httpx.ConnectError("network down")

    with pytest.raises(DecisionError) as excinfo:
        call_openrouter([], api_key=OPENROUTER_KEY, transport=fake_post)

    assert "ConnectError" in str(excinfo.value)
    assert OPENROUTER_KEY not in str(excinfo.value)


def test_propose_trade_requires_openrouter_key_when_not_stub():
    evidence = build_evidence()
    settings = decision_module.load_settings(settings_env(OPENROUTER_API_KEY=""))

    with pytest.raises(decision_module.ConfigError):
        propose_trade(evidence, stub=False, settings=settings)


def test_build_gate_hold_proposal_is_deterministic():
    proposal = build_gate_hold_proposal(build_evidence(passed=False, hold_reason="stale_data"))

    assert isinstance(proposal, TradeProposal)
    assert proposal.action == "HOLD"
    assert proposal.gate_skipped is True


def test_the_decision_module_exposes_no_trading_or_execution_helper():
    forbidden = (
        "submit",
        "cancel",
        "replace",
        "close_position",
        "close_all",
        "exercise",
        "place_",
        "order",
    )
    offenders = [
        name for name in dir(decision_module) if any(word in name.lower() for word in forbidden)
    ]
    assert offenders == []


class NotJsonResponse(FakeResponse):
    def json(self):
        raise ValueError("body is not JSON")


@pytest.mark.parametrize("status", [401, 404, 429, 500])
def test_call_openrouter_reports_the_http_status_without_secrets(status):
    body = {"error": {"message": f"BODY-TEXT-{status} key={OPENROUTER_KEY}"}}

    def fake_post(*args, **kwargs):
        return FakeResponse(body, status_code=status)

    with pytest.raises(DecisionError) as excinfo:
        call_openrouter([], api_key=OPENROUTER_KEY, transport=fake_post)

    text = str(excinfo.value)
    assert f"HTTP {status}" in text
    assert OPENROUTER_KEY not in text
    assert "BODY-TEXT" not in text


@pytest.mark.parametrize(
    "body",
    [
        None,
        [],
        "text",
        {},
        {"choices": None},
        {"choices": []},
        {"choices": [None]},
        {"choices": ["x"]},
        {"choices": [{}]},
        {"choices": [{"message": None}]},
        {"choices": [{"message": {"content": ""}}]},
        {"choices": [{"message": {"content": 5}}]},
    ],
)
def test_call_openrouter_turns_a_malformed_body_into_decision_error(body):
    def fake_post(*args, **kwargs):
        return FakeResponse(body)

    with pytest.raises(DecisionError):
        call_openrouter([], api_key=OPENROUTER_KEY, transport=fake_post)


def test_call_openrouter_turns_a_non_json_body_into_decision_error():
    def fake_post(*args, **kwargs):
        return NotJsonResponse({})

    with pytest.raises(DecisionError):
        call_openrouter([], api_key=OPENROUTER_KEY, transport=fake_post)
