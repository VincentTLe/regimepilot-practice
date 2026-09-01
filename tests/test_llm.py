import json

import httpx
import pytest

import llm
from models import Event, SymbolFeatures


def candidate(symbol="SPY", events=(Event(kind="breakout_up", direction="CALL"),), block=None):
    return SymbolFeatures(
        symbol=symbol, mid=100.0, rsi=55.0, atr=1.2, macd_hist=0.05,
        events=tuple(events), bar_age_seconds=1.0, gate_block=block,
    )


def transport_returning(status_code=200, body=None):
    def handler(request):
        return httpx.Response(status_code, json=body if body is not None else {})
    return httpx.MockTransport(handler)


def chat_body(content, model="test-model"):
    return {"model": model, "choices": [{"message": {"content": content}}]}


# --- parse hardening: anything malformed means no entry ---

@pytest.mark.parametrize(
    "text",
    [
        "not json at all",
        json.dumps({"action": "pass"}),
        json.dumps({"action": "enter", "symbol": "TSLA", "direction": "CALL"}),  # off-list
        json.dumps({"action": "enter", "symbol": "SPY", "direction": "SIDEWAYS"}),
        json.dumps({"action": "enter", "direction": "CALL"}),  # no symbol
        json.dumps([1, 2, 3]),
    ],
)
def test_parse_entry_choice_rejects(text):
    assert llm.parse_entry_choice(text, {"SPY", "QQQ"}, "m") is None


def test_parse_entry_choice_accepts_valid_and_fenced():
    raw = json.dumps({"action": "enter", "symbol": "spy", "direction": "PUT", "thesis": "t"})
    choice = llm.parse_entry_choice(raw, {"SPY"}, "m")
    assert choice is not None and choice.symbol == "SPY" and choice.direction == "PUT"
    fenced = f"```json\n{raw}\n```"
    assert llm.parse_entry_choice(fenced, {"SPY"}, "m") is not None


# --- transport-level hardening ---

def test_call_openrouter_happy_path():
    transport = transport_returning(200, chat_body('{"action":"pass"}', model="fallback-x"))
    content, model = llm.call_openrouter([], "key", transport=transport)
    assert content == '{"action":"pass"}' and model == "fallback-x"


def test_call_openrouter_http_error_names_status_only():
    with pytest.raises(llm.LlmError) as excinfo:
        llm.call_openrouter([], "key", transport=transport_returning(429))
    assert "429" in str(excinfo.value)
    assert "key" not in str(excinfo.value)


def test_call_openrouter_bad_shape():
    with pytest.raises(llm.LlmError):
        llm.call_openrouter([], "key", transport=transport_returning(200, {"choices": []}))


def test_decide_entry_end_to_end():
    body = chat_body(json.dumps({"action": "enter", "symbol": "QQQ", "direction": "CALL", "thesis": "up"}))
    choice = llm.decide_entry(
        [candidate("SPY"), candidate("QQQ")], "key", transport=transport_returning(200, body)
    )
    assert choice is not None and choice.symbol == "QQQ" and choice.model == "test-model"


def test_decide_entry_skips_blocked_candidates_entirely():
    # only blocked candidates -> no LLM call is even attempted (transport would 500)
    blocked = [candidate("SPY", block="stale_data")]
    assert llm.decide_entry(blocked, "key", transport=transport_returning(500)) is None


# --- stub ---

def test_stub_takes_first_symbol_with_an_event_in_its_direction():
    picks = llm.stub_decide([
        candidate("SPY", events=(Event(kind="macd_cross_up", direction="CALL"),)),
        candidate("NVDA", events=(Event(kind="gap_down", direction="PUT"),)),
        candidate("AAPL", events=(Event(kind="breakout_up", direction="CALL"),), block="stale_data"),
    ])
    # AAPL is gated; NVDA sorts before SPY; its first event says PUT
    assert picks is not None and picks.symbol == "NVDA" and picks.direction == "PUT"
    assert picks.model == "stub" and "gap_down" in picks.thesis


def test_stub_passes_without_events():
    assert llm.stub_decide([candidate("SPY", events=())]) is None
    assert llm.stub_decide([]) is None
