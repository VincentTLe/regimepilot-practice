import json

import httpx
import pytest

import decision_layer
import settings
from data_models import Event, SymbolFeatures


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
    assert decision_layer.parse_entry_choice(text, {"SPY", "QQQ"}, "m") is None


def test_parse_entry_choice_accepts_valid_and_fenced():
    raw = json.dumps({"action": "enter", "symbol": "spy", "direction": "PUT", "thesis": "t"})
    choice = decision_layer.parse_entry_choice(raw, {"SPY"}, "m")
    assert choice is not None and choice.symbol == "SPY" and choice.direction == "PUT"
    fenced = f"```json\n{raw}\n```"
    assert decision_layer.parse_entry_choice(fenced, {"SPY"}, "m") is not None


# --- transport-level hardening (Featherless, OpenAI-compatible) ---

@pytest.fixture(autouse=True)
def llm_settings(monkeypatch):
    monkeypatch.setattr(settings, "LLM_BASE_URL", "https://llm.test/v1")
    monkeypatch.setattr(settings, "PRIMARY_MODEL", "m-primary")
    monkeypatch.setattr(settings, "FALLBACK_MODELS", ("m-fallback",))
    monkeypatch.setattr(settings, "LLM_REASONING_EFFORT", "low")
    monkeypatch.setattr(settings, "LLM_JSON_MODE", True)
    monkeypatch.setattr(settings, "LLM_TIMEOUT_SECONDS", 5.0)


def recording_transport(responder):
    """responder(model, body) -> httpx.Response; records every request."""
    seen = []

    def handler(request):
        body = json.loads(request.content)
        seen.append({"url": str(request.url), "body": body})
        return responder(body.get("model"), body)

    return seen, httpx.MockTransport(handler)


def test_call_llm_happy_path():
    transport = transport_returning(200, chat_body('{"action":"pass"}', model="m-primary"))
    content, model = decision_layer.call_llm([], "key", transport=transport)
    assert content == '{"action":"pass"}' and model == "m-primary"


def test_call_llm_posts_openai_shape_to_base_url():
    seen, transport = recording_transport(lambda m, b: httpx.Response(200, json=chat_body("{}", m)))
    decision_layer.call_llm([{"role": "user", "content": "hi"}], "key", transport=transport)
    assert len(seen) == 1
    assert seen[0]["url"] == "https://llm.test/v1/chat/completions"
    body = seen[0]["body"]
    assert body["model"] == "m-primary"
    assert "models" not in body  # OpenRouter-only field must be gone
    assert body["reasoning_effort"] == "low"
    assert body["response_format"] == {"type": "json_object"}


def test_call_llm_falls_back_to_next_model_on_http_error():
    def responder(model, body):
        if model == "m-primary":
            return httpx.Response(500)
        return httpx.Response(200, json=chat_body('{"action":"pass"}', model))

    seen, transport = recording_transport(responder)
    content, model = decision_layer.call_llm([], "key", transport=transport)
    assert model == "m-fallback"
    assert [s["body"]["model"] for s in seen] == ["m-primary", "m-fallback"]


def test_call_llm_retries_same_model_without_json_mode_on_400():
    def responder(model, body):
        if "response_format" in body:
            return httpx.Response(400)
        return httpx.Response(200, json=chat_body('{"action":"pass"}', model))

    seen, transport = recording_transport(responder)
    content, model = decision_layer.call_llm([], "key", transport=transport)
    assert model == "m-primary"
    assert len(seen) == 2
    assert "response_format" in seen[0]["body"] and "response_format" not in seen[1]["body"]


def test_call_llm_all_models_fail_names_status_only():
    with pytest.raises(decision_layer.LlmError) as excinfo:
        decision_layer.call_llm([], "key", transport=transport_returning(429))
    assert "429" in str(excinfo.value)
    assert "key" not in str(excinfo.value)


def test_call_llm_bad_shape_or_empty_content_exhausts_models():
    with pytest.raises(decision_layer.LlmError):
        decision_layer.call_llm([], "key", transport=transport_returning(200, {"choices": []}))
    with pytest.raises(decision_layer.LlmError):
        decision_layer.call_llm([], "key", transport=transport_returning(200, chat_body("")))


def test_ping_reports_model_and_latency():
    transport = transport_returning(200, chat_body('{"ok": true}', model="m-primary"))
    model, seconds = decision_layer.ping("key", transport=transport)
    assert model == "m-primary" and seconds >= 0.0


def test_decide_entry_end_to_end():
    body = chat_body(json.dumps({"action": "enter", "symbol": "QQQ", "direction": "CALL", "thesis": "up"}))
    choice = decision_layer.decide_entry(
        [candidate("SPY"), candidate("QQQ")], "key", transport=transport_returning(200, body)
    )
    assert choice is not None and choice.symbol == "QQQ" and choice.model == "test-model"


def test_decide_entry_briefing_marks_adds_with_held_direction():
    seen = {}

    def handler(request):
        seen["briefing"] = json.loads(json.loads(request.content)["messages"][1]["content"])
        return httpx.Response(200, json=chat_body(json.dumps({"action": "pass"})))

    from dataclasses import replace

    add = replace(candidate("NVDA"), held="CALL")
    decision_layer.decide_entry([candidate("SPY"), add], "key", transport=httpx.MockTransport(handler))
    by_symbol = {c["symbol"]: c for c in seen["briefing"]["candidates"]}
    assert by_symbol["NVDA"]["held"] == "CALL"
    assert by_symbol["SPY"]["held"] is None


def test_decide_entry_skips_blocked_candidates_entirely():
    # only blocked candidates -> no LLM call is even attempted (transport would 500)
    blocked = [candidate("SPY", block="stale_data")]
    assert decision_layer.decide_entry(blocked, "key", transport=transport_returning(500)) is None


# --- manual mode ---

def scripted(*answers):
    iterator = iter(answers)
    return lambda prompt="": next(iterator)


def silent(_message):
    pass


def refuse_input(prompt=""):
    raise AssertionError("manual_decide must not prompt when there is nothing to pick")


def test_manual_selects_candidate_with_default_direction():
    choice = decision_layer.manual_decide(
        [
            candidate("SPY", events=(Event(kind="macd_cross_up", direction="CALL"),)),
            candidate("NVDA", events=(Event(kind="gap_down", direction="PUT"),)),
            candidate("AAPL", events=(Event(kind="breakout_up", direction="CALL"),), block="stale_data"),
        ],
        input_fn=scripted("1", ""),  # pick #1, accept default direction
        echo=silent,
    )
    # AAPL is gated out; candidates listed sorted, so #1 is NVDA; default = its event's PUT
    assert choice is not None and choice.symbol == "NVDA" and choice.direction == "PUT"
    assert choice.model == "manual" and "gap_down" in choice.thesis


def test_manual_direction_override():
    choice = decision_layer.manual_decide(
        [candidate("SPY", events=(Event(kind="breakout_up", direction="CALL"),))],
        input_fn=scripted("1", "put"),
        echo=silent,
    )
    assert choice is not None and choice.direction == "PUT"


@pytest.mark.parametrize("answer", ["", "0", "99", "abc", " enter "])
def test_manual_garbage_or_blank_input_is_a_pass(answer):
    choice = decision_layer.manual_decide(
        [candidate("SPY", events=(Event(kind="breakout_up", direction="CALL"),))],
        input_fn=scripted(answer, ""),
        echo=silent,
    )
    assert choice is None  # no order ever results from garbage input


@pytest.mark.parametrize("answers", [[], ["1"]])  # EOF at the number prompt, or at the direction prompt
def test_manual_end_of_input_is_a_pass(answers):
    queue = iter(answers)

    def piped(prompt):
        try:
            return next(queue)
        except StopIteration:
            raise EOFError

    assert decision_layer.manual_decide([candidate()], input_fn=piped, echo=lambda s: None) is None


def test_manual_never_prompts_without_candidates():
    assert decision_layer.manual_decide([], input_fn=refuse_input, echo=silent) is None
    assert decision_layer.manual_decide(
        [candidate("SPY", events=())], input_fn=refuse_input, echo=silent
    ) is None
    assert decision_layer.manual_decide(
        [candidate("SPY", block="stale_data")], input_fn=refuse_input, echo=silent
    ) is None


def test_call_llm_reads_reasoning_when_content_is_empty():
    body = {"model": "m-primary", "choices": [{"message": {
        "content": "", "reasoning": 'The user wants JSON. {"action": "pass"}'}}]}
    content, model = decision_layer.call_llm([], "key", transport=transport_returning(200, body))
    assert model == "m-primary" and decision_layer.parse_entry_choice(content, {"SPY"}, model) is None
    entered = {"model": "m-primary", "choices": [{"message": {
        "content": "", "reasoning": 'ok {"action": "enter", "symbol": "SPY", "direction": "CALL", "thesis": "t"}'}}]}
    content, _ = decision_layer.call_llm([], "key", transport=transport_returning(200, entered))
    assert decision_layer.parse_entry_choice(content, {"SPY"}, "m").symbol == "SPY"
