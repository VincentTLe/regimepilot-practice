"""Decision layer: the LLM (or the human, in manual mode) picks at most one entry
per call. The cycle may call again with the remaining candidates once an entry
is placed, until the per-cycle premium cap is used up.

The decider only ever chooses a symbol from the scored candidate list and a
direction. Strikes, expiration, quantity and price are deterministic code.
Any malformed output means no entry. Errors surface as status codes or
exception type names only — response bodies are never copied into errors.
"""

from __future__ import annotations

import json
import time
from typing import Callable

import httpx
from loguru import logger

import settings
from data_models import EntryChoice, SymbolFeatures

# Endpoint, models, reasoning effort and timeouts live in settings.yaml (llm section).
MAX_TOKENS = 1200  # reasoning tokens count against this on GLM-style models; the JSON reply itself is tiny
TEMPERATURE = 0.2

SYSTEM_PROMPT = """You are the entry-signal module of a paper-trading agent that buys
debit vertical spreads on liquid US options. Every candidate underlying has fired
at least one technical event on its latest completed bar:
  gap_up / gap_down           - bar opened more than 2 ATR away from the prior close
  breakout_up / breakout_down - bar body (close minus open) exceeded 2 ATR
  macd_cross_up / macd_cross_down - MACD histogram crossed zero
Each candidate also carries its RSI, ATR and MACD histogram readings, plus
advisory trend context: ema_fast_dist / ema_slow_dist are the last close minus
a fast/slow trend EMA (positive = price above the anchor, an up-regime;
negative = below). Weigh trend alignment - entering against both anchors needs
a strong reason. flow_imbalance is the tape: tick-rule buy volume minus sell
volume over their sum for the last minutes of prints (-1 = every print hit the
bid, +1 = every print lifted the offer), with flow_trades prints behind it. The
candidate's events already agree with the tape; a weak reading (|flow| below
0.3) or few prints is a reason to pass rather than chase. A candidate whose "held" field is set already has an open
spread in that direction: entering it is an ADD to that position, and your
direction must match it (code rejects any other direction). Choose at most
ONE candidate from this list to enter, or pass. Once an entry is placed you may
be asked again in the same cycle with the remaining candidates.

Reply with strict JSON only:
{"action": "enter" | "pass", "symbol": "<one of the candidate symbols>",
 "direction": "CALL" | "PUT", "thesis": "<one sentence>"}

Rules: only pick a symbol from the candidate list. CALL means you expect the
underlying to rise, PUT to fall. The event direction is a hint, not an order;
an exhausted move (e.g. extreme RSI) may argue against following it. Pass when
nothing is convincing - passing is always acceptable."""


class LlmError(Exception):
    pass


def _payload(model: str, messages: list[dict], json_mode: bool) -> dict:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "reasoning_effort": settings.LLM_REASONING_EFFORT,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    return payload


def _parse_chat(response: httpx.Response, model: str) -> tuple[str, str] | None:
    """(content, model_used) from an OpenAI-style chat reply; None on any unexpected shape.

    An empty content (a reasoning-only reply, or a truncated one) is treated as
    unexpected: the caller moves on to the next model rather than parsing air.
    """
    try:
        body = response.json()
        message = body["choices"][0]["message"]
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            # GLM-style replies sometimes land in the reasoning field with an empty
            # content (seen on zai-org/GLM-5.3 via Featherless, 2026-09-03). The JSON
            # extractor downstream tolerates prose around the object.
            content = message.get("reasoning") or message.get("reasoning_content")
        model_used = body.get("model", model)
        if not isinstance(content, str) or not content.strip() or not isinstance(model_used, str):
            return None
    except Exception:
        return None
    return content, model_used


def call_llm(
    messages: list[dict],
    api_key: str,
    transport: httpx.BaseTransport | None = None,
) -> tuple[str, str]:
    """POST to the OpenAI-compatible endpoint in settings.yaml (Featherless).

    Tries the primary model, then every fallback in order — the fallback runs
    client-side (no OpenRouter-style `models` array). A model that answers
    HTTP 400 while json_mode is on is retried once without response_format.
    Returns (content, model_used). Raises LlmError only when every model
    failed; the message carries status codes / exception type names only.
    """
    url = f"{settings.LLM_BASE_URL}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}"}
    failures: list[str] = []
    timeout = httpx.Timeout(settings.LLM_TIMEOUT_SECONDS, connect=10.0)  # read budget per attempt, quick connect
    with httpx.Client(timeout=timeout, transport=transport) as client:
        for model in (settings.PRIMARY_MODEL, *settings.FALLBACK_MODELS):
            json_mode = settings.LLM_JSON_MODE
            while True:
                try:
                    response = client.post(url, json=_payload(model, messages, json_mode), headers=headers)
                except Exception as error:
                    failures.append(f"{model}: {type(error).__name__}")
                    break
                if response.status_code == 400 and json_mode:
                    logger.warning("{}: HTTP 400 with json_mode, retrying without response_format", model)
                    json_mode = False
                    continue
                if response.status_code != 200:
                    failures.append(f"{model}: HTTP {response.status_code}")
                    break
                parsed = _parse_chat(response, model)
                if parsed is None:
                    failures.append(f"{model}: unexpected response shape")
                    break
                return parsed
            logger.warning("LLM model failed, trying next: {}", failures[-1])
    raise LlmError(f"{settings.LLM_PROVIDER}: every model failed ({'; '.join(failures)})")


def ping(api_key: str, transport: httpx.BaseTransport | None = None) -> tuple[str, float]:
    """One tiny round trip for preflight: (model that answered, seconds). Raises LlmError."""
    messages = [{"role": "user", "content": 'Reply with JSON {"ok": true}'}]
    started = time.perf_counter()
    _, model_used = call_llm(messages, api_key, transport=transport)
    return model_used, time.perf_counter() - started


def _extract_json(text: str) -> dict | None:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            parsed = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    return parsed if isinstance(parsed, dict) else None


def parse_entry_choice(text: str, allowed_symbols: set[str], model: str) -> EntryChoice | None:
    """Strictly validate the model's reply; anything malformed means no entry."""
    data = _extract_json(text)
    if data is None:
        return None
    if data.get("action") != "enter":
        return None
    symbol = data.get("symbol")
    direction = data.get("direction")
    if not isinstance(symbol, str) or symbol.upper() not in allowed_symbols:
        return None
    if direction not in ("CALL", "PUT"):
        return None
    thesis = data.get("thesis")
    thesis = thesis if isinstance(thesis, str) else ""
    return EntryChoice(symbol=symbol.upper(), direction=direction, thesis=thesis, model=model)


def decide_entry(
    candidates: list[SymbolFeatures],
    api_key: str,
    transport: httpx.BaseTransport | None = None,
) -> EntryChoice | None:
    """Ask the LLM to pick at most one entry from the gate-passing candidates (one call = one pick)."""
    tradeable = [c for c in candidates if c.gate_block is None]
    if not tradeable:
        return None
    briefing = {
        "candidates": [
            {
                "symbol": c.symbol,
                "spot": c.mid,
                "events": [{"kind": e.kind, "direction": e.direction} for e in c.events],
                "rsi": c.rsi,
                "atr": c.atr,
                "macd_hist": c.macd_hist,
                "ema_fast_dist": c.ema_fast_dist,
                "ema_slow_dist": c.ema_slow_dist,
                "flow_imbalance": c.flow_imbalance,
                "flow_trades": c.flow_trades,
                "held": c.held,
            }
            for c in tradeable
        ]
    }
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(briefing)},
    ]
    content, model_used = call_llm(messages, api_key, transport=transport)
    return parse_entry_choice(content, {c.symbol for c in tradeable}, model_used)


def manual_decide(
    candidates: list[SymbolFeatures],
    input_fn: Callable[[str], str] | None = None,
    echo: Callable[[str], None] = print,
) -> EntryChoice | None:
    """Manual mode: the human picks which candidate to trade, or passes.

    Anything unparseable — blank, not a number, out of range — is a pass:
    no order ever results from garbage input. End of input (EOF) is a pass too:
    the cycle can ask a second time after an entry, and piped answers such as
    "1\nCALL\n" run out there. The direction defaults to the selected
    candidate's first event; only an explicit CALL/PUT overrides it.
    """
    if input_fn is None:
        input_fn = input  # resolved at call time so tests can patch builtins.input
    tradeable = sorted(
        (c for c in candidates if c.gate_block is None and c.events),
        key=lambda c: c.symbol,
    )
    if not tradeable:
        return None
    echo("Candidates with fired events:")
    for index, c in enumerate(tradeable, start=1):
        events = ", ".join(e.kind for e in c.events)
        echo(
            f"  [{index}] {c.symbol:<6} spot={c.mid} events={events} "
            f"rsi={c.rsi} atr={c.atr} macd_hist={c.macd_hist} "
            f"ema_fast_dist={c.ema_fast_dist} ema_slow_dist={c.ema_slow_dist} "
            f"flow={c.flow_imbalance} ({c.flow_trades} prints)"
            + (f" held={c.held} (add)" if c.held else "")
        )
    try:
        raw = input_fn("Select a candidate number to trade (blank to pass): ").strip()
        if not raw.isdigit() or not (1 <= int(raw) <= len(tradeable)):
            return None
        chosen = tradeable[int(raw) - 1]
        default_direction = chosen.events[0].direction
        raw_direction = input_fn(f"Direction CALL or PUT [default {default_direction}]: ").strip().upper()
    except EOFError:
        return None
    direction = raw_direction if raw_direction in ("CALL", "PUT") else default_direction
    event_kinds = ", ".join(e.kind for e in chosen.events)
    return EntryChoice(
        symbol=chosen.symbol,
        direction=direction,
        thesis=f"Manual selection ({event_kinds}).",
        model="manual",
    )
