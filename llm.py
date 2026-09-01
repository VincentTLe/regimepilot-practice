"""Entry signal, decision side: the LLM (or stub) picks at most one entry.

The model only ever chooses a symbol from the scored candidate list and a
direction. Strikes, expiration, quantity and price are deterministic code.
Any malformed output means no entry. Errors surface as status codes or
exception type names only — response bodies are never copied into errors.
"""

from __future__ import annotations

import json

import httpx

from models import EntryChoice, SymbolFeatures

OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
PRIMARY_MODEL = "z-ai/glm-5.3-flash"
FALLBACK_MODELS = ("minimax/minimax-m3:free", "nvidia/nemotron-3-super-120b-a12b:free")
TIMEOUT_SECONDS = 60.0

SYSTEM_PROMPT = """You are the entry-signal module of a paper-trading agent that buys
debit vertical spreads on liquid US options. Every candidate underlying has fired
at least one technical event on its latest completed bar:
  gap_up / gap_down           - bar opened more than 2 ATR away from the prior close
  breakout_up / breakout_down - bar body (close minus open) exceeded 2 ATR
  macd_cross_up / macd_cross_down - MACD histogram crossed zero
Each candidate also carries its RSI, ATR and MACD histogram readings. Choose at
most ONE candidate to enter, or pass.

Reply with strict JSON only:
{"action": "enter" | "pass", "symbol": "<one of the candidate symbols>",
 "direction": "CALL" | "PUT", "thesis": "<one sentence>"}

Rules: only pick a symbol from the candidate list. CALL means you expect the
underlying to rise, PUT to fall. The event direction is a hint, not an order;
an exhausted move (e.g. extreme RSI) may argue against following it. Pass when
nothing is convincing - passing is always acceptable."""


class LlmError(Exception):
    pass


def call_openrouter(
    messages: list[dict],
    api_key: str,
    transport: httpx.BaseTransport | None = None,
) -> tuple[str, str]:
    """POST to OpenRouter; returns (content, model_used). Raises LlmError."""
    payload = {
        "model": PRIMARY_MODEL,
        "models": [PRIMARY_MODEL, *FALLBACK_MODELS],
        "messages": messages,
        "response_format": {"type": "json_object"},
    }
    try:
        with httpx.Client(timeout=TIMEOUT_SECONDS, transport=transport) as client:
            response = client.post(
                OPENROUTER_CHAT_URL,
                json=payload,
                headers={"Authorization": f"Bearer {api_key}"},
            )
    except Exception as error:
        raise LlmError(f"openrouter request failed: {type(error).__name__}") from None
    if response.status_code != 200:
        raise LlmError(f"openrouter returned HTTP {response.status_code}") from None
    try:
        body = response.json()
        content = body["choices"][0]["message"]["content"]
        model_used = body.get("model", PRIMARY_MODEL)
        if not isinstance(content, str) or not isinstance(model_used, str):
            raise TypeError
    except Exception:
        raise LlmError("openrouter response had an unexpected shape") from None
    return content, model_used


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
    """Ask the LLM to pick at most one entry from the gate-passing candidates."""
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
            }
            for c in tradeable
        ]
    }
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(briefing)},
    ]
    content, model_used = call_openrouter(messages, api_key, transport=transport)
    return parse_entry_choice(content, {c.symbol for c in tradeable}, model_used)


def stub_decide(candidates: list[SymbolFeatures]) -> EntryChoice | None:
    """Rule-based fallback: first candidate (by symbol) enters in its first event's direction."""
    for c in sorted(candidates, key=lambda c: c.symbol):
        if c.gate_block is not None or not c.events:
            continue
        event = c.events[0]
        return EntryChoice(
            symbol=c.symbol,
            direction=event.direction,
            thesis=f"Stub rule: {event.kind} event fired.",
            model="stub",
        )
    return None
