"""LLM direction reasoning for Phase 3D.

Produces a TradeProposal (BUY_CALL / BUY_PUT / HOLD) from an EvidencePacket.
Read-only with respect to Alpaca trading: this module never submits orders.

Two modes:
* ``--stub`` uses deterministic rules for local learning and tests.
* default calls OpenRouter through a fixed chain of named free models.
"""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Callable, Sequence
from typing import Any

import httpx

from regimepilot.config import ConfigError, Settings, load_settings, require_openrouter_api_key
from regimepilot.evidence import EvidenceError, format_summary as format_evidence_summary, observe_evidence
from regimepilot.models import Confidence, EvidencePacket, TradeAction, TradeProposal
from regimepilot.news import build_news_client
from regimepilot.smoke_test import build_clients

# Chosen 2026-08-26 from live probes with the real prompt. The primary is the
# paid GLM-5.3 Flash (what the retired ``stealth/ox-alpha`` turned out to be;
# fractions of a cent per call); the fallbacks are named free models in
# quality order. OpenRouter tries the list in order, falls through on rate
# limits or downtime, and reports the model that answered in the response,
# which lands in ``TradeProposal.model``. Every entry is named on purpose:
# ``openrouter/free`` would pick an arbitrary model per request.
#
# OpenRouter accepts at most THREE entries in ``models``; a fourth is an
# HTTP 400. ``z-ai/glm-5.2:free`` was left out for that reason: it was
# rate-limited upstream on every probe, so it would only spend a slot.
PRIMARY_MODEL = "z-ai/glm-5.3-flash"
FALLBACK_MODELS = (
    "minimax/minimax-m3:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
)
OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"

VALID_ACTIONS = frozenset({"BUY_CALL", "BUY_PUT", "HOLD"})
VALID_CONFIDENCE = frozenset({"low", "medium", "high"})

SYSTEM_PROMPT = (
    "You are RegimePilot, a SPY options direction classifier for a paper-trading agent. "
    "Read the evidence JSON and decide whether a directional options trade is justified. "
    "Return ONLY valid JSON with keys: action, confidence, thesis, evidence_used. "
    "action must be one of BUY_CALL, BUY_PUT, HOLD. "
    "confidence must be one of low, medium, high. "
    "thesis must be 1-3 sentences. "
    "evidence_used must be a JSON array of field names you relied on. "
    "Do not choose strikes, expirations, quantities, or order types."
)

__all__ = [
    "DecisionError",
    "build_gate_hold_proposal",
    "build_prompt_messages",
    "call_openrouter",
    "format_summary",
    "parse_trade_proposal",
    "propose_trade",
    "stub_proposal",
    "main",
]


class DecisionError(RuntimeError):
    """Direction reasoning could not be completed."""


def build_gate_hold_proposal(evidence: EvidencePacket) -> TradeProposal:
    """Deterministic HOLD when pre-gates already rejected the cycle."""
    reason = evidence.gates.hold_reason or "pre_gate"
    return TradeProposal(
        observed_at=evidence.observed_at,
        symbol=evidence.symbol,
        action="HOLD",
        confidence="high",
        thesis=f"Pre-gate blocked reasoning: {reason}.",
        evidence_used=("gates.hold_reason",),
        gate_skipped=True,
        model="pre_gate",
    )


def build_prompt_messages(evidence: EvidencePacket) -> list[dict[str, str]]:
    """Deterministic chat messages for the LLM."""
    payload = json.loads(evidence.model_dump_json())
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Evaluate this EvidencePacket and respond with JSON only:\n"
                f"{json.dumps(payload, separators=(',', ':'))}"
            ),
        },
    ]


def _extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)

    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
    if match is None:
        raise ValueError("no JSON object found")

    parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("JSON root must be an object")
    return parsed


def parse_trade_proposal(
    raw_text: str,
    evidence: EvidencePacket,
    *,
    model: str | None = None,
) -> TradeProposal:
    """Validate LLM JSON. Any failure becomes a safe HOLD."""
    try:
        payload = _extract_json_object(raw_text)
        action = str(payload.get("action", "")).strip().upper()
        confidence = str(payload.get("confidence", "")).strip().lower()
        thesis = str(payload.get("thesis", "")).strip()
        evidence_used_raw = payload.get("evidence_used", [])

        if action not in VALID_ACTIONS:
            raise ValueError("invalid action")
        if confidence not in VALID_CONFIDENCE:
            raise ValueError("invalid confidence")
        if not thesis:
            raise ValueError("missing thesis")
        if not isinstance(evidence_used_raw, list):
            raise ValueError("evidence_used must be a list")

        evidence_used = tuple(str(item).strip() for item in evidence_used_raw if str(item).strip())
        return TradeProposal(
            observed_at=evidence.observed_at,
            symbol=evidence.symbol,
            action=action,  # type: ignore[arg-type]
            confidence=confidence,  # type: ignore[arg-type]
            thesis=thesis,
            evidence_used=evidence_used,
            gate_skipped=False,
            model=model,
        )
    except (ValueError, TypeError, json.JSONDecodeError):
        return TradeProposal(
            observed_at=evidence.observed_at,
            symbol=evidence.symbol,
            action="HOLD",
            confidence="low",
            thesis="LLM output was invalid; defaulting to HOLD.",
            evidence_used=("parse_error",),
            gate_skipped=False,
            model=model,
        )


def stub_proposal(evidence: EvidencePacket) -> TradeProposal:
    """Rule-based stand-in for the LLM while learning the schema."""
    align = evidence.gates.momentum_align
    if align == "aligned_up":
        return TradeProposal(
            observed_at=evidence.observed_at,
            symbol=evidence.symbol,
            action="BUY_CALL",
            confidence="medium",
            thesis="Stub rule: 15m and 60m momentum align upward.",
            evidence_used=("gates.momentum_align", "underlying.return_15m", "underlying.return_60m"),
            gate_skipped=False,
            model="stub",
        )
    if align == "aligned_down":
        return TradeProposal(
            observed_at=evidence.observed_at,
            symbol=evidence.symbol,
            action="BUY_PUT",
            confidence="medium",
            thesis="Stub rule: 15m and 60m momentum align downward.",
            evidence_used=("gates.momentum_align", "underlying.return_15m", "underlying.return_60m"),
            gate_skipped=False,
            model="stub",
        )
    return TradeProposal(
        observed_at=evidence.observed_at,
        symbol=evidence.symbol,
        action="HOLD",
        confidence="low",
        thesis="Stub rule: momentum is mixed or unknown.",
        evidence_used=("gates.momentum_align",),
        gate_skipped=False,
        model="stub",
    )


def call_openrouter(
    messages: Sequence[dict[str, str]],
    *,
    api_key: str,
    primary_model: str = PRIMARY_MODEL,
    fallback_models: Sequence[str] = FALLBACK_MODELS,
    transport: Callable[..., Any] | None = None,
) -> tuple[str, str]:
    """Call OpenRouter chat completions and return ``(text, model_used)``.

    ``models`` is OpenRouter's priority list: the primary first, then each
    fallback in order. ``model_used`` is whichever of them actually answered.
    """
    payload = {
        "model": primary_model,
        "models": [primary_model, *fallback_models],
        "messages": list(messages),
        "response_format": {"type": "json_object"},
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/regimepilot/practice",
        "X-Title": "RegimePilot Practice",
    }

    post = transport or httpx.post
    try:
        response = post(
            OPENROUTER_CHAT_URL,
            headers=headers,
            json=payload,
            timeout=60.0,
        )
    except Exception as error:  # noqa: BLE001 - uniform external failure
        raise DecisionError(f"openrouter request failed: {type(error).__name__}") from None

    # Only the status code is surfaced. The body and headers may quote the
    # request, and an HTTP client's exception text may too, so neither is used.
    status = getattr(response, "status_code", None)
    if not isinstance(status, int):
        raise DecisionError("openrouter response missing HTTP status")
    if not 200 <= status < 300:
        raise DecisionError(f"openrouter request failed: HTTP {status}")

    try:
        body = response.json()
    except Exception as error:  # noqa: BLE001 - uniform external failure
        raise DecisionError(f"openrouter response was not JSON: {type(error).__name__}") from None

    return _completion_text(body, primary_model=primary_model)


def _completion_text(body: Any, *, primary_model: str) -> tuple[str, str]:
    """Validate the chat-completion shape; any deviation is a ``DecisionError``.

    Every access is type-checked first, so a null, list or otherwise malformed
    body can never escape as ``AttributeError`` / ``IndexError``.
    """
    if not isinstance(body, dict):
        raise DecisionError("openrouter response malformed: body is not an object")

    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        raise DecisionError("openrouter response missing choices")

    first = choices[0]
    if not isinstance(first, dict):
        raise DecisionError("openrouter response malformed: choice is not an object")

    message = first.get("message")
    if not isinstance(message, dict):
        raise DecisionError("openrouter response missing message")

    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise DecisionError("openrouter response missing content")

    model_used = body.get("model")
    if not isinstance(model_used, str) or not model_used.strip():
        model_used = primary_model
    return content, model_used


def propose_trade(
    evidence: EvidencePacket,
    *,
    stub: bool = False,
    settings: Settings | None = None,
    transport: Callable[..., Any] | None = None,
) -> TradeProposal:
    """Turn one EvidencePacket into one TradeProposal."""
    if not evidence.gates.passed:
        return build_gate_hold_proposal(evidence)

    if stub:
        return stub_proposal(evidence)

    active_settings = settings or load_settings()
    api_key = require_openrouter_api_key(active_settings)
    messages = build_prompt_messages(evidence)
    raw_text, model_used = call_openrouter(messages, api_key=api_key, transport=transport)
    return parse_trade_proposal(raw_text, evidence, model=model_used)


def format_summary(proposal: TradeProposal) -> str:
    """Compact terminal summary."""
    skipped = "  (pre-gate HOLD)" if proposal.gate_skipped else ""
    model = proposal.model or "unknown"
    return "\n".join(
        [
            f"RegimePilot decision  {proposal.symbol}  @ "
            f"{proposal.observed_at.strftime('%Y-%m-%d %H:%M:%SZ')}",
            f"  action      {proposal.action}{skipped}",
            f"  confidence  {proposal.confidence}",
            f"  model       {model}",
            f"  thesis      {proposal.thesis}",
            f"  evidence    {', '.join(proposal.evidence_used) or '(none)'}",
        ]
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Observe evidence and print one TradeProposal."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    use_stub = "--stub" in arguments

    try:
        settings = load_settings()
        trading_client, data_client = build_clients(settings)
        news_client = build_news_client(settings)
    except ConfigError as error:
        print(f"configuration error: {error}", file=sys.stderr)
        return 1

    try:
        evidence = observe_evidence(trading_client, data_client, news_client)
    except EvidenceError as error:
        print(f"evidence read failed: {error}", file=sys.stderr)
        return 1

    try:
        proposal = propose_trade(evidence, stub=use_stub, settings=settings)
    except (ConfigError, DecisionError) as error:
        print(f"decision failed: {error}", file=sys.stderr)
        return 1

    if "--json" in arguments:
        print(json.dumps(json.loads(proposal.model_dump_json()), indent=2))
    else:
        print(format_evidence_summary(evidence))
        print()
        print(format_summary(proposal))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
