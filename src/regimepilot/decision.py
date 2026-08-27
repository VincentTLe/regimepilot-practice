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
from regimepilot.console import tolerant_console
from regimepilot.evidence import EvidenceError, format_summary as format_evidence_summary, observe_evidence
from regimepilot.models import (
    Confidence,
    EntryDecision,
    EvidencePacket,
    OpenPositionContext,
    PortfolioContext,
    PortfolioDecision,
    PositionDecision,
    TradeAction,
    TradeProposal,
)
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
    "PORTFOLIO_SYSTEM_PROMPT",
    "build_portfolio_messages",
    "decide_portfolio",
    "format_portfolio_summary",
    "parse_portfolio_decision",
    "safe_portfolio_decision",
    "stub_portfolio_decision",
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


# ---------------------------------------------------------------------------
# Portfolio decision (approved 2026-08-27, with the three corrections). The
# model manages every held SPY option (HOLD or CLOSE) and may ask for at most
# one new entry; deterministic code validates all of it before anything else
# happens. Nothing here imports an order, plan or receipt model.
# ---------------------------------------------------------------------------

VALID_POSITION_ACTIONS = frozenset({"HOLD", "CLOSE"})
VALID_DIRECTIONS = frozenset({"CALL", "PUT"})
INVALID_OUTPUT_REASON = "LLM output was invalid; holding every position and opening nothing."
NOT_ADDRESSED_REASON = "not addressed by the model"

PORTFOLIO_SYSTEM_PROMPT = (
    "You are RegimePilot, an autonomous SPY options PORTFOLIO manager trading a paper account. "
    "You receive one EvidencePacket as JSON. Its 'portfolio' lists every SPY option currently held "
    "(symbol, option_type, strike_price, expiration_date, days_to_expiration, qty, avg_entry_price, "
    "cost_basis, current_price, unrealized_pl, unrealized_plpc, pending_order_side, hours_held, "
    "entry_thesis, previous_decision), the pending orders, whether a new entry is allowed "
    "(entry_allowed, entry_blocked_reason) and the limits you work under. 'underlying' holds the SPY "
    "features (returns, realized volatility, gap, minutes to close), 'gates' the momentum label, 'news' "
    "recent headlines.\n"
    "Return ONLY valid JSON of exactly this shape:\n"
    '{"positions":[{"symbol":"<exact held symbol>","action":"HOLD"|"CLOSE","reason":"..."}],'
    ' "new_entry": null | {"direction":"CALL"|"PUT","candidate_id": null,"thesis":"..."},'
    ' "confidence":"low"|"medium"|"high", "portfolio_thesis":"1-3 sentences",'
    ' "evidence_used":["field", ...]}\n'
    "Rules: every held position must appear exactly once with HOLD or CLOSE. CLOSE means sell the whole "
    "position now. new_entry must be null unless portfolio.entry_allowed is true. Never choose a "
    "quantity, a price, a strike, an expiration or an OCC symbol: deterministic code sizes and prices "
    "every order and may refuse it.\n"
    "Options playbook: theta decay accelerates as days_to_expiration falls; under about 5 DTE a long "
    "option bleeds fast, so prefer closing losers and taking profits early. Moneyness and delta are your "
    "directional exposure; an out-of-the-money long option needs a move soon. Judge unrealized_plpc "
    "against the entry thesis: close when the thesis is invalidated (momentum flipped against the "
    "position, news contradicts it), not merely because the position is red. Never add to a loser. A wide "
    "bid/ask spread is a real cost paid twice. Near the close prefer reducing risk over opening new risk. "
    "Consistency with previous_decision matters, but new evidence overrides it. If unsure, HOLD existing "
    "positions and do not open a new one."
)


def _held_positions(evidence: EvidencePacket) -> tuple[OpenPositionContext, ...]:
    return () if evidence.portfolio is None else evidence.portfolio.positions


def build_portfolio_messages(evidence: EvidencePacket) -> list[dict[str, str]]:
    """Deterministic chat messages for the portfolio decision."""
    payload = json.loads(evidence.model_dump_json())
    return [
        {"role": "system", "content": PORTFOLIO_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Decide for this EvidencePacket and respond with JSON only:\n"
                f"{json.dumps(payload, separators=(',', ':'))}"
            ),
        },
    ]


def safe_portfolio_decision(
    evidence: EvidencePacket,
    *,
    reason: str,
    gate_skipped: bool,
    model: str | None,
) -> PortfolioDecision:
    """The decision that changes nothing: HOLD every held symbol, open nothing."""
    positions = tuple(
        PositionDecision(symbol=p.symbol, action="HOLD", reason=reason) for p in _held_positions(evidence)
    )
    return PortfolioDecision(
        observed_at=evidence.observed_at,
        symbol=evidence.symbol,
        positions=positions,
        new_entry=None,
        confidence="low",
        portfolio_thesis=reason,
        evidence_used=("gates.hold_reason",) if gate_skipped else ("parse_error",),
        gate_skipped=gate_skipped,
        model=model,
    )


def stub_portfolio_decision(evidence: EvidencePacket) -> PortfolioDecision:
    """Rule-based stand-in for the model: momentum only, for offline runs and tests."""
    align = evidence.gates.momentum_align
    portfolio = evidence.portfolio
    positions = []
    for position in _held_positions(evidence):
        kind = position.option_type.lower()
        flipped = (kind == "call" and align == "aligned_down") or (kind == "put" and align == "aligned_up")
        positions.append(
            PositionDecision(
                symbol=position.symbol,
                action="CLOSE" if flipped else "HOLD",
                reason=(
                    "Stub rule: momentum flipped against the position."
                    if flipped
                    else "Stub rule: thesis intact."
                ),
            )
        )

    new_entry = None
    if portfolio is not None and portfolio.entry_allowed:
        if align == "aligned_up":
            new_entry = EntryDecision(direction="CALL", thesis="Stub rule: 15m and 60m momentum align upward.")
        elif align == "aligned_down":
            new_entry = EntryDecision(direction="PUT", thesis="Stub rule: 15m and 60m momentum align downward.")

    return PortfolioDecision(
        observed_at=evidence.observed_at,
        symbol=evidence.symbol,
        positions=tuple(positions),
        new_entry=new_entry,
        confidence="medium",
        portfolio_thesis=f"Stub rule: momentum is {align}; positions follow it, entries follow it.",
        evidence_used=("gates.momentum_align", "portfolio.positions", "portfolio.entry_allowed"),
        gate_skipped=False,
        model="stub",
    )


def _text(value: Any, default: str) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return default


def _position_decisions(raw: Any, held: tuple[OpenPositionContext, ...]) -> tuple[PositionDecision, ...]:
    """One verdict per held symbol: the model's first valid one, else HOLD."""
    entries = raw if isinstance(raw, list) else []
    held_symbols = {p.symbol for p in held}
    chosen: dict[str, PositionDecision] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        symbol = entry.get("symbol")
        if not isinstance(symbol, str) or symbol.strip() not in held_symbols:
            continue
        symbol = symbol.strip()
        if symbol in chosen:
            continue
        action = str(entry.get("action", "")).strip().upper()
        if action not in VALID_POSITION_ACTIONS:
            continue
        chosen[symbol] = PositionDecision(
            symbol=symbol,
            action=action,  # type: ignore[arg-type]
            reason=_text(entry.get("reason"), "no reason given"),
        )
    return tuple(
        chosen.get(p.symbol) or PositionDecision(symbol=p.symbol, action="HOLD", reason=NOT_ADDRESSED_REASON)
        for p in held
    )


def _entry_decision(raw: Any, portfolio: PortfolioContext | None) -> EntryDecision | None:
    """The model's entry request, kept only when the pre-check allowed one."""
    if portfolio is None or not portfolio.entry_allowed or not isinstance(raw, dict):
        return None
    direction = str(raw.get("direction", "")).strip().upper()
    if direction not in VALID_DIRECTIONS:
        return None
    shortlist = portfolio.call_candidates if direction == "CALL" else portfolio.put_candidates
    candidate_id = raw.get("candidate_id")
    if not (isinstance(candidate_id, str) and candidate_id in {c.candidate_id for c in shortlist}):
        candidate_id = None
    return EntryDecision(
        direction=direction,  # type: ignore[arg-type]
        candidate_id=candidate_id,
        thesis=_text(raw.get("thesis"), "no thesis given"),
    )


def parse_portfolio_decision(
    raw_text: str,
    evidence: EvidencePacket,
    *,
    model: str | None = None,
) -> PortfolioDecision:
    """Validate the model's JSON. Anything malformed becomes HOLD-all / no entry.

    Only ``positions``, ``new_entry``, ``confidence``, ``portfolio_thesis`` and
    ``evidence_used`` are read. Quantities, prices, symbols or anything else
    the model may have added are never looked at.
    """
    try:
        payload = _extract_json_object(raw_text)
    except (ValueError, TypeError, json.JSONDecodeError):
        return safe_portfolio_decision(evidence, reason=INVALID_OUTPUT_REASON, gate_skipped=False, model=model)

    confidence = str(payload.get("confidence", "")).strip().lower()
    if confidence not in VALID_CONFIDENCE:
        confidence = "low"
    raw_used = payload.get("evidence_used", [])
    evidence_used = (
        tuple(str(item).strip() for item in raw_used if str(item).strip()) if isinstance(raw_used, list) else ()
    )
    return PortfolioDecision(
        observed_at=evidence.observed_at,
        symbol=evidence.symbol,
        positions=_position_decisions(payload.get("positions"), _held_positions(evidence)),
        new_entry=_entry_decision(payload.get("new_entry"), evidence.portfolio),
        confidence=confidence,  # type: ignore[arg-type]
        portfolio_thesis=_text(payload.get("portfolio_thesis"), "no thesis given"),
        evidence_used=evidence_used,
        gate_skipped=False,
        model=model,
    )


def decide_portfolio(
    evidence: EvidencePacket,
    *,
    stub: bool = False,
    settings: Settings | None = None,
    transport: Callable[..., Any] | None = None,
) -> PortfolioDecision:
    """Turn one EvidencePacket into one PortfolioDecision.

    No model is called when the market is closed (nothing could be executed)
    or when nothing is held and no entry is allowed. Any other failed entry
    gate still lets the model manage the held positions; the parser then
    forces ``new_entry`` to ``None`` because the pre-check said so.
    """
    portfolio = evidence.portfolio
    held = _held_positions(evidence)
    entry_allowed = portfolio.entry_allowed if portfolio is not None else False

    if evidence.underlying.market_is_open is not True or evidence.gates.hold_reason == "market_closed":
        return safe_portfolio_decision(evidence, reason="market_closed", gate_skipped=True, model="pre_gate")
    if not held and not entry_allowed:
        blocked = portfolio.entry_blocked_reason if portfolio is not None else None
        return safe_portfolio_decision(
            evidence, reason=f"nothing to decide: {blocked or 'no portfolio'}", gate_skipped=True, model="pre_gate"
        )

    if stub:
        return stub_portfolio_decision(evidence)

    active_settings = settings or load_settings()
    api_key = require_openrouter_api_key(active_settings)
    raw_text, model_used = call_openrouter(build_portfolio_messages(evidence), api_key=api_key, transport=transport)
    return parse_portfolio_decision(raw_text, evidence, model=model_used)


def format_portfolio_summary(decision: PortfolioDecision) -> str:
    """A few lines: one per position verdict, the entry, confidence and thesis."""
    header = f"RegimePilot portfolio  {decision.symbol}  @ {decision.observed_at.strftime('%Y-%m-%d %H:%M:%SZ')}"
    if decision.gate_skipped:
        header += "  (pre-gate)"
    lines = [header]
    if decision.positions:
        lines.extend(f"  {v.symbol:<22} {v.action:<6} {v.reason}" for v in decision.positions)
    else:
        lines.append("  positions       (none held)")
    entry = decision.new_entry
    lines.append("  new entry       " + ("none" if entry is None else f"{entry.direction}: {entry.thesis}"))
    lines.append(f"  confidence      {decision.confidence}   model {decision.model or 'unknown'}")
    lines.append(f"  thesis          {decision.portfolio_thesis}")
    return "\n".join(lines)


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
    tolerant_console()
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
