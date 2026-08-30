"""Deterministic contract selection for Phase 4B.

Pure module: no network, no vendor SDK, no LLM. It takes the ChainPacket that
Phase 4A observed and answers "which contract, if any?" with rules a person
can check by hand: the expiration nearest the target, the strike nearest the
underlying midpoint, and a fixed list of reasons to refuse a quote. Every
threshold here was approved on 2026-08-26 from market-hours evidence; do not
change one without approval.

The selector never invents a contract. A missing midpoint, an empty chain, or
a target expiration whose every quote fails a rule all end in "no contract",
with the verdict on each candidate kept for inspection. It also never jumps
to another expiration: if the expiration the rule picked has nothing
acceptable, the answer is no, not "something else".

Nothing here sizes, prices or submits anything. Phase 5 decides what to do
with a SelectedContract.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from collections.abc import Iterable, Sequence
from datetime import date, datetime

from regimepilot.console import tolerant_console
from regimepilot.features import quote_age_seconds, spread_bps, to_utc
from regimepilot.models import (
    CandidateVerdict,
    ChainPacket,
    ContractCandidate,
    RejectReason,
    SelectedContract,
    SelectionResult,
)

# Approved 2026-08-26. Calendar days from the observation's New York date to
# the expiration. Phase 4A only queries 5-10 days; this picks within that.
DTE_TARGET_DAYS = 7

# Approved 2026-08-26 after four market-hours samples (near-the-money SPY
# spreads ran 100-145 bps at the median and ~400 bps at worst) and the common
# "under 5%" rule of thumb. Compared after rounding to a millionth of a basis
# point, so a spread that is exactly 3.50% in cents is not rejected by
# floating-point noise.
MAX_SPREAD_BPS = 350

# Approved 2026-08-26. The option feed sends at most one quote per second and
# in-hours ages ran under 6 s, so anything older means the feed or the request
# stalled. Measured against the server clock the chain observer recorded.
MAX_QUOTE_AGE_SECONDS = 10

_BPS_PRECISION = 6

# The only two directions that name an option type. Keys are values, not
# names, so they stay out of the module namespace.
_OPTION_TYPE_FOR_ACTION = {"BUY_CALL": "call", "BUY_PUT": "put"}

__all__ = [
    "choose_expiration",
    "format_summary",
    "judge_candidate",
    "select_contract",
    "main",
]


def judge_candidate(
    candidate: ContractCandidate,
    *,
    option_type: str,
    reference: datetime,
) -> RejectReason | None:
    """Apply the refusal rules in a fixed order. First failing rule wins.

    ``None`` means the candidate is eligible. ``reference`` is the clock the
    quote's age is measured against; a quote stamped after it is invalid, not
    fresh, because a stamp from the future says nothing about the present.
    """
    if (
        candidate.strike_price is None
        or candidate.expiration_date is None
        or candidate.days_to_expiration is None
        or candidate.days_to_expiration < 1
        or candidate.option_type != option_type
    ):
        return "invalid_contract"

    if candidate.tradable is not True or candidate.status != "active":
        return "not_tradable"

    if candidate.bid is None or candidate.ask is None or candidate.quote_at is None:
        return "no_quote"

    bps = spread_bps(candidate.bid, candidate.ask)
    if bps is None or candidate.quote_at > to_utc(reference):
        return "invalid_quote"

    age = quote_age_seconds(candidate.quote_at, reference)
    if age is None or age > MAX_QUOTE_AGE_SECONDS:
        return "stale_quote"

    if round(bps, _BPS_PRECISION) > MAX_SPREAD_BPS:
        return "wide_spread"

    return None


def choose_expiration(candidates: Iterable[ContractCandidate]) -> date | None:
    """The expiration whose days-to-expiration is nearest the target; ties go later.

    Chosen by identity alone, before any quote is judged, so a bad quote can
    never move the selection to a different expiration.
    """
    best: tuple[tuple[int, int], date] | None = None
    for candidate in candidates:
        if candidate.expiration_date is None or candidate.days_to_expiration is None:
            continue
        key = (abs(candidate.days_to_expiration - DTE_TARGET_DAYS), -candidate.days_to_expiration)
        if best is None or key < best[0]:
            best = (key, candidate.expiration_date)
    return None if best is None else best[1]


def _rank_key(candidate: ContractCandidate, *, option_type: str, mid: float) -> tuple[float, float, str]:
    """Nearest the midpoint first; equal distances go to the in-the-money side.

    The distance is rounded so a midpoint computed from cent prices, which can
    sit a few billionths off x.50, still produces an exact tie.
    """
    strike = candidate.strike_price
    if strike is None:
        return (float("inf"), 0.0, candidate.symbol)
    distance = round(abs(strike - mid), 4)
    in_the_money_first = strike if option_type == "call" else -strike
    return (distance, in_the_money_first, candidate.symbol)


def _verdict(candidate: ContractCandidate, reason: RejectReason | None) -> CandidateVerdict:
    return CandidateVerdict(
        symbol=candidate.symbol,
        expiration_date=candidate.expiration_date,
        days_to_expiration=candidate.days_to_expiration,
        strike_price=candidate.strike_price,
        reject_reason=reason,
    )


def _selected(candidate: ContractCandidate, *, reference: datetime, underlying_mid: float) -> SelectedContract:
    """Only ever built for a candidate ``judge_candidate`` accepted, so no field is null."""
    bid, ask = candidate.bid, candidate.ask
    return SelectedContract(
        symbol=candidate.symbol,
        option_type=candidate.option_type,
        strike_price=candidate.strike_price,
        expiration_date=candidate.expiration_date,
        days_to_expiration=candidate.days_to_expiration,
        bid=bid,
        ask=ask,
        mid=(bid + ask) / 2,
        spread_bps=spread_bps(bid, ask),
        quote_at=candidate.quote_at,
        quote_age_seconds=quote_age_seconds(candidate.quote_at, reference),
        underlying_mid=underlying_mid,
    )


def select_contract(packet: ChainPacket) -> SelectionResult:
    """Choose one contract from an observed chain slice, or explain why not.

    Order of the answer: a HOLD has nothing to select; no midpoint means no
    strike window; no candidates means nothing to judge; otherwise pick the
    expiration, rank that expiration's candidates by distance from the mid,
    judge each, and take the first eligible one.
    """
    base = {"observed_at": packet.observed_at, "symbol": packet.symbol, "action": packet.action}

    option_type = _OPTION_TYPE_FOR_ACTION.get(packet.action)
    if option_type is None:
        return SelectionResult(**base, status="not_applicable")

    mid = packet.underlying_mid
    if mid is None:
        return SelectionResult(**base, status="no_contract", reason="no_underlying_price")

    if not packet.candidates:
        return SelectionResult(**base, status="no_contract", reason="no_candidates")

    target = choose_expiration(packet.candidates)
    reference = packet.quotes_read_at or packet.observed_at
    pool = [c for c in packet.candidates if target is None or c.expiration_date == target]
    ranked = sorted(pool, key=lambda c: _rank_key(c, option_type=option_type, mid=mid))
    verdicts = tuple(
        _verdict(c, judge_candidate(c, option_type=option_type, reference=reference)) for c in ranked
    )

    winner = next((c for c, v in zip(ranked, verdicts) if v.reject_reason is None), None)
    if winner is None:
        return SelectionResult(
            **base,
            status="no_contract",
            reason="all_candidates_rejected",
            target_expiration=target,
            verdicts=verdicts,
        )

    return SelectionResult(
        **base,
        status="selected",
        target_expiration=target,
        selected=_selected(winner, reference=reference, underlying_mid=mid),
        verdicts=verdicts,
    )


def _number(value: float | None, digits: int = 2) -> str:
    return "-" if value is None else f"{value:,.{digits}f}"


def format_summary(result: SelectionResult) -> str:
    """The pick, the rules it was picked by, and every rejection at the target expiry."""
    status = result.status if result.reason is None else f"{result.status}  ({result.reason})"
    lines = [
        f"RegimePilot selection  {result.symbol}  {result.action}"
        f"  @ {result.observed_at.strftime('%Y-%m-%d %H:%M:%SZ')}",
        f"  {'status':<15} {status}",
        f"  {'rules':<15} DTE target {DTE_TARGET_DAYS}   strike nearest mid"
        f"   max spread {MAX_SPREAD_BPS} bps   max quote age {MAX_QUOTE_AGE_SECONDS} s",
    ]

    if result.status == "not_applicable":
        lines.append("  (HOLD: no direction, nothing to select)")
        return "\n".join(lines)

    target = result.target_expiration
    target_dte = next(
        (v.days_to_expiration for v in result.verdicts if v.expiration_date == target), None
    )
    target_text = "-" if target is None else str(target)
    if target_dte is not None:
        target_text += f"  ({target_dte} DTE)"
    lines.append(f"  {'target expiry':<15} {target_text}")

    chosen = result.selected
    if chosen is None:
        lines.append(f"  {'selected':<15} -")
    else:
        lines.append(
            f"  {'selected':<15} {chosen.symbol}   strike {_number(chosen.strike_price)}"
            f"   bid {_number(chosen.bid)}  ask {_number(chosen.ask)}  mid {_number(chosen.mid, 3)}"
            f"   spread {_number(chosen.spread_bps, 1)} bps   age {_number(chosen.quote_age_seconds, 1)} s"
        )

    rejected = [v for v in result.verdicts if v.reject_reason is not None]
    eligible = len(result.verdicts) - len(rejected)
    counts = Counter(v.reject_reason for v in rejected)
    breakdown = ", ".join(f"{count} {reason}" for reason, count in sorted(counts.items()))
    lines.append(
        f"  {'verdicts':<15} {len(result.verdicts)} judged: {eligible} eligible, {len(rejected)} rejected"
        + (f"  ({breakdown})" if breakdown else "")
    )
    for verdict in rejected:
        lines.append(
            f"    {verdict.symbol:<22} strike {_number(verdict.strike_price):>9}   {verdict.reject_reason}"
        )
    return "\n".join(lines)


def _action_argument(arguments: Sequence[str]) -> str | None:
    """The value of ``--action X`` or ``--action=X``, upper-cased, or ``None``."""
    for index, argument in enumerate(arguments):
        if argument == "--action" and index + 1 < len(arguments):
            return arguments[index + 1].strip().upper()
        if argument.startswith("--action="):
            return argument.split("=", 1)[1].strip().upper()
    return None


def main(argv: Sequence[str] | None = None) -> int:
    """Run evidence -> proposal -> chain -> selection and print the result.

    ``--action BUY_CALL|BUY_PUT`` skips the evidence and the LLM and selects
    for that direction, so the selector can be watched on real quotes during
    market hours. ``--stub`` uses the rule-based proposal instead of the LLM.
    A ``no_contract`` result is a normal outcome and exits 0; only a failed
    read or a configuration problem exits 1. Nothing is ever submitted.
    """
    tolerant_console()
    arguments = list(sys.argv[1:] if argv is None else argv)

    forced = _action_argument(arguments)
    wants_direction = "--action" in arguments or any(a.startswith("--action=") for a in arguments)
    if wants_direction and forced not in _OPTION_TYPE_FOR_ACTION:
        print(
            "usage: python -m regimepilot.selector [--stub] [--action BUY_CALL|BUY_PUT] [--json]",
            file=sys.stderr,
        )
        return 1

    # The market-data and reasoning modules are imported here, not at module
    # level, so that importing the selector never loads a vendor SDK.
    from regimepilot.chain import ChainError, build_option_data_client, observe_chain
    from regimepilot.chain import format_summary as format_chain_summary
    from regimepilot.config import ConfigError, load_settings
    from regimepilot.decision import DecisionError, propose_trade
    from regimepilot.decision import format_summary as format_proposal_summary
    from regimepilot.evidence import EvidenceError, observe_evidence
    from regimepilot.history import HistoryError
    from regimepilot.news import build_news_client
    from regimepilot.smoke_test import build_clients

    try:
        settings = load_settings()
        trading_client, data_client = build_clients(settings)
        option_client = build_option_data_client(settings)
    except ConfigError as error:
        # ConfigError messages are built by us and never contain a credential.
        print(f"configuration error: {error}", file=sys.stderr)
        return 1

    proposal = None
    if forced is None:
        try:
            news_client = build_news_client(settings)
            evidence = observe_evidence(trading_client, data_client, news_client)
            proposal = propose_trade(evidence, stub="--stub" in arguments, settings=settings)
        except (ConfigError, EvidenceError, DecisionError) as error:
            print(f"decision failed: {error}", file=sys.stderr)
            return 1
        action = proposal.action
    else:
        action = forced

    try:
        chain = observe_chain(trading_client, data_client, option_client, action=action)  # type: ignore[arg-type]
    except (ChainError, HistoryError) as error:
        print(f"chain read failed: {error}", file=sys.stderr)
        return 1

    result = select_contract(chain)

    if "--json" in arguments:
        print(json.dumps(json.loads(result.model_dump_json()), indent=2))
        return 0

    if proposal is not None:
        print(format_proposal_summary(proposal))
        print()
    print(format_summary(result))
    # The chain summary's server-clock line, so any clock skew stays visible.
    for line in format_chain_summary(chain).splitlines():
        if "clock" in line:
            print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
