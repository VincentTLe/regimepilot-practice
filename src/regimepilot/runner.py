"""One portfolio cycle end to end, a JSONL journal, and a 15-minute loop (MVP).

The runner is the outermost entry point. Since the portfolio agent (approved
2026-08-27, with the three corrections of the same day) one cycle is:

    memory.load_position_memory      journal memos; any failure -> empty memory
    observe_evidence                 features + account + news + gates + portfolio
    decide_portfolio                 LLM (or stub): HOLD/CLOSE per held symbol,
                                     at most one new entry
    --close / --enter overrides      applied AFTER the decision; --enter only
                                     if the deterministic entry pre-check passed
    one CLOSE action per CLOSE       observe_execution_state -> decide_exit
                                     -> submit_paper_order (execute only)
    at most one OPEN action          observe_chain -> select_contract ->
                                     observe_execution_state -> decide_order
                                     -> submit_paper_order (execute only)

Each action carries its own outcome and error; one action's failure never
touches another. The cycle outcome aggregates them: ``submitted`` if any order
went out, else ``planned`` if any was approved in a dry run, else ``rejected``
if actions were requested and none got through, else ``hold`` if positions
exist, else ``wait``. ``error`` is reserved for a failure before any action
could run (the evidence or the decision).

Every cycle, whatever its outcome, is appended as one ``CycleRecord`` line to
``logs/cycles.jsonl``, which is also the agent's only memory store.

SAFETY: submission happens only when ``--execute`` is given; the default is a
dry run that prints and logs every plan. The trading client comes from
``smoke_test.build_clients`` (``paper=True`` hard-coded), so ``--execute`` can
only ever reach the paper endpoint. ``--close`` and ``--enter`` replace only
the model's verdicts: the exit rules, the entry gates, the portfolio caps and
the risk layer still apply to every forced action.

Every pipeline step is imported by name at module level so a test can replace
it on this module; nothing here talks to Alpaca directly.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from regimepilot.account import AccountError
from regimepilot.chain import ChainError, build_option_data_client, observe_chain
from regimepilot.config import ConfigError, Settings, load_settings
from regimepilot.console import tolerant_console
from regimepilot.decision import DecisionError, decide_portfolio
from regimepilot.evidence import EvidenceError, observe_evidence
from regimepilot.execution import ExecutionError, observe_execution_state, submit_paper_order
from regimepilot.features import to_utc
from regimepilot.history import HistoryError
from regimepilot.memory import load_position_memory
from regimepilot.models import (
    UNDERLYING_SYMBOL,
    ActionKind,
    ActionResult,
    CycleOutcome,
    CycleRecord,
    EntryDecision,
    EntryDirection,
    EvidencePacket,
    OpenPositionContext,
    PortfolioDecision,
    PositionDecision,
    PositionMemo,
    RiskDecision,
    RunMode,
    TradeAction,
)
from regimepilot.news import build_news_client
from regimepilot.risk import decide_exit, decide_order
from regimepilot.selector import select_contract
from regimepilot.smoke_test import build_clients

# Approved 2026-08-27: one cycle every fifteen minutes.
DEFAULT_INTERVAL_MINUTES = 15

# One line per cycle, appended. The directory is git-ignored.
DEFAULT_JOURNAL_PATH = Path("logs") / "cycles.jsonl"

# The chain is queried by the old direction vocabulary.
_ENTRY_ACTION: dict[EntryDirection, TradeAction] = {"CALL": "BUY_CALL", "PUT": "BUY_PUT"}

FORCED_CLOSE_REASON = "forced by --close"
FORCED_ENTER_THESIS = "forced by --enter after the entry pre-check passed"

# Errors whose messages this project builds itself, so they may be copied into
# a record. Anything else is named by type only: an unknown exception's text
# may quote the request that failed, keys included.
_STEP_ERRORS = (
    ConfigError,
    EvidenceError,
    DecisionError,
    ChainError,
    HistoryError,
    AccountError,
    ExecutionError,
)

__all__ = [
    "DEFAULT_INTERVAL_MINUTES",
    "DEFAULT_JOURNAL_PATH",
    "append_record",
    "format_summary",
    "new_cycle_id",
    "run_cycle",
    "run_loop",
    "main",
]


def new_cycle_id(now: datetime) -> str:
    """A cycle id from the UTC start time, e.g. ``20260828-143000``.

    Also the middle of every client order id the cycle may use
    (``regimepilot-<cycle_id>-open``, ``-close1``, ``-close2`` ...), so a
    journal line and an Alpaca order can be matched by eye.
    """
    return to_utc(now).strftime("%Y%m%d-%H%M%S")


def _error_text(step: str, error: BaseException) -> str:
    """``"<step>: <message>"`` for our own errors, ``"<step>: <TypeName>"`` for any other."""
    if isinstance(error, _STEP_ERRORS):
        return f"{step}: {error}"
    return f"{step}: {type(error).__name__}"


def _apply_forced(
    decision: PortfolioDecision,
    evidence: EvidencePacket,
    *,
    forced_close: Sequence[str],
    forced_enter: EntryDirection | None,
) -> tuple[PortfolioDecision, str | None]:
    """Overlay ``--close`` / ``--enter`` on the model's decision.

    Applied after the decision so nothing is skipped: a forced CLOSE still has
    to pass the exit rules, and a forced entry exists only if the deterministic
    entry pre-check allowed one. A ``--close`` symbol that is not held is
    ignored. Returns the decision and a short note of what was actually forced
    (``None`` when nothing was).
    """
    portfolio = evidence.portfolio
    held = () if portfolio is None else tuple(p.symbol for p in portfolio.positions)
    update: dict[str, Any] = {}
    notes: list[str] = []

    closes = [symbol for symbol in dict.fromkeys(forced_close) if symbol in held]
    if closes:
        verdicts = {v.symbol: v for v in decision.positions}
        for symbol in closes:
            verdicts[symbol] = PositionDecision(symbol=symbol, action="CLOSE", reason=FORCED_CLOSE_REASON)
        update["positions"] = tuple(verdicts.values())
        notes.append("close=" + ",".join(closes))

    if forced_enter is not None and portfolio is not None and portfolio.entry_allowed:
        update["new_entry"] = EntryDecision(direction=forced_enter, thesis=FORCED_ENTER_THESIS)
        notes.append(f"enter={forced_enter}")

    if not update:
        return decision, None
    return decision.model_copy(update=update), " ".join(notes)


def _settle(
    trading_client: Any,
    *,
    kind: ActionKind,
    symbol: str,
    execute: bool,
    risk: RiskDecision,
    gathered: dict[str, Any],
) -> ActionResult:
    """The last step of any action: refused, planned, or (execute only) submitted."""
    if not risk.approved or risk.plan is None:
        return ActionResult(kind=kind, symbol=symbol, outcome="rejected", **gathered)
    if not execute:
        return ActionResult(kind=kind, symbol=symbol, outcome="planned", **gathered)
    receipt = submit_paper_order(trading_client, risk.plan)
    if receipt.submitted:
        return ActionResult(kind=kind, symbol=symbol, outcome="submitted", receipt=receipt, **gathered)
    return ActionResult(
        kind=kind,
        symbol=symbol,
        outcome="error",
        error=receipt.error or "submit_paper_order: order was not submitted",
        receipt=receipt,
        **gathered,
    )


def _close_action(
    trading_client: Any,
    option_client: Any,
    position: OpenPositionContext,
    *,
    cycle_id: str,
    sequence: int,
    execute: bool,
) -> ActionResult:
    """Sell to close the whole fresh quantity of one held contract, if the exit rules allow."""
    gathered: dict[str, Any] = {}
    step = "observe_execution_state"
    try:
        # No ``now`` here on purpose: the point of this read is that it is fresh.
        state = observe_execution_state(trading_client, option_client, selected=position)
        gathered["execution_state"] = state
        step = "decide_exit"
        risk = decide_exit(position, state, cycle_id=cycle_id, sequence=sequence)
        gathered["risk"] = risk
        step = "submit_paper_order"
        return _settle(
            trading_client, kind="close", symbol=position.symbol, execute=execute, risk=risk, gathered=gathered
        )
    except Exception as error:  # noqa: BLE001 - one action's failure must not touch another
        return ActionResult(
            kind="close", symbol=position.symbol, outcome="error", error=_error_text(step, error), **gathered
        )


def _open_action(
    trading_client: Any,
    data_client: Any,
    option_client: Any,
    entry: EntryDecision,
    *,
    cycle_id: str,
    execute: bool,
    now: datetime,
) -> ActionResult:
    """Buy to open one contract chosen by the deterministic selector, if the risk rules allow.

    ``entry.candidate_id`` is ignored in this pass: the contract always comes
    from ``observe_chain`` + ``select_contract``.
    """
    gathered: dict[str, Any] = {"direction": entry.direction}
    symbol = UNDERLYING_SYMBOL
    step = "observe_chain"
    try:
        chain = observe_chain(
            trading_client, data_client, option_client, action=_ENTRY_ACTION[entry.direction], now=now
        )
        step = "select_contract"
        selection = select_contract(chain)
        gathered["selection"] = selection
        if selection.status != "selected" or selection.selected is None:
            return ActionResult(kind="open", symbol=symbol, outcome="no_contract", **gathered)
        symbol = selection.selected.symbol

        # No ``now`` here on purpose: the point of this read is that it is fresh.
        step = "observe_execution_state"
        state = observe_execution_state(trading_client, option_client, selected=selection.selected)
        gathered["execution_state"] = state
        step = "decide_order"
        risk = decide_order(selection, state, cycle_id=cycle_id)
        gathered["risk"] = risk
        step = "submit_paper_order"
        return _settle(trading_client, kind="open", symbol=symbol, execute=execute, risk=risk, gathered=gathered)
    except Exception as error:  # noqa: BLE001 - one action's failure must not touch another
        return ActionResult(kind="open", symbol=symbol, outcome="error", error=_error_text(step, error), **gathered)


def _aggregate(actions: Sequence[ActionResult], evidence: EvidencePacket) -> CycleOutcome:
    outcomes = {a.outcome for a in actions}
    if "submitted" in outcomes:
        return "submitted"
    if "planned" in outcomes:
        return "planned"
    if actions:
        return "rejected"
    if evidence.portfolio is not None and evidence.portfolio.positions:
        return "hold"
    return "wait"


def run_cycle(
    trading_client: Any,
    data_client: Any,
    option_client: Any,
    news_client: Any,
    settings: Settings,
    *,
    execute: bool,
    stub: bool = False,
    forced_close: Sequence[str] = (),
    forced_enter: EntryDirection | None = None,
    now: datetime | None = None,
    transport: Callable[..., Any] | None = None,
    cycle_id: str | None = None,
    journal_path: Path = DEFAULT_JOURNAL_PATH,
) -> CycleRecord:
    """Run one complete portfolio cycle and return its record.

    Never raises for a trading outcome. A failed evidence read or decision is
    a cycle-level ``error`` with no actions; a failure inside one action is
    that action's ``error`` and the others still run. ``forced_close`` and
    ``forced_enter`` are applied after the decision, so gates, caps and risk
    rules still apply. ``execute=False`` never calls ``submit_paper_order``.
    """
    started = to_utc(now) if now else datetime.now(timezone.utc)
    cycle_id = cycle_id or new_cycle_id(started)
    mode: RunMode = "execute" if execute else "dry_run"

    def stamp() -> datetime:
        return to_utc(now) if now else datetime.now(timezone.utc)

    # Memory is best effort: a journal that cannot be read is an empty memory.
    memory: Mapping[str, PositionMemo]
    try:
        memory = load_position_memory(journal_path)
    except Exception:  # noqa: BLE001 - memory must never stop a cycle
        memory = {}

    evidence: EvidencePacket | None = None
    step = "observe_evidence"
    try:
        evidence = observe_evidence(trading_client, data_client, news_client, now=started, memory=memory)
        step = "decide_portfolio"
        decision = decide_portfolio(evidence, stub=stub, settings=settings, transport=transport)
    except Exception as error:  # noqa: BLE001 - deliberately uniform
        return CycleRecord(
            cycle_id=cycle_id,
            started_at=started,
            finished_at=stamp(),
            mode=mode,
            outcome="error",
            evidence=evidence,
            error=_error_text(step, error),
        )

    decision, forced = _apply_forced(decision, evidence, forced_close=forced_close, forced_enter=forced_enter)

    held = {} if evidence.portfolio is None else {p.symbol: p for p in evidence.portfolio.positions}
    actions: list[ActionResult] = []
    closes = sorted((v for v in decision.positions if v.action == "CLOSE"), key=lambda v: v.symbol)
    for sequence, verdict in enumerate(closes, start=1):
        position = held.get(verdict.symbol)
        if position is None:
            # A CLOSE for a symbol the briefing does not hold: nothing to sell.
            actions.append(
                ActionResult(
                    kind="close", symbol=verdict.symbol, outcome="error", error="find_position: symbol is not held"
                )
            )
            continue
        actions.append(
            _close_action(
                trading_client, option_client, position, cycle_id=cycle_id, sequence=sequence, execute=execute
            )
        )
    if decision.new_entry is not None:
        actions.append(
            _open_action(
                trading_client,
                data_client,
                option_client,
                decision.new_entry,
                cycle_id=cycle_id,
                execute=execute,
                now=started,
            )
        )

    return CycleRecord(
        cycle_id=cycle_id,
        started_at=started,
        finished_at=stamp(),
        mode=mode,
        outcome=_aggregate(actions, evidence),
        forced=forced,
        evidence=evidence,
        decision=decision,
        actions=tuple(actions),
    )


def append_record(record: CycleRecord, path: Path = DEFAULT_JOURNAL_PATH) -> None:
    """Append one compact JSON line (``model_dump_json()``) to the journal, creating the directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as journal:
        journal.write(record.model_dump_json() + "\n")


def _number(value: float | None, digits: int = 2) -> str:
    return "-" if value is None else f"{value:,.{digits}f}"


def _portfolio_lines(evidence: EvidencePacket | None) -> list[str]:
    portfolio = None if evidence is None else evidence.portfolio
    if portfolio is None:
        return []
    entry = "allowed" if portfolio.entry_allowed else f"blocked ({portfolio.entry_blocked_reason})"
    lines = [
        f"  {'portfolio':<15} {portfolio.open_position_count} position(s)"
        f"   cost basis {_number(portfolio.total_cost_basis)}"
        f"   options bp {_number(portfolio.options_buying_power)}   new entry {entry}"
    ]
    for position in portfolio.positions:
        lines.append(
            f"    {position.symbol:<22} {position.option_type:<4} strike {_number(position.strike_price)}"
            f"  {position.days_to_expiration} DTE  qty {position.qty:g}"
            f"  entry {_number(position.avg_entry_price)}  mark {_number(position.current_price)}"
            f"  upl {_number(position.unrealized_pl)}"
            + (f"  pending {position.pending_order_side}" if position.pending_order_side else "")
        )
    return lines


def _decision_lines(decision: PortfolioDecision | None) -> list[str]:
    if decision is None:
        return []
    lines = [f"  {'decision':<15} {v.symbol} {v.action} - {v.reason}" for v in decision.positions]
    entry = decision.new_entry
    lines.append(f"  {'new entry':<15} " + ("none" if entry is None else f"{entry.direction} - {entry.thesis}"))
    lines.append(f"  {'thesis':<15} {decision.portfolio_thesis}")
    lines.append(
        f"  {'model':<15} {decision.model or 'unknown'}   confidence {decision.confidence}"
        + ("   (gate skipped)" if decision.gate_skipped else "")
    )
    return lines


def _action_lines(action: ActionResult) -> list[str]:
    head = f"  {'action':<15} {action.kind} {action.symbol} {action.outcome}"
    if action.direction is not None:
        head += f"   direction {action.direction}"
    selection = action.selection
    if selection is not None and selection.selected is None:
        head += f"   selection {selection.status}" + (f" ({selection.reason})" if selection.reason else "")
    risk = action.risk
    if risk is not None and not risk.approved:
        head += f"   risk refused ({risk.reason})"
    lines = [head]

    plan = None if risk is None else risk.plan
    if plan is not None:
        lines.append(
            f"    {'plan':<13} {plan.side} {plan.qty} x {plan.symbol}   {plan.order_type} {_number(plan.limit_price)}"
            f"   {plan.time_in_force}   {plan.position_intent}   notional {_number(plan.notional_usd)}"
            f"   id {plan.client_order_id}"
        )
    receipt = action.receipt
    if receipt is not None:
        lines.append(
            f"    {'receipt':<13} submitted {'YES' if receipt.submitted else 'no'}"
            f"   order {receipt.order_id or '-'}   status {receipt.status or '-'}"
            f"   filled {_number(receipt.filled_qty, 0)} @ {_number(receipt.filled_avg_price)}"
        )
    if action.error is not None:
        lines.append(f"    {'error':<13} {action.error}")
    return lines


def format_summary(record: CycleRecord) -> str:
    """Human summary of one cycle: portfolio, decision, one block per action, error."""
    lines = [
        f"RegimePilot cycle {record.cycle_id}  {record.mode}  {record.outcome}"
        f"  @ {record.finished_at.strftime('%Y-%m-%d %H:%M:%SZ')}"
    ]
    if record.forced is not None:
        lines.append(f"  {'forced':<15} {record.forced}")
    lines.extend(_portfolio_lines(record.evidence))
    lines.extend(_decision_lines(record.decision))
    for action in record.actions:
        lines.extend(_action_lines(action))
    if record.error is not None:
        lines.append(f"  {'error':<15} {record.error}")
    return "\n".join(lines)


def run_loop(
    run_once: Callable[[], CycleRecord],
    *,
    interval_minutes: int = DEFAULT_INTERVAL_MINUTES,
    sleep: Callable[[float], None] | None = None,
    max_cycles: int | None = None,
) -> int:
    """Call ``run_once`` every ``interval_minutes`` until Ctrl-C (or ``max_cycles``).

    An exception escaping ``run_once`` is printed and the loop continues; a
    ``KeyboardInterrupt`` ends it cleanly with exit code 0. ``sleep`` is
    injectable so a test can run the loop without waiting.

    Printing the record is ``run_once``'s job (see ``main``), so the same
    loop serves the summary and the ``--json`` line.
    """
    pause = sleep or time.sleep
    completed = 0
    try:
        while True:
            try:
                run_once()
            except Exception as error:  # noqa: BLE001 - the loop must outlive one bad cycle
                # Type only: an exception that escaped a cycle is not one we built.
                print(f"cycle failed: {type(error).__name__}", file=sys.stderr)
            completed += 1
            if max_cycles is not None and completed >= max_cycles:
                return 0
            pause(interval_minutes * 60)
    except KeyboardInterrupt:
        print("stopped", file=sys.stderr)
        return 0


def _parse_arguments(arguments: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m regimepilot.runner",
        description="Run one RegimePilot portfolio cycle (dry run by default) or loop every N minutes.",
    )
    parser.add_argument("--execute", action="store_true", help="submit the approved paper orders")
    parser.add_argument("--stub", action="store_true", help="rule-based decision instead of the LLM")
    parser.add_argument(
        "--enter",
        type=str.upper,
        choices=("CALL", "PUT"),
        default=None,
        metavar="CALL|PUT",
        help="force one new entry in this direction if the entry pre-check passes (replaces the model's entry only)",
    )
    parser.add_argument(
        "--close",
        action="append",
        type=str.upper,
        default=[],
        metavar="SYMBOL",
        help="force CLOSE of this held OCC symbol (repeatable); the exit rules still apply",
    )
    parser.add_argument("--loop", action="store_true", help="keep cycling until Ctrl-C")
    parser.add_argument(
        "--interval-minutes",
        type=int,
        default=DEFAULT_INTERVAL_MINUTES,
        help=f"minutes between cycles with --loop (default {DEFAULT_INTERVAL_MINUTES})",
    )
    parser.add_argument("--json", dest="as_json", action="store_true", help="print the record as JSON")
    parser.add_argument(
        "--journal", type=Path, default=DEFAULT_JOURNAL_PATH, help=f"JSONL journal (default {DEFAULT_JOURNAL_PATH})"
    )
    args = parser.parse_args(list(arguments))
    if args.interval_minutes < 1:
        parser.error("--interval-minutes must be at least 1")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m regimepilot.runner [--execute] [--stub] [--enter CALL|PUT]
    [--close SYMBOL ...] [--loop] [--interval-minutes N] [--json] [--journal PATH]``

    Default is a single dry-run cycle. ``--execute`` arms paper submission.
    Exit 1 only for a configuration problem or a bad argument; every trading
    outcome, including ``error``, is recorded and exits 0.
    """
    tolerant_console()
    try:
        args = _parse_arguments(sys.argv[1:] if argv is None else argv)
    except SystemExit as request:
        # argparse has already printed the usage error (or --help) itself.
        return 0 if request.code in (None, 0) else 1

    try:
        settings = load_settings()
        trading_client, data_client = build_clients(settings)
        option_client = build_option_data_client(settings)
        news_client = build_news_client(settings)
    except ConfigError as error:
        # ConfigError messages are built by us and never contain a credential.
        print(f"configuration error: {error}", file=sys.stderr)
        return 1

    if args.execute:
        print("ARMED: paper order submission is enabled (paper account only)", file=sys.stderr)

    def run_once() -> CycleRecord:
        record = run_cycle(
            trading_client,
            data_client,
            option_client,
            news_client,
            settings,
            execute=args.execute,
            stub=args.stub,
            forced_close=tuple(args.close),
            forced_enter=args.enter,
            journal_path=args.journal,
        )
        # Journal first: the record must survive a console that cannot print it.
        append_record(record, args.journal)
        # flush: a loop piped to a file or `tee` must show each cycle as it
        # ends, not when the process exits.
        if not args.as_json:
            print(format_summary(record), flush=True)
        elif args.loop:
            print(record.model_dump_json(), flush=True)
        else:
            print(json.dumps(json.loads(record.model_dump_json()), indent=2), flush=True)
        return record

    if args.loop:
        return run_loop(run_once, interval_minutes=args.interval_minutes)
    run_once()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
