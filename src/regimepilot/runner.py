"""One trading cycle end to end, a JSONL journal, and a 15-minute loop (MVP).

The runner is the outermost entry point. It stitches the existing modules in
this fixed order and stops at the first step that ends the cycle:

    observe_evidence            features + account + news + gates
      gates fail ------------------------------------> outcome "hold"
    propose_trade / forced action (only after gates passed)
      HOLD ------------------------------------------> outcome "hold"
    observe_chain -> select_contract
      no contract -----------------------------------> outcome "no_contract"
    execution.observe_execution_state                fresh account, clock, quote
    risk.decide_order
      refused ---------------------------------------> outcome "rejected"
      approved, dry run -----------------------------> outcome "planned"
    execution.submit_paper_order (only with execute=True)
      Alpaca returned an order --------------------> outcome "submitted"
    any read/submit failure -------------------------> outcome "error"

Every cycle, whatever its outcome, is appended as one ``CycleRecord`` line to
``logs/cycles.jsonl`` so a HOLD, a refusal and an error are as visible as a
trade.

``--action BUY_CALL|BUY_PUT`` replaces only the LLM call: evidence and gates
always run first and a failed gate is a HOLD regardless of the flag. Risk
rules always apply. Nothing in this module bypasses a gate.

SAFETY: submission happens only when ``--execute`` is given; the default is a
dry run that prints and logs the plan. The trading client comes from
``smoke_test.build_clients`` (``paper=True`` hard-coded), so ``--execute``
can only ever reach the paper endpoint. There is no live mode and no flag
that could create one.

Every pipeline step is imported by name at module level so a test can replace
it on this module; nothing here talks to Alpaca directly.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from regimepilot.account import AccountError
from regimepilot.chain import ChainError, build_option_data_client, observe_chain
from regimepilot.config import ConfigError, Settings, load_settings
from regimepilot.console import tolerant_console
from regimepilot.decision import DecisionError, build_gate_hold_proposal, propose_trade
from regimepilot.evidence import EvidenceError, observe_evidence
from regimepilot.execution import ExecutionError, observe_execution_state, submit_paper_order
from regimepilot.features import quote_age_seconds, to_utc
from regimepilot.history import HistoryError
from regimepilot.models import CycleRecord, EvidencePacket, RunMode, TradeAction, TradeProposal
from regimepilot.news import build_news_client
from regimepilot.risk import decide_order
from regimepilot.selector import select_contract
from regimepilot.smoke_test import build_clients

# Approved 2026-08-27: one cycle every fifteen minutes.
DEFAULT_INTERVAL_MINUTES = 15

# One line per cycle, appended. The directory is git-ignored.
DEFAULT_JOURNAL_PATH = Path("logs") / "cycles.jsonl"

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

    Also the tail of the client order id, so one cycle can place at most one
    order and the journal line and the Alpaca order can be matched by eye.
    """
    return to_utc(now).strftime("%Y%m%d-%H%M%S")


def _forced_proposal(evidence: EvidencePacket, action: TradeAction) -> TradeProposal:
    """The proposal ``--action`` stands in for, built only after the gates passed."""
    return TradeProposal(
        observed_at=evidence.observed_at,
        symbol=evidence.symbol,
        action=action,
        confidence="medium",
        thesis="Direction forced by --action after pre-gates passed.",
        evidence_used=("cli.action",),
        model="forced",
    )


def run_cycle(
    trading_client: Any,
    data_client: Any,
    option_client: Any,
    news_client: Any,
    settings: Settings,
    *,
    execute: bool,
    stub: bool = False,
    forced_action: TradeAction | None = None,
    now: datetime | None = None,
    transport: Callable[..., Any] | None = None,
    cycle_id: str | None = None,
) -> CycleRecord:
    """Run one complete cycle and return its record. Never raises for a
    trading outcome; a failed read or submission becomes outcome ``error``.

    ``forced_action`` is applied only after ``observe_evidence`` ran and its
    gates passed. ``transport`` is forwarded to ``propose_trade`` so a test
    can drive the LLM path offline. ``execute=False`` never calls
    ``submit_paper_order``.
    """
    started = to_utc(now) if now else datetime.now(timezone.utc)
    cycle_id = cycle_id or new_cycle_id(started)
    mode: RunMode = "execute" if execute else "dry_run"

    # Everything gathered before the cycle ended, attached whatever the outcome.
    gathered: dict[str, Any] = {}

    def finish(outcome: str, *, error: str | None = None) -> CycleRecord:
        return CycleRecord(
            cycle_id=cycle_id,
            started_at=started,
            finished_at=to_utc(now) if now else datetime.now(timezone.utc),
            mode=mode,
            outcome=outcome,  # type: ignore[arg-type]
            forced_action=forced_action,
            error=error,
            **gathered,
        )

    step = "observe_evidence"
    try:
        evidence = observe_evidence(trading_client, data_client, news_client, now=started)

        step = "propose_trade"
        if forced_action is None:
            proposal = propose_trade(evidence, stub=stub, settings=settings, transport=transport)
        elif not evidence.gates.passed:
            # The flag replaces the LLM, never a gate.
            proposal = build_gate_hold_proposal(evidence)
        else:
            proposal = _forced_proposal(evidence, forced_action)
        gathered["proposal"] = proposal
        if proposal.action == "HOLD":
            return finish("hold")

        step = "observe_chain"
        chain = observe_chain(
            trading_client, data_client, option_client, action=proposal.action, now=started
        )
        step = "select_contract"
        selection = select_contract(chain)
        gathered["selection"] = selection
        if selection.status != "selected" or selection.selected is None:
            return finish("no_contract")

        # No ``now`` here on purpose: the point of this read is that it is fresh.
        step = "observe_execution_state"
        state = observe_execution_state(trading_client, option_client, selected=selection.selected)
        gathered["execution_state"] = state

        step = "decide_order"
        risk = decide_order(selection, state, cycle_id=cycle_id)
        gathered["risk"] = risk
        if not risk.approved or risk.plan is None:
            return finish("rejected")
        if not execute:
            return finish("planned")

        step = "submit_paper_order"
        receipt = submit_paper_order(trading_client, risk.plan)
        gathered["receipt"] = receipt
        if receipt.submitted:
            return finish("submitted")
        return finish("error", error=receipt.error or f"{step}: order was not submitted")
    except _STEP_ERRORS as error:
        return finish("error", error=f"{step}: {error}")
    except Exception as error:  # noqa: BLE001 - deliberately uniform
        return finish("error", error=f"{step}: {type(error).__name__}")


def append_record(record: CycleRecord, path: Path = DEFAULT_JOURNAL_PATH) -> None:
    """Append one compact JSON line (``model_dump_json()``) to the journal, creating the directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as journal:
        journal.write(record.model_dump_json() + "\n")


def _number(value: float | None, digits: int = 2) -> str:
    return "-" if value is None else f"{value:,.{digits}f}"


def _flag(value: bool | None) -> str:
    if value is None:
        return "-"
    return "YES" if value else "no"


def format_summary(record: CycleRecord) -> str:
    """Human summary of one cycle: outcome, proposal, contract, risk, plan, receipt."""
    lines = [
        f"RegimePilot cycle {record.cycle_id}  {record.mode}  {record.outcome}"
        f"  @ {record.finished_at.strftime('%Y-%m-%d %H:%M:%SZ')}"
    ]
    if record.forced_action is not None:
        lines.append(f"  {'forced action':<15} {record.forced_action}")

    proposal = record.proposal
    if proposal is not None:
        lines.append(
            f"  {'proposal':<15} {proposal.action}   confidence {proposal.confidence}"
            f"   model {proposal.model or 'unknown'}"
        )
        lines.append(f"  {'thesis':<15} {proposal.thesis}")

    selection = record.selection
    if selection is not None:
        chosen = selection.selected
        if chosen is None:
            reason = f"  ({selection.reason})" if selection.reason else ""
            lines.append(f"  {'selection':<15} {selection.status}{reason}")
        else:
            lines.append(
                f"  {'selected':<15} {chosen.symbol}   strike {_number(chosen.strike_price)}"
                f"   bid {_number(chosen.bid)}  ask {_number(chosen.ask)}  mid {_number(chosen.mid, 3)}"
                f"   age {_number(chosen.quote_age_seconds, 1)} s"
            )

    state = record.execution_state
    if state is not None:
        account = state.account
        lines.append(
            f"  {'account':<15} spy option position {_flag(account.has_open_spy_option_position)}"
            f"   open order {_flag(account.has_open_spy_option_order)}"
            f"   options bp {_number(account.options_buying_power)}"
        )
        lines.append(
            f"  {'clock':<15} open {_flag(state.market_is_open)}"
            f"   minutes to close {_number(state.minutes_to_close, 1)}"
        )
        quote = state.quote
        age = None if quote.server_time is None else quote_age_seconds(quote.quote_at, quote.server_time)
        lines.append(
            f"  {'fresh quote':<15} {quote.symbol}   bid {_number(quote.bid)}  ask {_number(quote.ask)}"
            f"   age {_number(age, 1)} s   verdict {quote.reject_reason or 'acceptable'}"
        )

    risk = record.risk
    if risk is not None:
        lines.append(f"  {'risk':<15} " + ("approved" if risk.approved else f"refused  ({risk.reason})"))
        plan = risk.plan
        if plan is not None:
            lines.append(
                f"  {'plan':<15} {plan.side} {plan.qty} x {plan.symbol}   {plan.order_type}"
                f" {_number(plan.limit_price)}   {plan.time_in_force}   {plan.position_intent}"
                f"   max premium {_number(plan.max_premium_usd)}   id {plan.client_order_id}"
            )

    receipt = record.receipt
    if receipt is not None:
        lines.append(
            f"  {'receipt':<15} submitted {_flag(receipt.submitted)}   order {receipt.order_id or '-'}"
            f"   status {receipt.status or '-'}   filled {_number(receipt.filled_qty, 0)}"
            f" @ {_number(receipt.filled_avg_price)}"
            + (f"   error {receipt.error}" if receipt.error else "")
        )

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
        description="Run one RegimePilot cycle (dry run by default) or loop every N minutes.",
    )
    parser.add_argument("--execute", action="store_true", help="submit the approved paper order")
    parser.add_argument("--stub", action="store_true", help="rule-based proposal instead of the LLM")
    parser.add_argument(
        "--action",
        type=str.upper,
        choices=("BUY_CALL", "BUY_PUT"),
        default=None,
        metavar="BUY_CALL|BUY_PUT",
        help="force the direction after the pre-gates passed (replaces the LLM call only)",
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
    """``python -m regimepilot.runner [--execute] [--stub] [--action BUY_CALL|BUY_PUT]
    [--loop] [--interval-minutes N] [--json]``

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
            forced_action=args.action,
        )
        # Journal first: the record must survive a console that cannot print it.
        append_record(record, args.journal)
        if not args.as_json:
            print(format_summary(record))
        elif args.loop:
            print(record.model_dump_json())
        else:
            print(json.dumps(json.loads(record.model_dump_json()), indent=2))
        return record

    if args.loop:
        return run_loop(run_once, interval_minutes=args.interval_minutes)
    run_once()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
