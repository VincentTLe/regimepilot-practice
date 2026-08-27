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
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from regimepilot.config import Settings
from regimepilot.models import CycleRecord, TradeAction

# Approved 2026-08-27: one cycle every fifteen minutes.
DEFAULT_INTERVAL_MINUTES = 15

# One line per cycle, appended. The directory is git-ignored.
DEFAULT_JOURNAL_PATH = Path("logs") / "cycles.jsonl"

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
    raise NotImplementedError("Worker C implements this")


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
    raise NotImplementedError("Worker C implements this")


def append_record(record: CycleRecord, path: Path = DEFAULT_JOURNAL_PATH) -> None:
    """Append one compact JSON line (``model_dump_json()``) to the journal, creating the directory."""
    raise NotImplementedError("Worker C implements this")


def format_summary(record: CycleRecord) -> str:
    """Human summary of one cycle: outcome, proposal, contract, risk, plan, receipt."""
    raise NotImplementedError("Worker C implements this")


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
    """
    raise NotImplementedError("Worker C implements this")


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m regimepilot.runner [--execute] [--stub] [--action BUY_CALL|BUY_PUT]
    [--loop] [--interval-minutes N] [--json]``

    Default is a single dry-run cycle. ``--execute`` arms paper submission.
    Exit 1 only for a configuration problem or a bad argument; every trading
    outcome, including ``error``, is recorded and exits 0.
    """
    raise NotImplementedError("Worker C implements this")


if __name__ == "__main__":
    raise SystemExit(main())
