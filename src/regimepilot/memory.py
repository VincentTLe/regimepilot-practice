"""What the agent remembers between cycles, read back from the journal.

The journal (``logs/cycles.jsonl``, one ``CycleRecord`` per line) is the only
memory store. Before each cycle the runner asks this module for one
``PositionMemo`` per symbol: when the contract was opened, the thesis the
model gave for opening it, and the reason it gave last cycle for holding or
closing it. All of it is best effort: a line that cannot be parsed (an older
journal format, a truncated write) is skipped, and a symbol with no history
simply has no memo.

Pure file reading; no network, no SDK, nothing written here.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from regimepilot.models import CycleRecord, PositionMemo

__all__ = ["load_position_memory"]


def _memo_from_line(line: str) -> CycleRecord | None:
    try:
        return CycleRecord.model_validate_json(line)
    except Exception:  # noqa: BLE001 - an unreadable line is simply not remembered
        return None


def load_position_memory(path: Path) -> Mapping[str, PositionMemo]:
    """One memo per symbol from the journal at ``path``; empty if it is missing.

    Reads every line in order. A submitted ``open`` action starts a fresh
    memo for its symbol (a later re-entry replaces an older one); every
    ``PositionDecision`` seen afterwards updates that symbol's
    ``previous_decision`` as ``"<ACTION>: <reason>"``.
    """
    journal = Path(path)
    if not journal.exists():
        return {}

    memos: dict[str, PositionMemo] = {}
    with journal.open(encoding="utf-8") as lines:
        for raw in lines:
            line = raw.strip()
            if not line:
                continue
            record = _memo_from_line(line)
            if record is None:
                continue

            decision = record.decision
            # The decision precedes the actions in time: a HOLD/CLOSE reason
            # is about a position that already existed when the cycle began.
            if decision is not None:
                for verdict in decision.positions:
                    note = f"{verdict.action}: {verdict.reason}"
                    existing = memos.get(verdict.symbol)
                    memos[verdict.symbol] = (
                        PositionMemo(symbol=verdict.symbol, previous_decision=note)
                        if existing is None
                        else existing.model_copy(update={"previous_decision": note})
                    )

            for action in record.actions:
                if action.kind == "open" and action.outcome == "submitted":
                    thesis = (
                        decision.new_entry.thesis
                        if decision is not None and decision.new_entry is not None
                        else None
                    )
                    memos[action.symbol] = PositionMemo(
                        symbol=action.symbol,
                        entered_at=record.finished_at,
                        entry_thesis=thesis,
                        previous_decision=None,
                    )
    return memos
