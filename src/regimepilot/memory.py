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

from regimepilot.models import PositionMemo

__all__ = ["load_position_memory"]


def load_position_memory(path: Path) -> Mapping[str, PositionMemo]:
    """One memo per symbol from the journal at ``path``; empty if it is missing.

    Reads every line in order. A submitted ``open`` action starts a fresh
    memo for its symbol (a later re-entry replaces an older one); every
    ``PositionDecision`` seen afterwards updates that symbol's
    ``previous_decision`` as ``"<ACTION>: <reason>"``.
    """
    raise NotImplementedError("lead implements this")
