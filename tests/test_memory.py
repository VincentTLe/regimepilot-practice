"""Memory tests: what the journal remembers per symbol."""

from datetime import datetime, timedelta, timezone

from regimepilot.memory import load_position_memory
from regimepilot.models import (
    ActionResult,
    CycleRecord,
    EntryDecision,
    OrderReceipt,
    PortfolioDecision,
    PositionDecision,
)

T0 = datetime(2026, 8, 27, 14, 0, tzinfo=timezone.utc)
SPY_CALL = "SPY260902C00765000"
SPY_PUT = "SPY260904P00760000"


def record(cycle, *, decision=None, actions=(), outcome="wait"):
    started = T0 + timedelta(minutes=15 * cycle)
    return CycleRecord(
        cycle_id=f"c{cycle}",
        started_at=started,
        finished_at=started + timedelta(seconds=5),
        mode="execute",
        outcome=outcome,
        decision=decision,
        actions=tuple(actions),
    )


def decision(*, positions=(), new_entry=None):
    return PortfolioDecision(
        observed_at=T0,
        positions=tuple(PositionDecision(symbol=s, action=a, reason=r) for s, a, r in positions),
        new_entry=new_entry,
        confidence="medium",
        portfolio_thesis="test",
        model="stub",
    )


def submitted_open(symbol):
    return ActionResult(
        kind="open",
        symbol=symbol,
        direction="CALL",
        receipt=OrderReceipt(submitted=True, order_id="o-1", client_order_id="regimepilot-c1-open"),
        outcome="submitted",
    )


def write(path, *lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_a_missing_journal_means_no_memory(tmp_path):
    assert load_position_memory(tmp_path / "logs" / "cycles.jsonl") == {}


def test_entry_thesis_and_previous_decision_are_remembered_per_symbol(tmp_path):
    path = tmp_path / "cycles.jsonl"
    opened = record(
        1,
        decision=decision(new_entry=EntryDecision(direction="CALL", thesis="momentum up")),
        actions=[submitted_open(SPY_CALL)],
        outcome="submitted",
    )
    held = record(
        2,
        decision=decision(positions=[(SPY_CALL, "HOLD", "thesis intact"), (SPY_PUT, "CLOSE", "flipped")]),
        outcome="hold",
    )
    write(
        path,
        "{not json at all",
        '{"cycle_id": "old-format", "outcome": "hold"}',
        opened.model_dump_json(),
        "",
        held.model_dump_json(),
    )

    memory = load_position_memory(path)

    call = memory[SPY_CALL]
    assert call.entered_at == opened.finished_at
    assert call.entry_thesis == "momentum up"
    assert call.previous_decision == "HOLD: thesis intact"
    # A symbol only ever mentioned in a decision still keeps its last reason.
    assert memory[SPY_PUT].previous_decision == "CLOSE: flipped"
    assert memory[SPY_PUT].entered_at is None


def test_a_re_entry_replaces_the_older_memo(tmp_path):
    path = tmp_path / "cycles.jsonl"
    first = record(
        1,
        decision=decision(new_entry=EntryDecision(direction="CALL", thesis="first")),
        actions=[submitted_open(SPY_CALL)],
        outcome="submitted",
    )
    later = record(3, decision=decision(positions=[(SPY_CALL, "CLOSE", "done")]), outcome="hold")
    again = record(
        5,
        decision=decision(new_entry=EntryDecision(direction="CALL", thesis="second")),
        actions=[submitted_open(SPY_CALL)],
        outcome="submitted",
    )
    write(path, first.model_dump_json(), later.model_dump_json(), again.model_dump_json())

    memo = load_position_memory(path)[SPY_CALL]

    assert memo.entry_thesis == "second"
    assert memo.entered_at == again.finished_at
    assert memo.previous_decision is None
