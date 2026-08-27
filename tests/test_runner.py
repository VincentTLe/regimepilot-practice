"""Runner tests: one portfolio cycle end to end with every pipeline step
replaced by a fake, the JSONL journal, the loop and the CLI.

Offline by construction: every ``runner.<step>`` name is monkeypatched with a
recorder, the clients are sentinels, and no credential is ever read.
"""

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from regimepilot import runner
from regimepilot.config import ConfigError, load_settings
from regimepilot.decision import DecisionError
from regimepilot.evidence import EvidenceError
from regimepilot.models import (
    AccountHint,
    AccountState,
    ActionResult,
    ChainPacket,
    CycleRecord,
    EntryDecision,
    EvidencePacket,
    ExecutionState,
    FreshQuote,
    GatesEvidence,
    NewsEvidence,
    OpenPositionContext,
    OrderPlan,
    OrderReceipt,
    PortfolioContext,
    PortfolioDecision,
    PositionDecision,
    PositionMemo,
    RiskDecision,
    SelectedContract,
    SelectionResult,
    UnderlyingEvidence,
)
from regimepilot.risk import DEFAULT_LIMITS
from regimepilot.runner import (
    DEFAULT_INTERVAL_MINUTES,
    DEFAULT_JOURNAL_PATH,
    append_record,
    format_summary,
    main,
    new_cycle_id,
    run_cycle,
    run_loop,
)

NOW = datetime(2026, 8, 27, 14, 30, tzinfo=timezone.utc)
TODAY = date(2026, 8, 27)
CYCLE_ID = "20260827-143000"

# Two held contracts (A sorts before B) and the contract the selector picks.
A = "SPY260902C00765000"
B = "SPY260904P00760000"
ENTRY = "SPY260903C00765000"

API_KEY = "SUPER-SECRET-KEY"

# Sentinels: the runner must forward these untouched and never call them.
TRADING = object()
DATA = object()
OPTION = object()
NEWS = object()
SETTINGS = load_settings(
    {"ALPACA_API_KEY": "test-key", "ALPACA_SECRET_KEY": "test-secret", "ALPACA_PAPER": "true"}
)


# --------------------------------------------------------------------------
# builders: one real model instance per pipeline stage
# --------------------------------------------------------------------------


def position(symbol, option_type, strike, expiration):
    return OpenPositionContext(
        symbol=symbol,
        option_type=option_type,
        strike_price=strike,
        expiration_date=expiration,
        days_to_expiration=(expiration - TODAY).days,
        qty=1.0,
        avg_entry_price=5.0,
        cost_basis=500.0,
        current_price=5.5,
        unrealized_pl=50.0,
        unrealized_plpc=0.1,
    )


POSITION_A = position(A, "call", 765.0, date(2026, 9, 2))
POSITION_B = position(B, "put", 760.0, date(2026, 9, 4))


def evidence_with(positions=(), *, entry_allowed=True, reason=None):
    positions = tuple(positions)
    portfolio = PortfolioContext(
        positions=positions,
        open_position_count=len(positions),
        total_cost_basis=500.0 * len(positions),
        options_buying_power=98000.75,
        equity=100000.0,
        entry_allowed=entry_allowed,
        entry_blocked_reason=None if entry_allowed else (reason or "max_positions"),
        limits=DEFAULT_LIMITS,
    )
    return EvidencePacket(
        observed_at=NOW,
        symbol="SPY",
        gates=GatesEvidence(passed=True, momentum_align="aligned_up"),
        underlying=UnderlyingEvidence(data_feed="iex", market_is_open=True, minutes_to_close=300.0),
        news=NewsEvidence(),
        account=AccountHint(has_open_option_position=bool(positions)),
        portfolio=portfolio,
    )


def entry(direction="CALL"):
    return EntryDecision(direction=direction, thesis="Test entry.")


def decision_with(positions=None, new_entry=None):
    verdicts = tuple(
        PositionDecision(symbol=symbol, action=action, reason=f"test {action.lower()}")
        for symbol, action in (positions or {}).items()
    )
    return PortfolioDecision(
        observed_at=NOW,
        positions=verdicts,
        new_entry=new_entry,
        confidence="medium",
        portfolio_thesis="Test thesis.",
        evidence_used=("gates.momentum_align",),
        model="stub",
    )


def chain_packet(action="BUY_CALL"):
    return ChainPacket(
        observed_at=NOW, action=action, option_feed="indicative", underlying_mid=765.0, quotes_read_at=NOW
    )


def selected_contract():
    return SelectedContract(
        symbol=ENTRY,
        option_type="call",
        strike_price=765.0,
        expiration_date=date(2026, 9, 3),
        days_to_expiration=7,
        bid=4.9,
        ask=5.0,
        mid=4.95,
        spread_bps=202.0,
        quote_at=NOW - timedelta(seconds=1),
        quote_age_seconds=1.0,
        underlying_mid=765.0,
    )


def selection(*, selected=True, action="BUY_CALL"):
    if selected:
        return SelectionResult(observed_at=NOW, action=action, status="selected", selected=selected_contract())
    return SelectionResult(observed_at=NOW, action=action, status="no_contract", reason="no_candidates")


def execution_state(symbol):
    account = AccountState(observed_at=NOW, account_id_masked="****7888", options_buying_power=98000.75)
    quote = FreshQuote(symbol=symbol, bid=4.9, ask=5.0, quote_at=NOW - timedelta(seconds=1), server_time=NOW)
    return ExecutionState(observed_at=NOW, account=account, market_is_open=True, minutes_to_close=120.0, quote=quote)


def sell_plan(symbol, sequence=1):
    return OrderPlan(
        symbol=symbol,
        side="sell",
        qty=1,
        position_intent="sell_to_close",
        limit_price=4.9,
        notional_usd=490.0,
        client_order_id=f"regimepilot-{CYCLE_ID}-close{sequence}",
    )


def buy_plan(symbol=ENTRY):
    return OrderPlan(
        symbol=symbol,
        side="buy",
        qty=1,
        position_intent="buy_to_open",
        limit_price=5.0,
        notional_usd=500.0,
        client_order_id=f"regimepilot-{CYCLE_ID}-open",
    )


def approved(plan):
    return RiskDecision(approved=True, plan=plan)


def refused(reason):
    return RiskDecision(approved=False, reason=reason)


# Fakes that answer from their arguments, the way the real functions would.


def fake_execution_state(trading_client, option_client, *, selected):
    return execution_state(selected.symbol)


def fake_decide_exit(position, state, *, cycle_id, sequence=1):
    return approved(sell_plan(position.symbol, sequence))


def fake_decide_order(selection, state, *, cycle_id):
    return approved(buy_plan(selection.selected.symbol))


def fake_submit(trading_client, plan):
    return OrderReceipt(
        submitted=True,
        order_id=f"order-{plan.client_order_id}",
        client_order_id=plan.client_order_id,
        status="accepted",
        submitted_at=NOW,
        filled_qty=0.0,
    )


class Pipeline:
    """Every pipeline step as a recorder returning a canned answer, calling a
    fake, or raising an exception. ``calls`` keeps the order and arguments of
    every call so a test can assert what ran and what did not.

    Defaults: two held positions (A, B), the model HOLDs both, no entry.
    """

    STEPS = (
        "load_position_memory",
        "observe_evidence",
        "decide_portfolio",
        "observe_chain",
        "select_contract",
        "observe_execution_state",
        "decide_exit",
        "decide_order",
        "submit_paper_order",
    )

    def __init__(self, monkeypatch, **answers):
        defaults = {
            "load_position_memory": {},
            "observe_evidence": evidence_with((POSITION_A, POSITION_B)),
            "decide_portfolio": decision_with({A: "HOLD", B: "HOLD"}),
            "observe_chain": chain_packet(),
            "select_contract": selection(),
            "observe_execution_state": fake_execution_state,
            "decide_exit": fake_decide_exit,
            "decide_order": fake_decide_order,
            "submit_paper_order": fake_submit,
        }
        defaults.update(answers)
        self.calls = []
        for step in self.STEPS:
            monkeypatch.setattr(runner, step, self._recorder(step, defaults[step]))

    def _recorder(self, step, answer):
        def fake(*args, **kwargs):
            self.calls.append((step, args, kwargs))
            if isinstance(answer, BaseException):
                raise answer
            if callable(answer):
                return answer(*args, **kwargs)
            return answer

        return fake

    def called(self, step):
        return [(args, kwargs) for name, args, kwargs in self.calls if name == step]

    def ran(self):
        return [name for name, _args, _kwargs in self.calls]


def cycle(*, execute=False, forced_close=(), forced_enter=None, stub=True, journal_path=DEFAULT_JOURNAL_PATH):
    return run_cycle(
        TRADING,
        DATA,
        OPTION,
        NEWS,
        SETTINGS,
        execute=execute,
        stub=stub,
        forced_close=forced_close,
        forced_enter=forced_enter,
        now=NOW,
        cycle_id=CYCLE_ID,
        journal_path=journal_path,
    )


def outcomes(record):
    return [(a.kind, a.symbol, a.outcome) for a in record.actions]


# --------------------------------------------------------------------------
# 1. cycle id
# --------------------------------------------------------------------------


def test_cycle_id_is_the_utc_start_time():
    new_york = datetime(2026, 8, 27, 10, 30, tzinfo=ZoneInfo("America/New_York"))
    assert new_cycle_id(new_york) == CYCLE_ID
    assert new_cycle_id(NOW) == CYCLE_ID


# --------------------------------------------------------------------------
# 2. actions: one per CLOSE verdict, at most one entry, each independent
# --------------------------------------------------------------------------


def test_hold_a_close_b_sells_only_b(monkeypatch):
    pipeline = Pipeline(monkeypatch, decide_portfolio=decision_with({A: "HOLD", B: "CLOSE"}))

    record = cycle(execute=True)

    assert isinstance(record, CycleRecord)
    assert record.outcome == "submitted"
    assert record.mode == "execute"
    assert record.forced is None
    assert outcomes(record) == [("close", B, "submitted")]
    assert pipeline.called("observe_execution_state") == [((TRADING, OPTION), {"selected": POSITION_B})]
    assert pipeline.called("decide_exit") == [
        ((POSITION_B, execution_state(B)), {"cycle_id": CYCLE_ID, "sequence": 1})
    ]
    assert pipeline.called("submit_paper_order") == [((TRADING, sell_plan(B)), {})]
    assert pipeline.called("observe_chain") == [] and pipeline.called("decide_order") == []
    (action,) = record.actions
    assert action.risk.plan == sell_plan(B)
    assert action.receipt.client_order_id == f"regimepilot-{CYCLE_ID}-close1"
    (evidence_args, evidence_kwargs), = pipeline.called("observe_evidence")
    assert evidence_args == (TRADING, DATA, NEWS)
    assert evidence_kwargs == {"now": NOW, "memory": {}}
    (decision_args, decision_kwargs), = pipeline.called("decide_portfolio")
    assert decision_args == (evidence_with((POSITION_A, POSITION_B)),)
    assert decision_kwargs == {"stub": True, "settings": SETTINGS, "transport": None}


def test_one_failed_action_never_touches_the_others(monkeypatch):
    def exit_or_boom(position, state, *, cycle_id, sequence=1):
        if position.symbol == A:
            raise RuntimeError(f'401 unauthorized for API_KEY="{API_KEY}"')
        return approved(sell_plan(position.symbol, sequence))

    pipeline = Pipeline(
        monkeypatch,
        decide_portfolio=decision_with({A: "CLOSE", B: "CLOSE"}, new_entry=entry("CALL")),
        decide_exit=exit_or_boom,
    )

    record = cycle(execute=True)

    assert record.outcome == "submitted"
    assert record.error is None
    assert outcomes(record) == [("close", A, "error"), ("close", B, "submitted"), ("open", ENTRY, "submitted")]
    failed, closed, opened = record.actions
    assert failed.error == "decide_exit: RuntimeError"
    assert failed.execution_state is not None and failed.risk is None and failed.receipt is None
    assert closed.risk.plan == sell_plan(B, 2)
    assert opened.direction == "CALL" and opened.risk.plan == buy_plan()
    assert [kwargs["sequence"] for _args, kwargs in pipeline.called("decide_exit")] == [1, 2]
    assert pipeline.called("submit_paper_order") == [((TRADING, sell_plan(B, 2)), {}), ((TRADING, buy_plan()), {})]
    for blob in (record.model_dump_json(), format_summary(record)):
        assert API_KEY not in blob


def test_a_dry_run_plans_every_approved_action_and_never_submits(monkeypatch):
    pipeline = Pipeline(monkeypatch, decide_portfolio=decision_with({A: "HOLD", B: "CLOSE"}, new_entry=entry("CALL")))

    record = cycle(execute=False)

    assert record.outcome == "planned"
    assert record.mode == "dry_run"
    assert outcomes(record) == [("close", B, "planned"), ("open", ENTRY, "planned")]
    assert all(a.receipt is None for a in record.actions)
    assert record.actions[0].risk.plan == sell_plan(B)
    assert record.actions[1].risk.plan == buy_plan()
    assert pipeline.called("submit_paper_order") == []


def test_a_refused_submission_is_an_action_error_carrying_the_receipt(monkeypatch):
    refusal = OrderReceipt(submitted=False, error="failed to submit order: APIError")
    Pipeline(monkeypatch, decide_portfolio=decision_with({A: "CLOSE", B: "HOLD"}), submit_paper_order=refusal)

    record = cycle(execute=True)

    assert record.outcome == "rejected"
    (action,) = record.actions
    assert action.outcome == "error"
    assert action.error == "failed to submit order: APIError"
    assert action.receipt == refusal
    assert action.risk.approved is True


# --------------------------------------------------------------------------
# 3. --close / --enter are applied after the decision, never around a gate
# --------------------------------------------------------------------------


def test_forced_close_turns_a_hold_into_a_close_and_ignores_unknown_symbols(monkeypatch):
    pipeline = Pipeline(monkeypatch)

    record = cycle(forced_close=(A, "SPY991231C00001000"))

    assert record.forced == f"close={A}"
    verdicts = {v.symbol: v for v in record.decision.positions}
    assert verdicts[A].action == "CLOSE" and "--close" in verdicts[A].reason
    assert verdicts[B].action == "HOLD"
    assert outcomes(record) == [("close", A, "planned")]
    ((args, kwargs),) = pipeline.called("decide_exit")
    assert args[0] == POSITION_A and kwargs == {"cycle_id": CYCLE_ID, "sequence": 1}


def test_forced_enter_is_ignored_when_the_entry_pre_check_failed(monkeypatch):
    pipeline = Pipeline(
        monkeypatch,
        observe_evidence=evidence_with((POSITION_A,), entry_allowed=False, reason="max_positions"),
        decide_portfolio=decision_with({A: "HOLD"}),
    )

    record = cycle(forced_enter="PUT")

    assert record.outcome == "hold"
    assert record.forced is None
    assert record.decision.new_entry is None
    assert record.actions == ()
    assert pipeline.called("observe_chain") == []


def test_forced_enter_opens_when_the_entry_pre_check_passed(monkeypatch):
    pipeline = Pipeline(monkeypatch, observe_evidence=evidence_with(()), decide_portfolio=decision_with())

    record = cycle(forced_enter="CALL")

    assert record.forced == "enter=CALL"
    assert record.decision.new_entry.direction == "CALL"
    assert "--enter" in record.decision.new_entry.thesis
    assert outcomes(record) == [("open", ENTRY, "planned")]
    assert record.actions[0].direction == "CALL"
    ((_args, kwargs),) = pipeline.called("observe_chain")
    assert kwargs["action"] == "BUY_CALL"


# --------------------------------------------------------------------------
# 4. entry path
# --------------------------------------------------------------------------


def test_no_contract_ends_the_entry_before_the_fresh_reread(monkeypatch):
    pipeline = Pipeline(
        monkeypatch,
        observe_evidence=evidence_with(()),
        decide_portfolio=decision_with(new_entry=entry("CALL")),
        select_contract=selection(selected=False),
    )

    record = cycle(execute=True)

    assert record.outcome == "rejected"
    ((chain_args, chain_kwargs),) = pipeline.called("observe_chain")
    assert chain_args == (TRADING, DATA, OPTION)
    assert chain_kwargs["action"] == "BUY_CALL"
    (action,) = record.actions
    assert action == ActionResult(
        kind="open", symbol="SPY", direction="CALL", selection=selection(selected=False), outcome="no_contract"
    )
    assert pipeline.called("observe_execution_state") == []
    assert pipeline.called("decide_order") == []
    assert pipeline.called("submit_paper_order") == []


def test_the_entry_forwards_the_selection_and_fresh_state_to_risk(monkeypatch):
    pipeline = Pipeline(monkeypatch, observe_evidence=evidence_with(()), decide_portfolio=decision_with(new_entry=entry("PUT")),
                        observe_chain=chain_packet("BUY_PUT"), select_contract=selection(action="BUY_PUT"))

    record = cycle(execute=True)

    assert outcomes(record) == [("open", ENTRY, "submitted")]
    assert pipeline.called("observe_chain")[0][1]["action"] == "BUY_PUT"
    assert pipeline.called("observe_execution_state") == [((TRADING, OPTION), {"selected": selected_contract()})]
    assert pipeline.called("decide_order") == [
        ((selection(action="BUY_PUT"), execution_state(ENTRY)), {"cycle_id": CYCLE_ID})
    ]
    assert pipeline.called("submit_paper_order") == [((TRADING, buy_plan()), {})]


# --------------------------------------------------------------------------
# 5. cycle-level failures: error, no actions, evidence kept if it was read
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("step", "error", "evidence_attached"),
    [
        ("observe_evidence", EvidenceError("failed to read account: Boom"), False),
        ("decide_portfolio", DecisionError("openrouter request failed: HTTP 500"), True),
        ("decide_portfolio", ConfigError("Missing OPENROUTER_API_KEY."), True),
    ],
)
def test_a_cycle_level_failure_is_an_error_with_no_actions(monkeypatch, step, error, evidence_attached):
    pipeline = Pipeline(monkeypatch, **{step: error})

    record = cycle(execute=True)

    assert record.outcome == "error"
    assert record.error == f"{step}: {error}"
    assert record.actions == ()
    assert record.decision is None
    assert (record.evidence is not None) == evidence_attached
    assert pipeline.ran()[-1] == step
    assert pipeline.called("submit_paper_order") == []


def test_an_unexpected_cycle_failure_is_named_by_type_only(monkeypatch):
    class Boom(RuntimeError):
        def __init__(self):
            super().__init__(f'401 unauthorized for API_KEY="{API_KEY}"')

    Pipeline(monkeypatch, decide_portfolio=Boom())

    record = cycle(execute=True)

    assert record.outcome == "error"
    assert record.error == "decide_portfolio: Boom"
    for blob in (record.error, record.model_dump_json(), format_summary(record)):
        assert API_KEY not in blob


# --------------------------------------------------------------------------
# 6. aggregate outcomes
# --------------------------------------------------------------------------


def test_a_flat_account_with_nothing_requested_waits(monkeypatch):
    Pipeline(monkeypatch, observe_evidence=evidence_with(()), decide_portfolio=decision_with())

    record = cycle()

    assert record.outcome == "wait"
    assert record.actions == ()


def test_held_positions_with_nothing_requested_hold(monkeypatch):
    Pipeline(monkeypatch)

    record = cycle()

    assert record.outcome == "hold"
    assert record.actions == ()
    assert [v.action for v in record.decision.positions] == ["HOLD", "HOLD"]


def test_every_action_refused_is_rejected_and_nothing_is_submitted(monkeypatch):
    pipeline = Pipeline(
        monkeypatch,
        decide_portfolio=decision_with({A: "CLOSE", B: "HOLD"}, new_entry=entry("CALL")),
        decide_exit=refused("no_position"),
        decide_order=refused("market_closed"),
    )

    record = cycle(execute=True)

    assert record.outcome == "rejected"
    assert outcomes(record) == [("close", A, "rejected"), ("open", ENTRY, "rejected")]
    assert [a.risk.reason for a in record.actions] == ["no_position", "market_closed"]
    assert all(a.receipt is None for a in record.actions)
    assert pipeline.called("submit_paper_order") == []


# --------------------------------------------------------------------------
# 7. memory
# --------------------------------------------------------------------------


def test_a_memory_failure_never_stops_the_cycle(monkeypatch, tmp_path):
    journal = tmp_path / "cycles.jsonl"
    pipeline = Pipeline(monkeypatch, load_position_memory=NotImplementedError("lead implements this"))

    record = cycle(journal_path=journal)

    assert record.outcome == "hold"
    assert record.error is None
    assert pipeline.called("load_position_memory") == [((journal,), {})]
    ((_args, kwargs),) = pipeline.called("observe_evidence")
    assert kwargs == {"now": NOW, "memory": {}}


def test_journal_memory_reaches_the_evidence(monkeypatch):
    memory = {A: PositionMemo(symbol=A, entered_at=NOW - timedelta(hours=3), entry_thesis="why")}
    pipeline = Pipeline(monkeypatch, load_position_memory=memory)

    cycle()

    assert pipeline.called("observe_evidence")[0][1]["memory"] == memory


# --------------------------------------------------------------------------
# 8. journal
# --------------------------------------------------------------------------


def test_append_record_round_trips_records_with_actions(monkeypatch, tmp_path):
    Pipeline(monkeypatch, decide_portfolio=decision_with({A: "CLOSE", B: "HOLD"}, new_entry=entry("CALL")))
    journal = tmp_path / "logs" / "cycles.jsonl"
    first = cycle(execute=True)
    second = cycle()

    append_record(first, journal)
    append_record(second, journal)

    lines = journal.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert CycleRecord.model_validate_json(lines[0]) == first
    assert CycleRecord.model_validate_json(lines[1]) == second
    assert len(first.actions) == 2 and first.outcome == "submitted"
    assert "\n" not in lines[0]


# --------------------------------------------------------------------------
# 9. summary
# --------------------------------------------------------------------------


def test_the_summary_shows_portfolio_decision_and_every_action(monkeypatch):
    Pipeline(monkeypatch, decide_portfolio=decision_with({A: "HOLD", B: "CLOSE"}, new_entry=entry("CALL")))
    record = cycle(execute=True, forced_close=(A,))

    text = format_summary(record)

    assert text.startswith(f"RegimePilot cycle {CYCLE_ID}  execute  submitted  @ 2026-08-27 14:30:00Z")
    for expected in (
        f"close={A}",
        "2 position(s)",
        "98,000.75",
        f"{A} CLOSE - forced by --close",
        f"{B} CLOSE - test close",
        "new entry       CALL - Test entry.",
        "Test thesis.",
        f"close {A} submitted",
        f"close {B} submitted",
        f"open {ENTRY} submitted",
        "sell 1 x",
        "buy 1 x",
        "490.00",
        "500.00",
        f"regimepilot-{CYCLE_ID}-close1",
        f"regimepilot-{CYCLE_ID}-close2",
        f"regimepilot-{CYCLE_ID}-open",
        "accepted",
    ):
        assert expected in text, expected


def test_the_summary_omits_what_a_hold_never_gathered(monkeypatch):
    Pipeline(monkeypatch)

    text = format_summary(cycle())

    assert "dry_run  hold" in text
    assert "HOLD" in text and "new entry       none" in text
    for absent in ("action", "plan", "receipt", "error"):
        assert absent not in text, absent


def test_the_summary_of_an_early_failure_is_the_error_alone(monkeypatch):
    Pipeline(monkeypatch, observe_evidence=EvidenceError("failed to read account: Boom"))

    text = format_summary(cycle())

    assert "dry_run  error" in text
    assert "observe_evidence: failed to read account: Boom" in text
    for absent in ("portfolio", "decision", "action", "plan"):
        assert absent not in text, absent


# --------------------------------------------------------------------------
# 10. loop
# --------------------------------------------------------------------------


def test_the_loop_runs_max_cycles_and_sleeps_between_them(capsys):
    calls = []
    sleeps = []

    def run_once():
        calls.append(len(calls))
        if len(calls) == 2:
            raise RuntimeError("cycle two blew up")
        return None

    assert run_loop(run_once, interval_minutes=15, sleep=sleeps.append, max_cycles=3) == 0
    assert calls == [0, 1, 2]
    assert sleeps == [900, 900]
    assert "cycle failed: RuntimeError" in capsys.readouterr().err


def test_the_loop_stops_cleanly_on_ctrl_c(capsys):
    calls = []

    def interrupt(_seconds):
        raise KeyboardInterrupt

    assert run_loop(lambda: calls.append(1), sleep=interrupt) == 0
    assert calls == [1]
    assert "stopped" in capsys.readouterr().err


# --------------------------------------------------------------------------
# 11. CLI
# --------------------------------------------------------------------------


class Cli:
    """Fakes for everything ``main`` builds or runs, recording what it passed."""

    def __init__(self, monkeypatch, record):
        self.record = record
        self.cycles = []
        monkeypatch.setattr(runner, "load_settings", lambda: SETTINGS)
        monkeypatch.setattr(runner, "build_clients", lambda settings: (TRADING, DATA))
        monkeypatch.setattr(runner, "build_option_data_client", lambda settings: OPTION)
        monkeypatch.setattr(runner, "build_news_client", lambda settings: NEWS)
        monkeypatch.setattr(runner, "run_cycle", self._run_cycle)

    def _run_cycle(self, *args, **kwargs):
        self.cycles.append((args, kwargs))
        return self.record


def explode(*args, **kwargs):
    raise AssertionError("must not be reached")


@pytest.mark.parametrize(
    "argv",
    [["--action", "BUY_CALL"], ["--enter", "BUY_CALL"], ["--enter"], ["--close"], ["--interval-minutes", "0"]],
)
def test_main_rejects_a_bad_argument_before_loading_settings(monkeypatch, argv, capsys):
    monkeypatch.setattr(runner, "load_settings", explode)

    assert main(argv) == 1
    assert "usage" in capsys.readouterr().err


def test_main_reports_a_configuration_error(monkeypatch, capsys):
    def refuse():
        raise ConfigError("Missing credentials: ALPACA_API_KEY.")

    monkeypatch.setattr(runner, "load_settings", refuse)

    assert main([]) == 1
    assert "configuration error" in capsys.readouterr().err


def test_main_dry_run_prints_the_summary_and_journals_the_record(monkeypatch, tmp_path, capsys):
    Pipeline(monkeypatch)
    held = cycle()
    cli = Cli(monkeypatch, held)
    journal = tmp_path / "journal" / "cycles.jsonl"

    assert main(["--stub", "--journal", str(journal)]) == 0

    ((args, kwargs),) = cli.cycles
    assert args == (TRADING, DATA, OPTION, NEWS, SETTINGS)
    assert kwargs == {"execute": False, "stub": True, "forced_close": (), "forced_enter": None, "journal_path": journal}
    captured = capsys.readouterr()
    assert f"RegimePilot cycle {CYCLE_ID}  dry_run  hold" in captured.out
    assert "ARMED" not in captured.err
    lines = journal.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert CycleRecord.model_validate_json(lines[0]) == held


def test_main_execute_passes_the_forced_actions_and_arms_submission(monkeypatch, tmp_path, capsys):
    Pipeline(monkeypatch, decide_portfolio=decision_with({A: "CLOSE", B: "HOLD"}))
    cli = Cli(monkeypatch, cycle(execute=True))
    journal = tmp_path / "cycles.jsonl"

    assert main(["--close", "X", "--enter", "put", "--execute", "--json", "--journal", str(journal)]) == 0

    ((_args, kwargs),) = cli.cycles
    assert kwargs == {"execute": True, "stub": False, "forced_close": ("X",), "forced_enter": "PUT", "journal_path": journal}
    captured = capsys.readouterr()
    assert "ARMED" in captured.err
    assert json.loads(captured.out)["outcome"] == "submitted"
    assert len(journal.read_text(encoding="utf-8").splitlines()) == 1


def test_main_loop_uses_the_interval_and_journals_every_cycle(monkeypatch, tmp_path, capsys):
    Pipeline(monkeypatch)
    cli = Cli(monkeypatch, cycle())
    journal = tmp_path / "cycles.jsonl"
    sleeps = []

    def run_loop_with_two_cycles(run_once, *, interval_minutes):
        return run_loop(run_once, interval_minutes=interval_minutes, sleep=sleeps.append, max_cycles=2)

    monkeypatch.setattr(runner, "run_loop", run_loop_with_two_cycles)

    assert main(["--loop", "--interval-minutes", "5", "--json", "--journal", str(journal)]) == 0

    assert len(cli.cycles) == 2
    assert sleeps == [300]
    lines = journal.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    out_lines = capsys.readouterr().out.splitlines()
    assert len(out_lines) == 2 and json.loads(out_lines[0])["outcome"] == "hold"


def test_the_default_interval_is_fifteen_minutes():
    assert DEFAULT_INTERVAL_MINUTES == 15
