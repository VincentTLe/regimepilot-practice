"""Runner tests: one cycle end to end with every pipeline step replaced by a
fake, the JSONL journal, the loop and the CLI.

Offline by construction: every ``runner.<step>`` name is monkeypatched with a
recorder, the clients are sentinels, and no credential is ever read.
"""

import json
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from regimepilot import runner
from regimepilot.account import AccountError
from regimepilot.chain import ChainError
from regimepilot.config import ConfigError, load_settings
from regimepilot.decision import build_gate_hold_proposal
from regimepilot.evidence import EvidenceError
from regimepilot.execution import ExecutionError
from regimepilot.models import (
    AccountHint,
    AccountState,
    ChainPacket,
    CycleRecord,
    EvidencePacket,
    ExecutionState,
    FreshQuote,
    GatesEvidence,
    NewsEvidence,
    OrderPlan,
    OrderReceipt,
    RiskDecision,
    SelectedContract,
    SelectionResult,
    TradeProposal,
    UnderlyingEvidence,
)
from regimepilot.runner import (
    DEFAULT_INTERVAL_MINUTES,
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
SYMBOL = "SPY260903C00765000"
ORDER_ID = "aaaabbbb-cccc-dddd-eeee-ffff00001111"

API_KEY = "SUPER-SECRET-KEY"
SECRET_KEY = "SUPER-SECRET-SECRET"

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


def build_evidence(*, passed=True, hold_reason=None):
    return EvidencePacket(
        observed_at=NOW,
        symbol="SPY",
        gates=GatesEvidence(passed=passed, hold_reason=hold_reason, momentum_align="aligned_up"),
        underlying=UnderlyingEvidence(data_feed="iex", market_is_open=True, minutes_to_close=300.0),
        news=NewsEvidence(),
        account=AccountHint(),
    )


def proposal(action="BUY_CALL", model="stub"):
    return TradeProposal(
        observed_at=NOW,
        action=action,
        confidence="medium",
        thesis="Test proposal.",
        evidence_used=("gates.momentum_align",),
        model=model,
    )


def chain_packet(action="BUY_CALL"):
    return ChainPacket(
        observed_at=NOW, action=action, option_feed="indicative", underlying_mid=765.0, quotes_read_at=NOW
    )


def selected_contract():
    return SelectedContract(
        symbol=SYMBOL,
        option_type="call",
        strike_price=765.0,
        expiration_date=TODAY + timedelta(days=7),
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


def execution_state(*, ask=5.0, reject_reason=None):
    account = AccountState(observed_at=NOW, account_id_masked="****7888", options_buying_power=98000.75)
    quote = FreshQuote(
        symbol=SYMBOL,
        bid=4.9,
        ask=ask,
        quote_at=NOW - timedelta(seconds=1),
        server_time=NOW,
        reject_reason=reject_reason,
    )
    return ExecutionState(observed_at=NOW, account=account, market_is_open=True, minutes_to_close=120.0, quote=quote)


def order_plan():
    return OrderPlan(
        symbol=SYMBOL, qty=1, limit_price=5.0, max_premium_usd=500.0, client_order_id=f"regimepilot-{CYCLE_ID}"
    )


def risk_decision(*, approved=True):
    if approved:
        return RiskDecision(approved=True, plan=order_plan())
    return RiskDecision(approved=False, reason="premium_over_cap")


def receipt(*, submitted=True):
    if submitted:
        return OrderReceipt(
            submitted=True,
            order_id=ORDER_ID,
            client_order_id=f"regimepilot-{CYCLE_ID}",
            status="accepted",
            submitted_at=NOW,
            filled_qty=0.0,
        )
    return OrderReceipt(submitted=False, error="submit_order: APIError")


class Pipeline:
    """Every pipeline step as a recorder returning a canned answer, or raising one.

    A step whose answer is an exception raises it. ``calls`` keeps the order
    and arguments of every call so a test can assert what ran and what did not.
    """

    STEPS = (
        "observe_evidence",
        "propose_trade",
        "observe_chain",
        "select_contract",
        "observe_execution_state",
        "decide_order",
        "submit_paper_order",
    )

    def __init__(self, monkeypatch, **answers):
        defaults = {
            "observe_evidence": build_evidence(),
            "propose_trade": proposal(),
            "observe_chain": chain_packet(),
            "select_contract": selection(),
            "observe_execution_state": execution_state(),
            "decide_order": risk_decision(),
            "submit_paper_order": receipt(),
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
            return answer

        return fake

    def called(self, step):
        return [(args, kwargs) for name, args, kwargs in self.calls if name == step]

    def ran(self):
        return [name for name, _args, _kwargs in self.calls]


def cycle(*, execute=False, forced_action=None, stub=True):
    return run_cycle(
        TRADING,
        DATA,
        OPTION,
        NEWS,
        SETTINGS,
        execute=execute,
        stub=stub,
        forced_action=forced_action,
        now=NOW,
        cycle_id=CYCLE_ID,
    )


# --------------------------------------------------------------------------
# 1. cycle id
# --------------------------------------------------------------------------


def test_cycle_id_is_the_utc_start_time():
    new_york = datetime(2026, 8, 27, 10, 30, tzinfo=ZoneInfo("America/New_York"))
    assert new_cycle_id(new_york) == CYCLE_ID
    assert new_cycle_id(NOW) == CYCLE_ID


# --------------------------------------------------------------------------
# 2. outcome routing, one terminal outcome per test
# --------------------------------------------------------------------------


def test_a_gate_hold_ends_the_cycle_before_any_chain_read(monkeypatch):
    evidence = build_evidence(passed=False, hold_reason="market_closed")
    pipeline = Pipeline(
        monkeypatch, observe_evidence=evidence, propose_trade=build_gate_hold_proposal(evidence)
    )

    record = cycle()

    assert isinstance(record, CycleRecord)
    assert record.outcome == "hold"
    assert record.cycle_id == CYCLE_ID
    assert record.mode == "dry_run"
    assert record.started_at == NOW
    assert record.proposal is not None and record.proposal.gate_skipped is True
    assert record.selection is None and record.risk is None and record.receipt is None
    assert record.error is None
    assert pipeline.ran() == ["observe_evidence", "propose_trade"]


def test_an_llm_hold_ends_the_cycle_before_any_chain_read(monkeypatch):
    pipeline = Pipeline(monkeypatch, propose_trade=proposal("HOLD"))

    record = cycle()

    assert record.outcome == "hold"
    assert record.proposal.action == "HOLD"
    assert pipeline.called("observe_chain") == []


def test_no_contract_ends_the_cycle_before_the_fresh_reread(monkeypatch):
    pipeline = Pipeline(monkeypatch, select_contract=selection(selected=False))

    record = cycle()

    assert record.outcome == "no_contract"
    assert record.selection.reason == "no_candidates"
    assert record.execution_state is None and record.risk is None
    assert pipeline.called("observe_execution_state") == []


def test_a_risk_refusal_is_rejected_and_never_submitted(monkeypatch):
    pipeline = Pipeline(monkeypatch, decide_order=risk_decision(approved=False))

    record = cycle(execute=True)

    assert record.outcome == "rejected"
    assert record.risk.reason == "premium_over_cap"
    assert record.execution_state is not None
    assert record.receipt is None
    assert pipeline.called("submit_paper_order") == []


def test_an_approved_dry_run_is_planned_and_never_submitted(monkeypatch):
    pipeline = Pipeline(monkeypatch)

    record = cycle(execute=False)

    assert record.outcome == "planned"
    assert record.mode == "dry_run"
    assert record.risk.approved is True
    assert record.risk.plan == order_plan()
    assert record.receipt is None
    assert pipeline.called("submit_paper_order") == []


def test_an_approved_execute_run_submits_the_plan_exactly_once(monkeypatch):
    pipeline = Pipeline(monkeypatch)

    record = cycle(execute=True)

    assert record.outcome == "submitted"
    assert record.mode == "execute"
    assert record.receipt == receipt()
    assert pipeline.called("submit_paper_order") == [((TRADING, order_plan()), {})]

    # Evidence and chain share the cycle's start; the pre-order re-read gets
    # a fresh clock, so ``now`` must not be forwarded to it.
    (evidence_args, evidence_kwargs), = pipeline.called("observe_evidence")
    assert evidence_args == (TRADING, DATA, NEWS)
    assert evidence_kwargs == {"now": NOW}
    (chain_args, chain_kwargs), = pipeline.called("observe_chain")
    assert chain_args == (TRADING, DATA, OPTION)
    assert chain_kwargs == {"action": "BUY_CALL", "now": NOW}
    (state_args, state_kwargs), = pipeline.called("observe_execution_state")
    assert state_args == (TRADING, OPTION)
    assert state_kwargs == {"selected": selected_contract()}
    (risk_args, risk_kwargs), = pipeline.called("decide_order")
    assert risk_args == (selection(), execution_state())
    assert risk_kwargs == {"cycle_id": CYCLE_ID}


def test_a_refused_submission_is_an_error_carrying_the_receipt(monkeypatch):
    Pipeline(monkeypatch, submit_paper_order=receipt(submitted=False))

    record = cycle(execute=True)

    assert record.outcome == "error"
    assert record.error == "submit_order: APIError"
    assert record.receipt == receipt(submitted=False)
    assert record.risk.approved is True


def test_the_proposal_step_forwards_stub_settings_and_transport(monkeypatch):
    pipeline = Pipeline(monkeypatch, propose_trade=proposal("HOLD"))
    transport = object()

    run_cycle(TRADING, DATA, OPTION, NEWS, SETTINGS, execute=False, stub=True, now=NOW, transport=transport)

    (args, kwargs), = pipeline.called("propose_trade")
    assert args == (build_evidence(),)
    assert kwargs == {"stub": True, "settings": SETTINGS, "transport": transport}


# --------------------------------------------------------------------------
# 3. --action replaces only the LLM call
# --------------------------------------------------------------------------


def test_a_forced_action_never_bypasses_a_failed_gate(monkeypatch):
    pipeline = Pipeline(monkeypatch, observe_evidence=build_evidence(passed=False, hold_reason="market_closed"))

    record = cycle(forced_action="BUY_CALL")

    assert record.outcome == "hold"
    assert record.forced_action == "BUY_CALL"
    assert record.proposal.action == "HOLD"
    assert record.proposal.model == "pre_gate"
    assert record.proposal.gate_skipped is True
    assert pipeline.ran() == ["observe_evidence"]


def test_a_forced_action_replaces_the_llm_after_the_gates_passed(monkeypatch):
    pipeline = Pipeline(monkeypatch, observe_chain=chain_packet("BUY_PUT"), select_contract=selection(action="BUY_PUT"))

    record = cycle(forced_action="BUY_PUT")

    assert record.outcome == "planned"
    assert record.forced_action == "BUY_PUT"
    assert record.proposal.action == "BUY_PUT"
    assert record.proposal.model == "forced"
    assert record.proposal.gate_skipped is False
    assert pipeline.called("propose_trade") == []
    (_args, kwargs), = pipeline.called("observe_chain")
    assert kwargs["action"] == "BUY_PUT"
    assert pipeline.ran() == [
        "observe_evidence",
        "observe_chain",
        "select_contract",
        "observe_execution_state",
        "decide_order",
    ]


# --------------------------------------------------------------------------
# 4. failures become outcome "error" with everything gathered so far
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("step", "error", "attached"),
    [
        ("observe_evidence", EvidenceError("failed to read account: Boom"), ()),
        ("observe_chain", ChainError("failed to read contracts: Boom"), ("proposal",)),
        ("observe_execution_state", ExecutionError("failed to read clock: Boom"), ("proposal", "selection")),
        ("observe_evidence", AccountError("failed to read positions: Boom"), ()),
        ("propose_trade", ConfigError("Missing OPENROUTER_API_KEY."), ()),
    ],
)
def test_a_step_failure_is_recorded_and_stops_the_cycle(monkeypatch, step, error, attached):
    pipeline = Pipeline(monkeypatch, **{step: error})

    record = cycle(execute=True)

    assert record.outcome == "error"
    assert record.error == f"{step}: {error}"
    assert pipeline.ran()[-1] == step
    for field in ("proposal", "selection", "execution_state", "risk"):
        assert (getattr(record, field) is not None) == (field in attached), field
    assert record.receipt is None
    assert pipeline.called("submit_paper_order") == []


def test_an_unexpected_exception_is_named_by_type_only(monkeypatch):
    class Boom(RuntimeError):
        def __init__(self):
            super().__init__(f'401 unauthorized for API_KEY="{API_KEY}" SECRET_KEY="{SECRET_KEY}"')

    Pipeline(monkeypatch, observe_chain=Boom())

    record = cycle(execute=True)

    assert record.outcome == "error"
    assert record.error == "observe_chain: Boom"
    for blob in (record.error, record.model_dump_json(), format_summary(record)):
        assert API_KEY not in blob
        assert SECRET_KEY not in blob


# --------------------------------------------------------------------------
# 5. journal
# --------------------------------------------------------------------------


def test_append_record_writes_one_line_per_cycle_and_creates_the_directory(monkeypatch, tmp_path):
    Pipeline(monkeypatch)
    journal = tmp_path / "logs" / "cycles.jsonl"
    first = cycle()
    second = cycle(execute=True)

    append_record(first, journal)
    append_record(second, journal)

    lines = journal.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert CycleRecord.model_validate_json(lines[0]) == first
    assert CycleRecord.model_validate_json(lines[1]) == second
    assert "\n" not in lines[0]


# --------------------------------------------------------------------------
# 6. summary
# --------------------------------------------------------------------------


def test_the_summary_shows_every_section_of_a_submitted_cycle(monkeypatch):
    Pipeline(monkeypatch)
    record = cycle(execute=True, forced_action="BUY_CALL")

    text = format_summary(record)

    assert text.startswith(f"RegimePilot cycle {CYCLE_ID}  execute  submitted  @ 2026-08-27 14:30:00Z")
    for expected in ("BUY_CALL", "forced", SYMBOL, "765.00", "98,000.75", "120.0", "approved", "500.00",
                     f"regimepilot-{CYCLE_ID}", ORDER_ID, "accepted"):
        assert expected in text, expected


def test_the_summary_omits_what_a_hold_never_gathered(monkeypatch):
    Pipeline(monkeypatch, propose_trade=proposal("HOLD"))

    text = format_summary(cycle())

    assert "dry_run  hold" in text
    assert "HOLD" in text
    for absent in ("selected", "risk", "plan", "receipt", "error"):
        assert absent not in text, absent


# --------------------------------------------------------------------------
# 7. loop
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
# 8. CLI
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


@pytest.mark.parametrize("argv", [["--action", "SELL_CALL"], ["--action", "HOLD"], ["--interval-minutes", "0"]])
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
    planned = cycle(forced_action="BUY_CALL")
    cli = Cli(monkeypatch, planned)
    journal = tmp_path / "journal" / "cycles.jsonl"

    assert main(["--stub", "--action", "buy_call", "--journal", str(journal)]) == 0

    (args, kwargs), = cli.cycles
    assert args == (TRADING, DATA, OPTION, NEWS, SETTINGS)
    assert kwargs == {"execute": False, "stub": True, "forced_action": "BUY_CALL"}
    captured = capsys.readouterr()
    assert f"RegimePilot cycle {CYCLE_ID}  dry_run  planned" in captured.out
    assert "ARMED" not in captured.err
    lines = journal.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert CycleRecord.model_validate_json(lines[0]) == planned


def test_main_execute_arms_submission_and_says_so(monkeypatch, tmp_path, capsys):
    Pipeline(monkeypatch)
    cli = Cli(monkeypatch, cycle(execute=True))

    assert main(["--execute", "--json", "--journal", str(tmp_path / "cycles.jsonl")]) == 0

    (_args, kwargs), = cli.cycles
    assert kwargs["execute"] is True
    captured = capsys.readouterr()
    assert "ARMED" in captured.err
    assert json.loads(captured.out)["outcome"] == "submitted"


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
    assert len(out_lines) == 2 and json.loads(out_lines[0])["outcome"] == "planned"


def test_the_default_interval_is_fifteen_minutes():
    assert DEFAULT_INTERVAL_MINUTES == 15
