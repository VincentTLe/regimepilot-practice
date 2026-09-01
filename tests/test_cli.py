import json
from datetime import date, timedelta

import pytest
from typer.testing import CliRunner

import broker
import cli
from tests.fakes import (
    NOW,
    FakeOptionDataClient,
    FakeStockDataClient,
    FakeTradingClient,
    breakout_bars,
    fake_clock,
    fake_contract,
    fake_position,
    fake_snapshot,
    quiet_bars,
)

EXP = date(2026, 9, 11)  # 11 DTE from NOW
LONG_OCC = "SPY260911C00650000"
SHORT_OCC = "SPY260911C00655000"


@pytest.fixture(autouse=True)
def journal(tmp_path, monkeypatch):
    path = tmp_path / "cycles.jsonl"
    monkeypatch.setattr(cli, "JOURNAL_PATH", path)
    return path


def make_config(symbols="SPY"):
    return broker.load_config(
        {"ALPACA_API_KEY": "k", "ALPACA_SECRET_KEY": "s", "ALPACA_PAPER": "true",
         "SYMBOLS": symbols, "BAR_TIMEFRAME": "15m"}
    )


def entry_chain_snapshots():
    return {
        "SPY260911C00645000": fake_snapshot(6.0, 6.1, iv=0.20),
        "SPY260911C00650000": fake_snapshot(3.4, 3.5, iv=0.21),
        "SPY260911C00655000": fake_snapshot(1.5, 1.55, iv=0.25),
    }


def entry_contracts():
    return [
        fake_contract("SPY260911C00645000", 645.0, EXP),
        fake_contract("SPY260911C00650000", 650.0, EXP),
        fake_contract("SPY260911C00655000", 655.0, EXP),
    ]


def make_clients(**trading_kwargs):
    trading = FakeTradingClient(contracts=entry_contracts(), **trading_kwargs)
    stock = FakeStockDataClient(
        bars_by_symbol={"SPY": breakout_bars()},  # last bar body +5 vs ATR ~1 -> breakout_up
        quotes_by_symbol={"SPY": (649.9, 650.1)},
    )
    options = FakeOptionDataClient(entry_chain_snapshots())
    return trading, stock, options


def test_dry_run_cycle_plans_entry_but_submits_nothing(journal):
    trading, stock, options = make_clients()
    record = cli.run_cycle(make_config(), trading, stock, options, execute=False, use_stub=True)
    assert trading.submitted == []  # the core safety property of a dry run
    assert record["outcome"] == "planned"
    entry = record["entry"]
    assert entry["symbol"] == "SPY" and entry["direction"] == "CALL"
    # flattest skew wins: 645/650 (skew .01) over 650/655 (.04) and 645/655 (.05)
    assert entry["spread"]["long"] == "SPY260911C00645000"
    assert entry["spread"]["short"] == "SPY260911C00650000"
    assert entry["qty"] == 1
    assert entry["receipt"]["dry_run"] is True
    lines = journal.read_text().strip().splitlines()
    assert len(lines) == 1 and json.loads(lines[0])["cycle_id"] == record["cycle_id"]


def test_execute_cycle_submits_one_mleg_order():
    trading, stock, options = make_clients()
    record = cli.run_cycle(make_config(), trading, stock, options, execute=True, use_stub=True)
    assert record["outcome"] == "submitted"
    assert len(trading.submitted) == 1
    request = trading.submitted[0]
    assert [leg.symbol for leg in request.legs] == ["SPY260911C00645000", "SPY260911C00650000"]
    assert request.limit_price == pytest.approx(6.1 - 3.4)


def test_market_closed_does_nothing():
    trading, stock, options = make_clients(clock=fake_clock(is_open=False))
    record = cli.run_cycle(make_config(), trading, stock, options, execute=True, use_stub=True)
    assert record["outcome"] == "market_closed"
    assert trading.submitted == [] and "entry" not in record


def test_stale_quote_on_presubmit_recheck_aborts_entry():
    trading, stock, _ = make_clients()
    fresh = entry_chain_snapshots()
    stale = {
        symbol: fake_snapshot(snap.latest_quote.bid_price, snap.latest_quote.ask_price,
                              iv=snap.implied_volatility, stamp=NOW - timedelta(seconds=60))
        for symbol, snap in entry_chain_snapshots().items()
    }
    options = FakeOptionDataClient([fresh, stale])  # screen sees fresh, recheck sees stale
    record = cli.run_cycle(make_config(), trading, stock, options, execute=True, use_stub=True)
    assert trading.submitted == []
    assert record["entry"]["rejected"] == "recheck: stale_quote"


def test_stop_loss_exit_is_planned_and_underlying_blocked():
    positions = [
        fake_position(LONG_OCC, 1, 6.0, side="long"),
        fake_position(SHORT_OCC, 1, 4.0, side="short"),  # entry debit 2.00, stop at 1.00
    ]
    marks = {
        LONG_OCC: fake_snapshot(1.4, 1.6),  # mid 1.5
        SHORT_OCC: fake_snapshot(0.5, 0.7),  # mid 0.6 -> net mark 0.9 <= 1.00
    }
    trading = FakeTradingClient(positions=positions)
    stock = FakeStockDataClient(
        # an event fires, so SPY would otherwise be a candidate — being held must win
        bars_by_symbol={"SPY": breakout_bars()}, quotes_by_symbol={"SPY": (649.9, 650.1)}
    )
    record = cli.run_cycle(
        make_config(), trading, stock, FakeOptionDataClient(marks), execute=False, use_stub=True
    )
    assert record["exits"][0]["reason"] == "stop"
    assert record["exits"][0]["receipt"]["dry_run"] is True
    # a held underlying is never also an entry candidate
    spy = next(c for c in record["candidates"] if c["symbol"] == "SPY")
    assert spy["gate_block"] == "already_held"
    assert record["entry"] is None


def test_pending_order_on_leg_skips_exit():
    from types import SimpleNamespace

    positions = [
        fake_position(LONG_OCC, 1, 6.0, side="long"),
        fake_position(SHORT_OCC, 1, 4.0, side="short"),
    ]
    pending = SimpleNamespace(symbol=LONG_OCC, legs=None)
    marks = {LONG_OCC: fake_snapshot(1.4, 1.6), SHORT_OCC: fake_snapshot(0.5, 0.7)}
    trading = FakeTradingClient(positions=positions, orders=[pending])
    stock = FakeStockDataClient(
        bars_by_symbol={"SPY": quiet_bars()}, quotes_by_symbol={"SPY": (649.9, 650.1)}
    )
    record = cli.run_cycle(
        make_config(), trading, stock, FakeOptionDataClient(marks), execute=True, use_stub=True
    )
    assert record["exits"][0]["skipped"] == "pending_order"
    assert trading.submitted == []


def test_unpaired_leg_is_warned_and_untouched():
    trading = FakeTradingClient(positions=[fake_position(LONG_OCC, 1, 6.0)])
    stock = FakeStockDataClient(
        bars_by_symbol={"SPY": quiet_bars()}, quotes_by_symbol={"SPY": (649.9, 650.1)}
    )
    record = cli.run_cycle(
        make_config(), trading, stock, FakeOptionDataClient({}), execute=True, use_stub=True
    )
    assert any("unpaired" in w for w in record["warnings"])
    assert record["exits"] == [] and trading.submitted == []


def test_options_level_below_3_blocks_armed_entry():
    from tests.fakes import fake_account

    trading, stock, options = make_clients(account=fake_account(level=2))
    record = cli.run_cycle(make_config(), trading, stock, options, execute=True, use_stub=True)
    assert record["entry"]["rejected"] == "options_level_too_low"
    assert trading.submitted == []


# --- CLI smoke via typer ---

def test_account_command_smoke(monkeypatch):
    trading, _, _ = make_clients(positions=[
        fake_position(LONG_OCC, 1, 6.0, side="long"),
        fake_position(SHORT_OCC, 1, 4.0, side="short"),
    ])
    monkeypatch.setattr(cli, "_bootstrap", lambda: (make_config(), trading, None, None))
    result = CliRunner().invoke(cli.app, ["account"])
    assert result.exit_code == 0
    assert "equity: 100000.0" in result.output
    assert "SPY" in result.output


def test_candidates_command_smoke(monkeypatch):
    trading, stock, _ = make_clients()
    monkeypatch.setattr(cli, "_bootstrap", lambda: (make_config(), trading, stock, None))
    result = CliRunner().invoke(cli.app, ["candidates"])
    assert result.exit_code == 0
    assert "PASS" in result.output


def test_screen_command_rejects_bad_direction(monkeypatch):
    monkeypatch.setattr(cli, "_bootstrap", lambda: (make_config(), None, None, None))
    result = CliRunner().invoke(cli.app, ["screen", "SPY", "--direction", "SIDEWAYS"])
    assert result.exit_code != 0
