"""backtest_tape.py: the pure replay helpers on synthetic bars and prints."""

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

import backtest_tape
import settings
import signals

OPEN = datetime(2026, 9, 1, 13, 30, tzinfo=timezone.utc)  # 09:30 ET
BAR = 300


@pytest.fixture(autouse=True)
def thresholds(monkeypatch):
    monkeypatch.setattr(settings, "FLOW_MIN_IMBALANCE", 0.15)
    monkeypatch.setattr(settings, "FLOW_MIN_TRADES", 5)
    monkeypatch.setattr(settings, "FLOW_EXIT_BARS", 2)
    monkeypatch.setattr(settings, "MACD_MIN_HIST_ATR", 0.0)
    monkeypatch.setattr(settings, "RSI_OVERBOUGHT", 101.0)
    monkeypatch.setattr(settings, "RSI_OVERSOLD", -1.0)


def frame(closes, *, start=OPEN - timedelta(seconds=BAR * 60)):
    """Bars with 1-point ranges (ATR ~1) whose closes follow `closes`; opens = previous close."""
    rows, prev = [], closes[0]
    for n, close in enumerate(closes):
        rows.append({"timestamp": start + timedelta(seconds=BAR * n), "open": prev,
                     "high": max(prev, close) + 0.5, "low": min(prev, close) - 0.5, "close": close, "volume": 1000.0})
        prev = close
    return signals.add_indicators(pd.DataFrame(rows).set_index("timestamp"))


def prints_for(df, direction: str):
    """Ten prints per bar, all upticks (buying) or downticks (selling)."""
    out = []
    step = 0.01 if direction == "buy" else -0.01
    for ts, row in df.iterrows():
        base = float(row["open"])
        for k in range(10):
            out.append((ts.timestamp() + 5 + k * 20, base + step * (k + 1), 10.0))
    return sorted(out)


def test_flow_at_uses_only_the_trailing_window():
    prints = [(t, 100.0 + 0.01 * t, 1.0) for t in range(0, 130, 10)]  # upticks every 10 s
    stats = backtest_tape.flow_at(prints, end_ts=60, minutes=1, min_trades=1)
    assert stats.trades == 6 and stats.buy_volume == 5.0 and stats.sell_volume == 0.0  # (0, 60]


def test_breakout_with_buying_tape_is_agree_and_measures_the_follow_through():
    closes = [100.0 + 0.05 * (n % 3) for n in range(60)]  # 60 quiet bars
    closes += [105.0] + [105.0 + 0.5 * k for k in range(1, 14)]  # breakout bar then drift up
    df = frame(closes)
    session_close = df.index[-1] + timedelta(seconds=BAR)
    rows = backtest_tape.evaluate_day(df, prints_for(df, "buy"), OPEN, session_close, bar_seconds=BAR)
    assert len(rows) == 1
    row = rows[0]
    assert row["direction"] == "CALL" and "breakout_up" in row["events"]
    assert row["tape"] == "agree" and row["flow"] > 0.5  # synthetic prints: a few downticks at bar seams
    assert row["atr_move_6"] > 0 and row["pct_move_12"] > row["pct_move_6"] > 0
    assert row["exit_event_only_bars"] == 12 and row["exit_tape_rule_bars"] == 12  # no reversal: held to horizon


def test_breakout_with_selling_tape_is_disagree():
    closes = [100.0 + 0.05 * (n % 3) for n in range(60)] + [105.0] + [104.0] * 13
    df = frame(closes)
    rows = backtest_tape.evaluate_day(df, prints_for(df, "sell"), OPEN, df.index[-1] + timedelta(seconds=BAR),
                                      bar_seconds=BAR)
    assert rows and rows[0]["tape"] == "disagree"


def test_reversal_simulation_event_only_exits_first_and_tape_rule_needs_conviction():
    # breakout up at bar 60, an opposing breakout down at bar 63, then flat
    closes = [100.0 + 0.05 * (n % 3) for n in range(60)] + [105.0, 105.2, 105.1, 101.0] + [101.0] * 10
    df = frame(closes)
    session_close = df.index[-1] + timedelta(seconds=BAR)
    # buying prints everywhere: the tape never confirms the reversal -> tape rule holds to the horizon
    rows = backtest_tape.evaluate_day(df, prints_for(df, "buy"), OPEN, session_close, bar_seconds=BAR)
    row = rows[0]
    assert row["exit_event_only_bars"] == 3 and row["exit_tape_rule_bars"] == 12
    # selling prints everywhere: the tape confirms on the event bar itself
    rows = backtest_tape.evaluate_day(df, prints_for(df, "sell"), OPEN, session_close, bar_seconds=BAR)
    assert rows[0]["exit_tape_rule_bars"] == 3


def test_walk_exit_rules():
    path = [0.3, 0.8, 1.2, 1.6, 0.9, 0.4, -0.2, -1.1, -1.4]
    assert backtest_tape.walk_exit(path, stop=1.0, trail=None, tp=None) == (-1.1, 8)   # stop on the first close <= -1
    assert backtest_tape.walk_exit(path, stop=None, trail=1.0, tp=None) == (0.4, 6)    # peak 1.6, gives back 1.0 at 0.4
    assert backtest_tape.walk_exit(path, stop=None, trail=None, tp=1.5) == (1.6, 4)    # take profit
    assert backtest_tape.walk_exit(path, stop=None, trail=None, tp=None) == (-1.4, 9)  # hold to close
    assert backtest_tape.walk_exit([], stop=1.0, trail=1.0, tp=None) == (0.0, 0)
