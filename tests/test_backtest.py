"""Backtest engine tests. No network, no vendor SDK, no LLM.

Fixtures build a full trading day of 1-minute bars so gates (which need a
60-minute return) actually pass partway through the session, mirroring how
the live pipeline behaves.
"""

import socket
import tempfile
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from regimepilot.backtest import (
    BacktestError,
    daily_bars_from_minute_bars,
    load_minute_bars_csv,
    run_backtest,
)
from regimepilot.decision import stub_proposal
from regimepilot.models import EvidencePacket, OhlcvBar, TradeProposal

NY = ZoneInfo("America/New_York")


def _session_minute_bars(day: date, prices: list[float]) -> list[OhlcvBar]:
    """One bar per minute starting 09:30 ET, with the given closes in order."""
    start = datetime(day.year, day.month, day.day, 9, 30, tzinfo=NY)
    bars = []
    for i, price in enumerate(prices):
        ts = start + timedelta(minutes=i)
        bars.append(OhlcvBar(timestamp=ts, open=price, high=price, low=price, close=price, volume=100.0))
    return bars


def _trending_up_session(day: date, minutes: int = 200, start_price: float = 450.0) -> list[OhlcvBar]:
    prices = [start_price + 0.02 * i for i in range(minutes)]
    return _session_minute_bars(day, prices)


def _flat_session(day: date, minutes: int = 200, price: float = 450.0) -> list[OhlcvBar]:
    return _session_minute_bars(day, [price] * minutes)


def test_load_minute_bars_csv_round_trip():
    day = date(2026, 8, 24)
    bars = _trending_up_session(day, minutes=5)
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = Path(tmp) / "bars.csv"
        csv_path.write_text(
            "timestamp,open,high,low,close,volume\n"
            + "\n".join(
                f"{b.timestamp.isoformat()},{b.open},{b.high},{b.low},{b.close},{b.volume}" for b in bars
            )
        )
        loaded = load_minute_bars_csv(str(csv_path))
    assert len(loaded) == 5
    assert loaded[0].close == pytest.approx(450.0)
    assert loaded[-1].close == pytest.approx(450.08)


def test_load_minute_bars_csv_raises_on_malformed_row():
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = Path(tmp) / "bars.csv"
        csv_path.write_text("timestamp,open,high,low,close,volume\nnot-a-timestamp,1,1,1,1,1\n")
        with pytest.raises(BacktestError):
            load_minute_bars_csv(str(csv_path))


def test_daily_bars_from_minute_bars_aggregates_one_session():
    day = date(2026, 8, 24)
    bars = _session_minute_bars(day, [100.0, 105.0, 95.0, 102.0])
    daily = daily_bars_from_minute_bars(bars)
    assert len(daily) == 1
    assert daily[0].open == pytest.approx(100.0)
    assert daily[0].close == pytest.approx(102.0)
    assert daily[0].high == pytest.approx(105.0)
    assert daily[0].low == pytest.approx(95.0)
    assert daily[0].timestamp.hour == 0


def test_daily_bars_skip_sessions_with_no_priced_bars():
    day = date(2026, 8, 24)
    empty_bar = OhlcvBar(timestamp=datetime(day.year, day.month, day.day, 9, 30, tzinfo=NY))
    assert daily_bars_from_minute_bars([empty_bar]) == []


def test_run_backtest_produces_no_trades_on_a_flat_session():
    # Momentum never aligns on a flat tape, so the stub proposal is always
    # HOLD and no simulated position should ever open.
    day = date(2026, 8, 24)  # a Monday
    bars = _flat_session(day)
    trades = run_backtest(bars)
    assert trades == []


def test_run_backtest_opens_and_closes_a_call_on_a_clean_uptrend():
    day = date(2026, 8, 24)
    bars = _trending_up_session(day)
    trades = run_backtest(bars)
    assert len(trades) == 1
    trade = trades[0]
    assert trade.option_type == "call"
    assert trade.entry_time.date() == day
    assert trade.exit_time.date() == day
    assert trade.exit_spot >= trade.entry_spot  # uptrend session
    assert trade.regime_label in ("trending_up", "unknown", "low_vol_drift", "high_vol_chop")


def test_run_backtest_opens_at_most_one_position_per_session():
    day = date(2026, 8, 24)
    bars = _trending_up_session(day)
    trades = run_backtest(bars)
    assert len(trades) <= 1


def test_run_backtest_never_forces_a_trade_when_decision_fn_always_holds():
    def always_hold(evidence: EvidencePacket) -> TradeProposal:
        return TradeProposal(
            observed_at=evidence.observed_at,
            symbol=evidence.symbol,
            action="HOLD",
            confidence="low",
            thesis="test override",
        )

    day = date(2026, 8, 24)
    bars = _trending_up_session(day)
    trades = run_backtest(bars, decision_fn=always_hold)
    assert trades == []


def test_run_backtest_uses_the_default_stub_when_unspecified():
    # Sanity check that the default decision_fn really is stub_proposal, so a
    # backtest run needs no OpenRouter key by default.
    day = date(2026, 8, 24)
    bars = _trending_up_session(day)
    trades_default = run_backtest(bars)
    trades_explicit = run_backtest(bars, decision_fn=stub_proposal)
    assert len(trades_default) == len(trades_explicit)


def test_run_backtest_across_multiple_sessions():
    day1 = date(2026, 8, 24)
    day2 = date(2026, 8, 25)
    bars = _trending_up_session(day1) + _trending_up_session(day2, start_price=460.0)
    trades = run_backtest(bars)
    # Each session is independent: at most one trade per session.
    assert len(trades) <= 2
    for trade in trades:
        assert trade.entry_time.date() == trade.exit_time.date()


def test_backtest_never_touches_the_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("backtest must not open a socket")

    monkeypatch.setattr(socket, "socket", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)

    day = date(2026, 8, 24)
    run_backtest(_trending_up_session(day))
