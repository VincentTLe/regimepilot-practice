"""Scorecard tests. Hand-computed expected values, no network."""

import socket
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from regimepilot.backtest import SimulatedTrade
from regimepilot.models import OhlcvBar
from regimepilot.score import (
    BacktestScorecard,
    compute_scorecard,
    compute_scorecards_by_regime,
    format_summary,
)

NY = ZoneInfo("America/New_York")


def trade(entry_price, exit_price, *, day=date(2026, 8, 24), pnl=None):
    entry_time = datetime(day.year, day.month, day.day, 10, 0, tzinfo=NY)
    exit_time = datetime(day.year, day.month, day.day, 15, 30, tzinfo=NY)
    computed_pnl = (exit_price - entry_price) * 100 if pnl is None else pnl
    return SimulatedTrade(
        entry_time=entry_time,
        exit_time=exit_time,
        option_type="call",
        strike=450.0,
        expiration_date=day + timedelta(days=7),
        entry_spot=450.0,
        exit_spot=452.0,
        entry_price=entry_price,
        exit_price=exit_price,
        qty=1,
        pnl_usd=computed_pnl,
        regime_label="trending_up",
        confidence="medium",
        thesis="test fixture",
    )


def test_compute_scorecard_empty_input():
    card = compute_scorecard([])
    assert card.trade_count == 0
    assert card.win_rate is None
    assert card.total_pnl_usd == 0.0


def test_compute_scorecard_all_wins():
    trades = [trade(1.0, 2.0), trade(1.0, 1.5)]
    card = compute_scorecard(trades)
    assert card.trade_count == 2
    assert card.win_rate == pytest.approx(1.0)
    assert card.total_pnl_usd == pytest.approx(150.0)  # (100 + 50)
    assert card.profit_factor is None  # no losses to divide by


def test_compute_scorecard_mixed_wins_and_losses():
    trades = [trade(1.0, 2.0), trade(1.0, 0.5)]  # +100, -50
    card = compute_scorecard(trades)
    assert card.win_rate == pytest.approx(0.5)
    assert card.total_pnl_usd == pytest.approx(50.0)
    assert card.average_pnl_usd == pytest.approx(25.0)
    assert card.profit_factor == pytest.approx(100.0 / 50.0)


def test_compute_scorecard_max_drawdown():
    # PnL sequence in time order: +100, -150, +20 -> equity 100, -50, -30
    # Peak is 100 at step 1; trough after is -50 -> drawdown of 150.
    trades = [
        trade(1.0, 2.0, day=date(2026, 8, 24), pnl=100.0),
        trade(1.0, 2.0, day=date(2026, 8, 25), pnl=-150.0),
        trade(1.0, 2.0, day=date(2026, 8, 26), pnl=20.0),
    ]
    card = compute_scorecard(trades)
    assert card.max_drawdown_usd == pytest.approx(150.0)


def test_compute_scorecard_sharpe_none_with_single_trade():
    card = compute_scorecard([trade(1.0, 2.0)])
    assert card.sharpe_per_trade is None


def test_compute_scorecard_sharpe_computed_with_multiple_trades():
    trades = [
        trade(1.0, 2.0, day=date(2026, 8, 24)),
        trade(1.0, 1.2, day=date(2026, 8, 25)),
        trade(1.0, 0.8, day=date(2026, 8, 26)),
    ]
    card = compute_scorecard(trades)
    assert card.sharpe_per_trade is not None


def test_compute_scorecard_baseline_buy_and_hold():
    daily_bars = [
        OhlcvBar(timestamp=datetime(2026, 8, 24, tzinfo=NY), close=450.0),
        OhlcvBar(timestamp=datetime(2026, 8, 25, tzinfo=NY), close=459.0),
    ]
    trades = [trade(1.0, 2.0, day=date(2026, 8, 24)), trade(1.0, 2.0, day=date(2026, 8, 25))]
    card = compute_scorecard(trades, daily_bars)
    assert card.baseline_buy_and_hold_return_pct == pytest.approx(2.0, abs=1e-6)


def test_compute_scorecards_by_regime_splits_correctly():
    trades = [
        trade(1.0, 2.0, day=date(2026, 8, 24)),  # regime_label defaults to trending_up in the helper
    ]
    trending_trade = trades[0]
    chop_trade = trade(1.0, 0.5, day=date(2026, 8, 25))
    chop_trade = chop_trade.model_copy(update={"regime_label": "high_vol_chop"})

    grouped = compute_scorecards_by_regime([trending_trade, chop_trade])
    assert set(grouped) == {"trending_up", "high_vol_chop"}
    assert grouped["trending_up"].trade_count == 1
    assert grouped["high_vol_chop"].trade_count == 1
    assert grouped["trending_up"].total_pnl_usd == pytest.approx(100.0)
    assert grouped["high_vol_chop"].total_pnl_usd == pytest.approx(-50.0)


def test_compute_scorecards_by_regime_empty_input():
    assert compute_scorecards_by_regime([]) == {}


def test_compute_scorecards_by_regime_never_synthesizes_absent_labels():
    trades = [trade(1.0, 2.0)]  # only trending_up present
    grouped = compute_scorecards_by_regime(trades)
    assert "low_vol_drift" not in grouped
    assert "unknown" not in grouped


def test_format_summary_handles_all_none_fields():
    card = BacktestScorecard(trade_count=0)
    summary = format_summary(card)
    assert "trades=0" in summary
    assert "null" in summary


def test_score_never_touches_the_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("score must not open a socket")

    monkeypatch.setattr(socket, "socket", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)

    compute_scorecard([trade(1.0, 2.0)])
