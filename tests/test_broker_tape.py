"""broker: recent trades + L1 sizes for the tape sensor (fakes only, no network)."""

from datetime import datetime, timezone

import pytest

import broker
from tests.fakes import FakeStockDataClient

NOW = datetime(2026, 9, 3, 14, 0, tzinfo=timezone.utc)


def test_fetch_recent_trades_returns_price_size_pairs_per_symbol():
    stock = FakeStockDataClient(trades_by_symbol={"SPY": [(100.0, 5, NOW), (100.1, 3, NOW), (100.1, 0, NOW)]})
    trades = broker.fetch_recent_trades(stock, ("SPY", "QQQ"), 15, NOW)
    assert trades["SPY"] == [(100.0, 5.0), (100.1, 3.0)]  # zero-size prints dropped
    assert trades["QQQ"] == []


def test_fetch_recent_trades_wraps_errors_to_type_names():
    stock = FakeStockDataClient(trades_error=RuntimeError("secret detail"))
    with pytest.raises(broker.BrokerError) as excinfo:
        broker.fetch_recent_trades(stock, ("SPY",), 15, NOW)
    assert "RuntimeError" in str(excinfo.value) and "secret" not in str(excinfo.value)


def test_fetch_spot_quotes_carries_sizes_and_mids_wrapper_still_works():
    stock = FakeStockDataClient(quotes_by_symbol={"SPY": (99.0, 101.0, 300, 100), "QQQ": (50.0, 50.2)})
    quotes = broker.fetch_spot_quotes(stock, ("SPY", "QQQ", "IWM"))
    assert quotes["SPY"].mid == 100.0 and quotes["SPY"].bid_size == 300 and quotes["SPY"].ask_size == 100
    assert quotes["QQQ"].mid == pytest.approx(50.1)
    assert quotes["IWM"].mid is None and quotes["IWM"].bid_size is None
    assert broker.fetch_spot_mids(stock, ("SPY", "IWM")) == {"SPY": 100.0, "IWM": None}


def test_fetch_recent_trades_refuses_a_truncated_window(monkeypatch):
    # alpaca-py counts `limit` across ALL symbols: hitting it means the newest prints
    # of some symbol are missing, so the whole read is reported as unusable.
    monkeypatch.setattr(broker, "TRADES_LIMIT", 3)
    stock = FakeStockDataClient(trades_by_symbol={"SPY": [(100.0, 1, NOW)] * 2, "QQQ": [(50.0, 1, NOW)]})
    with pytest.raises(broker.BrokerError) as excinfo:
        broker.fetch_recent_trades(stock, ("SPY", "QQQ"), 15, NOW)
    assert "truncated" in str(excinfo.value)
