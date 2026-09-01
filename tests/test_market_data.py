from datetime import timedelta

import pytest

import market_data
from tests.fakes import NOW, FakeStockDataClient, fake_bar

SECRET = "SUPER-SECRET-VALUE"


# --- timeframe parsing (moved here from broker) ---

@pytest.mark.parametrize(
    ("raw", "expected"),
    [("15m", (15, "m", 900)), ("1h", (1, "h", 3600)), ("1d", (1, "d", 86400)), ("1w", (1, "w", 604800))],
)
def test_parse_timeframe(raw, expected):
    assert market_data.parse_timeframe(raw) == expected


@pytest.mark.parametrize("bad", ["", "m", "15x", "0m", "1.5h", "fifteen"])
def test_parse_timeframe_rejects_garbage(bad):
    with pytest.raises(market_data.MarketDataError):
        market_data.parse_timeframe(bad)


# --- fetch_ohlcv ---

def bars_ending_now(count=5):
    return [
        fake_bar(NOW - timedelta(seconds=900 * (count - i)), 100.0 + i, 101.0 + i, 99.0 + i, 100.5 + i)
        for i in range(count)
    ]


def test_fetch_ohlcv_returns_sorted_ohlcv_frame():
    client = FakeStockDataClient(bars_by_symbol={"SPY": list(reversed(bars_ending_now()))})
    df = market_data.fetch_ohlcv(client, "SPY", "15m", NOW)
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert len(df) == 5
    assert df.index.is_monotonic_increasing
    assert df["close"].iloc[-1] == 104.5


def test_fetch_ohlcv_drops_forming_bar_and_dedupes():
    forming = fake_bar(NOW - timedelta(seconds=60), 1, 2, 0, 1)  # still forming
    stamp = NOW - timedelta(seconds=900)
    first = fake_bar(stamp, 1, 2, 0, 1)
    rewrite = fake_bar(stamp, 1, 2, 0, 9)  # same stamp: last write wins
    client = FakeStockDataClient(bars_by_symbol={"SPY": [first, forming, rewrite]})
    df = market_data.fetch_ohlcv(client, "SPY", "15m", NOW)
    assert len(df) == 1 and df["close"].iloc[0] == 9


def test_fetch_ohlcv_missing_symbol_is_empty_not_invented():
    df = market_data.fetch_ohlcv(FakeStockDataClient(), "SPY", "15m", NOW)
    assert df.empty and list(df.columns) == ["open", "high", "low", "close", "volume"]


def test_fetch_ohlcv_respects_lookback():
    client = FakeStockDataClient(bars_by_symbol={"SPY": bars_ending_now(50)})
    df = market_data.fetch_ohlcv(client, "SPY", "15m", NOW, lookback_bars=10)
    assert len(df) == 10 and df["close"].iloc[-1] == 149.5


def test_fetch_ohlcv_wraps_vendor_error_to_type_name_only():
    client = FakeStockDataClient(bars_error=RuntimeError(f"boom {SECRET}"))
    with pytest.raises(market_data.MarketDataError) as excinfo:
        market_data.fetch_ohlcv(client, "SPY", "15m", NOW)
    assert SECRET not in str(excinfo.value)
    assert "RuntimeError" in str(excinfo.value)
