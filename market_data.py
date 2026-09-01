"""Stock OHLCV market data: one symbol at a time, any bar timeframe, as a DataFrame.

Owns the BAR_TIMEFRAME parsing shared with broker.load_config. Imports only the
SDK, pandas and stdlib — never broker. Vendor exceptions are reduced to their
type name (`from None`) so request details and credentials cannot leak.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import pandas as pd
from alpaca.data.enums import DataFeed
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

import settings

STOCK_FEED = DataFeed.IEX
DEFAULT_LOOKBACK_BARS = 150  # enough warmup for MACD(12/26/9) and Wilder smoothing

_SDK_UNITS = {
    "m": TimeFrameUnit.Minute,
    "h": TimeFrameUnit.Hour,
    "d": TimeFrameUnit.Day,
    "w": TimeFrameUnit.Week,
}


class MarketDataError(Exception):
    pass


def parse_timeframe(raw: str) -> tuple[int, str, int]:
    """'15m' -> (15, 'm', 900). Raises MarketDataError on anything unrecognized."""
    try:
        return settings.parse_timeframe(raw)
    except settings.SettingsError as error:
        raise MarketDataError(str(error)) from None


def fetch_ohlcv(
    stock_client: Any,
    symbol: str,
    timeframe: str,
    now: datetime,
    lookback_bars: int = DEFAULT_LOOKBACK_BARS,
) -> pd.DataFrame:
    """Completed OHLCV bars for ONE symbol at the given timeframe.

    Returns a DataFrame indexed by UTC bar-start timestamp with columns
    open/high/low/close/volume, sorted, deduplicated, and with the still-forming
    bar dropped. Missing data yields an empty frame, never invented rows.
    """
    amount, unit, seconds = parse_timeframe(timeframe)
    # Generous calendar window so nights/weekends still leave `lookback_bars`
    # completed bars; extra rows are harmless (callers slice from the end).
    window = timedelta(seconds=seconds * (lookback_bars + 2) * 5) + timedelta(days=3)
    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame(amount, _SDK_UNITS[unit]),
        start=now - window,
        feed=STOCK_FEED,
    )
    try:
        raw = stock_client.get_stock_bars(request)
    except Exception as error:
        raise MarketDataError(f"bars read for {symbol} failed: {type(error).__name__}") from None

    rows = []
    for bar in raw.data.get(symbol, []):
        stamp = getattr(bar, "timestamp", None)
        if stamp is None or stamp.timestamp() + seconds > now.timestamp():
            continue  # forming bar (or unstamped garbage) is never used
        try:
            rows.append(
                {
                    "timestamp": stamp,
                    "open": float(bar.open),
                    "high": float(bar.high),
                    "low": float(bar.low),
                    "close": float(bar.close),
                    "volume": float(bar.volume),
                }
            )
        except (TypeError, ValueError, AttributeError):
            continue  # a malformed bar is dropped, not repaired

    frame = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    frame = frame.drop_duplicates(subset="timestamp", keep="last")
    frame = frame.set_index("timestamp").sort_index()
    return frame.tail(lookback_bars)
