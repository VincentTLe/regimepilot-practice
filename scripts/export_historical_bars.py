"""Export real historical SPY minute bars to a CSV for backtest.py.

Read-only, same as history.py: uses the paper-only client construction from
smoke_test.build_clients and history.HISTORICAL_FEED (IEX), so this never
needs (or can accidentally hit) anything beyond a market-data read. This
script submits nothing and is not part of the trading pipeline; it exists
only to produce input for backtest.run_backtest.

Usage:
    uv run python scripts/export_historical_bars.py \\
        --start 2026-03-01 --end 2026-08-27 --out data/spy_minute_bars.csv

Requires the same .env / ALPACA_API_KEY / ALPACA_SECRET_KEY your live agent
already uses -- no new credential is needed. Alpaca's historical bar limits
depend on your subscription tier: IEX free-tier data typically covers recent
history; if a wide date range comes back thin, check your Alpaca plan's
historical data window before assuming the pipeline is broken.
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime, timedelta, timezone

from alpaca.data.enums import DataFeed
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from regimepilot.config import ConfigError, load_settings
from regimepilot.models import UNDERLYING_SYMBOL
from regimepilot.observer import normalize_bar
from regimepilot.smoke_test import build_clients

# Kept for backward compatibility with any external reference; no longer
# used to cap a request, since a request-level limit truncates the whole
# date range rather than paging it (see fetch_all_minute_bars).
PAGE_LIMIT = 10_000


def fetch_all_minute_bars(data_client, *, symbol: str, start: datetime, end: datetime):
    """Yield every 1-minute bar in [start, end].

    ``get_stock_bars`` already walks every page internally and returns the
    full result -- it is not paginated by this function. ``limit`` is
    deliberately omitted from the request: passing a limit caps the *total*
    number of bars returned across the whole date range, not the page size,
    so setting one would silently truncate a wide date range to that many
    bars regardless of how many sessions it actually spans.
    """
    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame(amount=1, unit=TimeFrameUnit.Minute),
        start=start,
        end=end,
        feed=DataFeed.IEX,
    )
    response = data_client.get_stock_bars(request)
    data = response if isinstance(response, dict) else getattr(response, "data", None)
    rows = list((data or {}).get(symbol) or [])
    for row in rows:
        bar = normalize_bar(row)
        if bar is not None:
            yield bar


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True, help="YYYY-MM-DD, inclusive")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD, inclusive")
    parser.add_argument("--symbol", default=UNDERLYING_SYMBOL)
    parser.add_argument("--out", required=True, help="Output CSV path")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    # +1 day so --end is inclusive of that whole session.
    end = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc) + timedelta(days=1)

    try:
        settings = load_settings()
        _trading_client, data_client = build_clients(settings)
    except ConfigError as error:
        print(f"config error: {error}", file=sys.stderr)
        return 1

    row_count = 0
    with open(args.out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp", "open", "high", "low", "close", "volume"])
        for bar in fetch_all_minute_bars(data_client, symbol=args.symbol, start=start, end=end):
            if bar.timestamp is None or bar.close is None:
                continue
            writer.writerow([bar.timestamp.isoformat(), bar.open, bar.high, bar.low, bar.close, bar.volume])
            row_count += 1

    print(f"wrote {row_count} minute bars for {args.symbol} to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())