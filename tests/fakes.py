"""Hand-written duck-typed fakes for the Alpaca clients. No network anywhere."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace

NOW = datetime(2026, 8, 31, 15, 0, tzinfo=timezone.utc)
TODAY = NOW.date()


def fake_clock(is_open: bool = True, server_time: datetime = NOW) -> SimpleNamespace:
    return SimpleNamespace(
        timestamp=server_time, is_open=is_open, next_close=server_time + timedelta(hours=3)
    )


def fake_account(equity: float | None = 100_000.0, level: int = 3) -> SimpleNamespace:
    return SimpleNamespace(equity=equity, options_trading_level=level)


def fake_position(symbol: str, qty: int, avg_entry_price: float, side: str = "long") -> SimpleNamespace:
    return SimpleNamespace(
        symbol=symbol,
        qty=str(abs(qty)),
        side=side,
        avg_entry_price=str(avg_entry_price),
        asset_class="us_option",
    )


def fake_contract(symbol: str, strike: float, expiration: date, open_interest: int = 500) -> SimpleNamespace:
    return SimpleNamespace(
        symbol=symbol,
        strike_price=str(strike),
        expiration_date=expiration,
        open_interest=str(open_interest),
    )


def fake_snapshot(
    bid: float | None,
    ask: float | None,
    iv: float | None = 0.20,
    stamp: datetime = NOW,
) -> SimpleNamespace:
    quote = None
    if bid is not None or ask is not None:
        quote = SimpleNamespace(bid_price=bid, ask_price=ask, timestamp=stamp)
    return SimpleNamespace(latest_quote=quote, implied_volatility=iv)


class FakeTradingClient:
    def __init__(
        self,
        *,
        account=None,
        clock=None,
        positions=(),
        orders=(),
        contracts=(),
        submit_error: Exception | None = None,
    ):
        self.account = account or fake_account()
        self.clock = clock or fake_clock()
        self.positions = list(positions)
        self.orders = list(orders)
        self.contracts = list(contracts)
        self.submit_error = submit_error
        self.submitted: list = []

    def get_clock(self):
        return self.clock

    def get_account(self):
        return self.account

    def get_all_positions(self):
        return self.positions

    def get_orders(self, request):
        return self.orders

    def get_option_contracts(self, request):
        return SimpleNamespace(option_contracts=self.contracts, next_page_token=None)

    def submit_order(self, request):
        if self.submit_error is not None:
            raise self.submit_error
        self.submitted.append(request)
        return SimpleNamespace(id="order-1", status="accepted")


def fake_bar(stamp: datetime, open_: float, high: float, low: float, close: float,
             volume: float = 1000.0) -> SimpleNamespace:
    return SimpleNamespace(timestamp=stamp, open=open_, high=high, low=low,
                           close=close, volume=volume)


def quiet_bars(count: int = 60, *, end: datetime | None = None, bar_seconds: int = 900,
               base: float = 100.0) -> list[SimpleNamespace]:
    """Gently drifting completed bars: ATR ~1, tiny bodies/gaps, no events."""
    end = end if end is not None else NOW - timedelta(seconds=bar_seconds)
    bars = []
    for i in range(count):
        stamp = end - timedelta(seconds=bar_seconds * (count - 1 - i))
        close = base + 0.05 * i
        bars.append(fake_bar(stamp, close - 0.02, close + 0.5, close - 0.5, close))
    return bars


def breakout_bars(count: int = 60, direction: str = "up", **kwargs) -> list[SimpleNamespace]:
    """quiet_bars but the last completed bar has a 5-point body (no gap):
    breakout_up, or breakout_down with direction="down"."""
    bars = quiet_bars(count, **kwargs)
    prev_close = bars[-2].close
    body = 5.0 if direction == "up" else -5.0
    close = prev_close + body
    bars[-1] = fake_bar(bars[-1].timestamp, prev_close, max(prev_close, close) + 0.2,
                        min(prev_close, close) - 0.2, close)
    return bars


class FakeStockDataClient:
    def __init__(self, bars_by_symbol=None, quotes_by_symbol=None, bars_error=None):
        self.bars_by_symbol = bars_by_symbol or {}
        self.quotes_by_symbol = quotes_by_symbol or {}
        self.bars_error = bars_error

    def get_stock_bars(self, request):
        if self.bars_error is not None:
            raise self.bars_error
        return SimpleNamespace(data=self.bars_by_symbol)

    def get_stock_latest_quote(self, request):
        return {
            symbol: SimpleNamespace(bid_price=bid, ask_price=ask)
            for symbol, (bid, ask) in self.quotes_by_symbol.items()
        }


class FakeOptionDataClient:
    """Serves snapshot dicts; pass a list to serve different answers per call."""

    def __init__(self, snapshots):
        self.queue = snapshots if isinstance(snapshots, list) else [snapshots]
        self.calls = 0

    def get_option_snapshot(self, request):
        snapshots = self.queue[min(self.calls, len(self.queue) - 1)]
        self.calls += 1
        return {s: snapshots[s] for s in request.symbol_or_symbols if s in snapshots}
