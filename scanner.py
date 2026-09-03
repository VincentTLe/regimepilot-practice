"""Market-wide "in play" scanner: names with abnormal range / participation today
that have listed options, added to the cycle's universe next to the static
whitelist in settings.yaml.

Sources, one Alpaca data call each: most actives by volume, most actives by
trade count, top gainers and top losers. Filters: tradable US equity carrying
the `has_options` asset attribute, price >= min_price (penny chains are junk),
IEX prints today >= min_trades (the tape needs prints), today's |change| or
range >= min_move_pct of yesterday's close. Ranked by range + |change|, top N.

Every failure returns [] and is logged by the caller: the loop keeps trading
the static list. Nothing here places orders; the option screener still decides
whether a name has a tradeable chain.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Iterable

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.historical.screener import ScreenerClient
from alpaca.data.requests import MarketMoversRequest, MostActivesRequest, StockSnapshotRequest
from alpaca.trading.enums import AssetClass, AssetStatus
from alpaca.trading.requests import GetAssetsRequest

import market_data
import settings

SCREENER_TOP = 100  # per most-actives list
MOVERS_TOP = 50  # gainers and losers each
OPTIONABLE_TTL_SECONDS = 3600.0  # the assets list (~14k rows) is re-read at most hourly


@dataclass(frozen=True)
class InPlay:
    symbol: str
    change_pct: float  # today's last vs yesterday's close
    range_pct: float  # today's high - low, as % of yesterday's close
    price: float
    trades: int  # IEX prints today
    volume: float

    @property
    def score(self) -> float:
        return abs(self.change_pct) + self.range_pct


def rank(
    rows: Iterable[InPlay],
    *,
    min_price: float,
    min_trades: int,
    min_move_pct: float,
    top: int,
    exclude: Iterable[str] = (),
) -> list[InPlay]:
    """Pure filter + order: the strongest `top` names not in `exclude`."""
    skip = set(exclude)
    keep = [
        r
        for r in rows
        if r.symbol not in skip
        and r.price >= min_price
        and r.trades >= min_trades
        and max(abs(r.change_pct), r.range_pct) >= min_move_pct
    ]
    return sorted(keep, key=lambda r: (-r.score, r.symbol))[:top]


def snapshot_row(symbol: str, snapshot: Any) -> InPlay | None:
    """One InPlay from a stock snapshot; None when today's or yesterday's bar is missing."""
    today = getattr(snapshot, "daily_bar", None)
    prev = getattr(snapshot, "previous_daily_bar", None)
    if today is None or prev is None or not prev.close or today.close is None:
        return None
    base = float(prev.close)
    return InPlay(
        symbol=symbol,
        change_pct=100.0 * (float(today.close) - base) / base,
        range_pct=100.0 * (float(today.high) - float(today.low)) / base,
        price=float(today.close),
        trades=int(today.trade_count or 0),
        volume=float(today.volume or 0.0),
    )


class Scanner:
    """Holds the screener client and an hourly cache of optionable symbols."""

    def __init__(self, screener: Any, trading: Any, stock_data: Any) -> None:
        self._screener = screener
        self._trading = trading
        self._stock = stock_data
        self._optionable: set[str] = set()
        self._optionable_at = 0.0

    @classmethod
    def build(cls, api_key: str, secret_key: str, trading: Any) -> "Scanner":
        return cls(ScreenerClient(api_key, secret_key), trading, StockHistoricalDataClient(api_key, secret_key))

    def optionable(self) -> set[str]:
        if not self._optionable or time.monotonic() - self._optionable_at > OPTIONABLE_TTL_SECONDS:
            assets = self._trading.get_all_assets(
                GetAssetsRequest(status=AssetStatus.ACTIVE, asset_class=AssetClass.US_EQUITY)
            )
            self._optionable = {
                a.symbol for a in assets if a.tradable and "has_options" in (a.attributes or [])
            }
            self._optionable_at = time.monotonic()
        return self._optionable

    def candidates(self) -> list[InPlay]:
        """Every screener name with options and a usable snapshot (unfiltered)."""
        by_volume = self._screener.get_most_actives(MostActivesRequest(top=SCREENER_TOP, by="volume"))
        by_trades = self._screener.get_most_actives(MostActivesRequest(top=SCREENER_TOP, by="trades"))
        movers = self._screener.get_market_movers(MarketMoversRequest(top=MOVERS_TOP))
        symbols = (
            {a.symbol for a in by_volume.most_actives}
            | {a.symbol for a in by_trades.most_actives}
            | {m.symbol for m in movers.gainers}
            | {m.symbol for m in movers.losers}
        )
        symbols = sorted(symbols & self.optionable())
        if not symbols:
            return []
        snapshots = self._stock.get_stock_snapshot(
            StockSnapshotRequest(symbol_or_symbols=symbols, feed=market_data.STOCK_FEED)
        )
        rows = [snapshot_row(symbol, snap) for symbol, snap in snapshots.items()]
        return [r for r in rows if r is not None]

    def in_play(self, exclude: Iterable[str] = ()) -> list[InPlay]:
        """The scanned names to add this cycle, by settings.yaml `scanner` values."""
        return rank(
            self.candidates(),
            min_price=settings.SCANNER_MIN_PRICE,
            min_trades=settings.SCANNER_MIN_TRADES,
            min_move_pct=settings.SCANNER_MIN_MOVE_PCT,
            top=settings.SCANNER_TOP,
            exclude=exclude,
        )
