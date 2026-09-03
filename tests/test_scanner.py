"""scanner: the pure ranking, snapshot parsing, and the client wrapper on fakes."""

from __future__ import annotations

from types import SimpleNamespace

import scanner


def row(symbol, change=5.0, rng=6.0, price=50.0, trades=5000, volume=1e6):
    return scanner.InPlay(symbol=symbol, change_pct=change, range_pct=rng, price=price, trades=trades, volume=volume)


def test_rank_filters_penny_thin_and_quiet_names_and_orders_by_score():
    rows = [
        row("BIG", change=8.0, rng=10.0),         # score 18 -> first
        row("MID", change=-4.0, rng=5.0),         # score 9
        row("PENNY", change=70.0, rng=70.0, price=0.44),
        row("THIN", change=30.0, rng=20.0, trades=40),
        row("QUIET", change=1.0, rng=2.0),        # below min_move_pct
        row("HELD", change=10.0, rng=10.0),       # strongest of all, but excluded by the caller
        row("RANGE", change=0.5, rng=4.0),        # range alone qualifies
    ]
    picked = scanner.rank(rows, min_price=10, min_trades=2000, min_move_pct=3.0, top=5, exclude=("HELD",))
    assert [r.symbol for r in picked] == ["BIG", "MID", "RANGE"]
    assert [r.symbol for r in scanner.rank(rows, min_price=10, min_trades=2000, min_move_pct=3.0, top=1)] == ["HELD"]


def test_snapshot_row_reads_todays_and_yesterdays_bars():
    snap = SimpleNamespace(daily_bar=SimpleNamespace(close=110.0, high=115.0, low=100.0, trade_count=3000, volume=2e6),
                           previous_daily_bar=SimpleNamespace(close=100.0))
    r = scanner.snapshot_row("SNOW", snap)
    assert r is not None and r.change_pct == 10.0 and r.range_pct == 15.0 and r.trades == 3000 and r.price == 110.0
    assert scanner.snapshot_row("X", SimpleNamespace(daily_bar=None, previous_daily_bar=None)) is None
    assert scanner.snapshot_row("Y", SimpleNamespace(daily_bar=snap.daily_bar, previous_daily_bar=SimpleNamespace(close=0))) is None


class FakeScreener:
    def get_most_actives(self, request):
        return SimpleNamespace(most_actives=[SimpleNamespace(symbol=s) for s in ("SNOW", "SPWR", "NOOPT")])

    def get_market_movers(self, request):
        return SimpleNamespace(gainers=[SimpleNamespace(symbol="CHPT")], losers=[SimpleNamespace(symbol="RARE")])


class FakeTrading:
    calls = 0

    def get_all_assets(self, request):
        FakeTrading.calls += 1
        return [
            SimpleNamespace(symbol="SNOW", tradable=True, attributes=["has_options"]),
            SimpleNamespace(symbol="SPWR", tradable=True, attributes=["has_options"]),
            SimpleNamespace(symbol="CHPT", tradable=True, attributes=["has_options", "overnight_tradable"]),
            SimpleNamespace(symbol="RARE", tradable=False, attributes=["has_options"]),  # not tradable
            SimpleNamespace(symbol="NOOPT", tradable=True, attributes=[]),
        ]


class FakeStock:
    def __init__(self):
        self.requested = None

    def get_stock_snapshot(self, request):
        self.requested = list(request.symbol_or_symbols)
        bar = lambda close, high, low, trades: SimpleNamespace(close=close, high=high, low=low, trade_count=trades, volume=1e6)  # noqa: E731
        return {
            "SNOW": SimpleNamespace(daily_bar=bar(372.0, 380.0, 350.0, 8686), previous_daily_bar=SimpleNamespace(close=306.0)),
            "SPWR": SimpleNamespace(daily_bar=bar(0.44, 0.5, 0.2, 3697), previous_daily_bar=SimpleNamespace(close=0.26)),
            "CHPT": SimpleNamespace(daily_bar=bar(8.17, 9.0, 6.0, 4273), previous_daily_bar=SimpleNamespace(close=5.22)),
        }


def test_scanner_uses_optionable_names_only_and_caches_the_assets(monkeypatch):
    monkeypatch.setattr(scanner.settings, "SCANNER_MIN_PRICE", 10.0)
    monkeypatch.setattr(scanner.settings, "SCANNER_MIN_TRADES", 2000)
    monkeypatch.setattr(scanner.settings, "SCANNER_MIN_MOVE_PCT", 3.0)
    monkeypatch.setattr(scanner.settings, "SCANNER_TOP", 6)
    FakeTrading.calls = 0
    stock = FakeStock()
    s = scanner.Scanner(FakeScreener(), FakeTrading(), stock)
    picked = s.in_play(exclude=("SPY",))
    assert stock.requested == ["CHPT", "SNOW", "SPWR"]  # RARE not tradable, NOOPT has no options
    assert [r.symbol for r in picked] == ["SNOW"]  # SPWR is a penny name, CHPT is below min_price
    assert round(picked[0].change_pct, 1) == 21.6
    s.in_play()
    assert FakeTrading.calls == 1  # optionable set cached for the hour
