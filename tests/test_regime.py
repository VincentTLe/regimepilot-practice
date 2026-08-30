"""Regime classification tests.

Expected values are computed by hand from the fixture bars, the same
convention ``test_features.py`` uses, never by re-running the code under test.
"""

import socket
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from regimepilot.models import OhlcvBar
from regimepilot.regime import (
    HIGH_VOL_ANNUALIZED_THRESHOLD,
    TREND_STRENGTH_THRESHOLD,
    RegimeReading,
    annualize_realized_vol_30m,
    classify_regime,
    iv_rank,
    trend_strength_adx,
)

NY = ZoneInfo("America/New_York")


def daily(day, *, high, low, close):
    return OhlcvBar(
        timestamp=datetime(day.year, day.month, day.day, 0, 0, tzinfo=NY),
        high=high,
        low=low,
        close=close,
    )


def test_annualize_realized_vol_30m_none_stays_none():
    assert annualize_realized_vol_30m(None) is None


def test_annualize_realized_vol_30m_scales_up():
    # A tiny 30-minute raw reading should annualize to a much larger number.
    raw = 0.002
    annualized = annualize_realized_vol_30m(raw)
    assert annualized is not None
    assert annualized > raw
    # Scaling factor is sqrt(98280 / 30) ~= 57.24, hand-computed.
    assert annualized == pytest.approx(raw * 57.24, rel=1e-2)


def test_trend_strength_adx_none_below_minimum_bars():
    bars = [daily(date(2026, 8, d), high=101, low=99, close=100) for d in range(1, 5)]
    assert trend_strength_adx(bars, period=14) is None


def test_trend_strength_adx_none_when_ohlc_missing():
    bars = [
        OhlcvBar(timestamp=datetime(2026, 8, d, tzinfo=NY), close=100.0)
        for d in range(1, 20)
    ]
    assert trend_strength_adx(bars, period=14) is None


def test_trend_strength_adx_high_for_a_clean_uptrend():
    # Fifteen consecutive days each stepping the whole range higher: a
    # textbook one-directional trend, so DX should be at or near 100.
    bars = [
        daily(date(2026, 8, 1) + timedelta(days=i), high=100 + i, low=99 + i, close=100 + i)
        for i in range(15)
    ]
    dx = trend_strength_adx(bars, period=14)
    assert dx is not None
    assert dx > 90.0


def test_trend_strength_adx_low_for_pure_chop():
    # Alternating up/down of equal magnitude every day: +DM and -DM roughly
    # cancel out, so DX should be low.
    bars = []
    price = 100.0
    for i in range(20):
        step = 1.0 if i % 2 == 0 else -1.0
        price += step
        bars.append(daily(date(2026, 8, 1) + timedelta(days=i), high=price + 0.5, low=price - 0.5, close=price))
    dx = trend_strength_adx(bars, period=14)
    assert dx is not None
    assert dx < 30.0


def test_iv_rank_none_without_current_iv_or_history():
    assert iv_rank(None, [0.1, 0.2]) is None
    assert iv_rank(0.15, []) is None


def test_iv_rank_midpoint():
    assert iv_rank(0.15, [0.10, 0.20]) == pytest.approx(50.0)


def test_iv_rank_degenerate_history_returns_fifty():
    assert iv_rank(0.15, [0.10, 0.10, 0.10]) == pytest.approx(50.0)


def test_iv_rank_clamped_to_bounds():
    assert iv_rank(1.0, [0.1, 0.2]) == pytest.approx(100.0)
    assert iv_rank(-1.0, [0.1, 0.2]) == pytest.approx(0.0)


UPTREND_BARS = [
    daily(date(2026, 8, 1) + timedelta(days=i), high=100 + i, low=99 + i, close=100 + i)
    for i in range(15)
]
CHOP_BARS = []
_price = 100.0
for _i in range(20):
    _price += 1.0 if _i % 2 == 0 else -1.0
    CHOP_BARS.append(
        daily(date(2026, 8, 1) + timedelta(days=_i), high=_price + 0.5, low=_price - 0.5, close=_price)
    )


def test_classify_regime_unknown_when_vol_missing():
    reading = classify_regime(realized_vol_30m=None, return_60m=0.01, trend_bars=UPTREND_BARS)
    assert reading.label == "unknown"


def test_classify_regime_unknown_when_trend_bars_insufficient():
    reading = classify_regime(realized_vol_30m=0.001, return_60m=0.01, trend_bars=[])
    assert reading.label == "unknown"


def test_classify_regime_trending_up():
    reading = classify_regime(realized_vol_30m=0.0005, return_60m=0.01, trend_bars=UPTREND_BARS)
    assert reading.label == "trending_up"
    assert reading.trend_strength is not None
    assert reading.trend_strength >= TREND_STRENGTH_THRESHOLD


def test_classify_regime_trending_down():
    downtrend = [
        daily(date(2026, 8, 1) + timedelta(days=i), high=100 - i + 1, low=99 - i, close=100 - i)
        for i in range(15)
    ]
    reading = classify_regime(realized_vol_30m=0.0005, return_60m=-0.01, trend_bars=downtrend)
    assert reading.label == "trending_down"


def test_classify_regime_high_vol_chop():
    # Chop (low trend strength) with a large annualized vol reading.
    high_raw_vol = HIGH_VOL_ANNUALIZED_THRESHOLD / 57.24 + 0.01
    reading = classify_regime(realized_vol_30m=high_raw_vol, return_60m=0.0001, trend_bars=CHOP_BARS)
    assert reading.label == "high_vol_chop"


def test_classify_regime_low_vol_drift():
    tiny_vol = 0.0001
    reading = classify_regime(realized_vol_30m=tiny_vol, return_60m=0.0, trend_bars=CHOP_BARS)
    assert reading.label == "low_vol_drift"


def test_classify_regime_carries_iv_rank_without_using_it_for_the_label():
    reading = classify_regime(
        realized_vol_30m=0.0001,
        return_60m=0.0,
        trend_bars=CHOP_BARS,
        current_iv=0.30,
        iv_history=[0.10, 0.50],
    )
    assert reading.label == "low_vol_drift"
    assert reading.iv_rank == pytest.approx(50.0)


def test_regime_reading_is_frozen():
    reading = RegimeReading()
    with pytest.raises(Exception):
        reading.label = "trending_up"


def test_regime_never_touches_the_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("regime must not open a socket")

    monkeypatch.setattr(socket, "socket", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)

    classify_regime(realized_vol_30m=0.001, return_60m=0.01, trend_bars=UPTREND_BARS)
