"""FeaturePacket tests.

Every input here is hand-built. Nothing in this file touches Alpaca or the
network, because nothing in ``regimepilot.features`` is allowed to either.

Expected values are written as closed-form arithmetic on the fixture prices
(``110 / 88 - 1``), never by re-running the code under test.
"""

import math
import socket
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from regimepilot import features as features_module
from regimepilot.features import FeaturePacket, build_feature_packet
from regimepilot.models import OhlcvBar

NY = ZoneInfo("America/New_York")

# A Monday. Its previous regular session is Friday 2026-08-21.
SESSION = date(2026, 8, 24)
PREVIOUS_SESSION = date(2026, 8, 21)

PREVIOUS_CLOSE = 250.0


def et(hour, minute, second=0, day=SESSION):
    """A timestamp in America/New_York, which is how the session is defined."""
    return datetime(day.year, day.month, day.day, hour, minute, second, tzinfo=NY)


def bar(hour, minute, close, *, open_=None, day=SESSION):
    return OhlcvBar(
        timestamp=et(hour, minute, day=day),
        open=close if open_ is None else open_,
        high=None if close is None else close,
        low=None if close is None else close,
        close=close,
        volume=1000.0,
    )


def daily(day, close, open_=None):
    """A daily bar. Alpaca stamps these 00:00 New York, i.e. 04:00Z in August."""
    return OhlcvBar(
        timestamp=datetime(day.year, day.month, day.day, 0, 0, tzinfo=NY),
        open=open_,
        close=close,
    )


DAILY_BARS = [
    daily(date(2026, 8, 20), 240.0),
    daily(PREVIOUS_SESSION, PREVIOUS_CLOSE),
]


def session_minutes(start=(9, 30), end=(10, 35), day=SESSION):
    """Every minute timestamp from ``start`` to ``end`` inclusive."""
    stamp = et(*start, day=day)
    last = et(*end, day=day)
    stamps = []
    while stamp <= last:
        stamps.append(stamp)
        stamp += timedelta(minutes=1)
    return stamps


# --------------------------------------------------------------------------
# The main fixture. 66 completed bars, 09:30 -> 10:35, all closes 100.0 except:
#   09:30  open  200.0   (the regular session open)
#   09:35  close  88.0   (60 minutes before the latest bar)
#   10:35  close 110.0   (the latest completed bar)
# --------------------------------------------------------------------------
def main_bars():
    bars = []
    for stamp in session_minutes():
        hm = (stamp.hour, stamp.minute)
        close = {(9, 35): 88.0, (10, 35): 110.0}.get(hm, 100.0)
        open_ = 200.0 if hm == (9, 30) else close
        bars.append(bar(stamp.hour, stamp.minute, close, open_=open_))
    return bars


OBSERVED_AT = et(10, 36, 5)


def build(**overrides):
    kwargs = dict(
        observed_at=OBSERVED_AT,
        minute_bars=main_bars(),
        daily_bars=DAILY_BARS,
        bid=99.90,
        ask=100.10,
        market_is_open=True,
        session_close_at=et(16, 0),
    )
    kwargs.update(overrides)
    return build_feature_packet(**kwargs)


# --------------------------------------------------------------------------
# 1-5. the five signal features, each to an exact hand-checked value
# --------------------------------------------------------------------------


def test_return_15m_is_latest_close_over_the_close_fifteen_minutes_earlier():
    # latest bar 10:35 closes at 110.0; the 10:20 bar closes at 100.0.
    assert build().return_15m == pytest.approx(110.0 / 100.0 - 1)
    assert build().return_15m == pytest.approx(0.10)


def test_return_60m_is_latest_close_over_the_close_sixty_minutes_earlier():
    # latest bar 10:35 closes at 110.0; the 09:35 bar closes at 88.0.
    assert build().return_60m == pytest.approx(110.0 / 88.0 - 1)
    assert build().return_60m == pytest.approx(0.25)


def test_return_since_open_uses_the_open_of_the_0930_bar():
    # 110.0 / 200.0 - 1. The 09:30 *open* is used, not its close.
    assert build().return_since_open == pytest.approx(110.0 / 200.0 - 1)
    assert build().return_since_open == pytest.approx(-0.45)


def test_overnight_gap_is_session_open_over_previous_regular_session_close():
    # 200.0 / 250.0 - 1.
    assert build().overnight_gap_pct == pytest.approx(200.0 / 250.0 - 1)
    assert build().overnight_gap_pct == pytest.approx(-0.20)


def test_realized_vol_30m_is_the_unscaled_root_sum_of_squared_log_returns():
    """The last 31 closes are 10:05..10:34 at 100.0 then 10:35 at 110.0.

    Twenty-nine of the thirty log returns are exactly zero, so the whole sum of
    squares collapses to ln(1.1)**2 and the result is |ln(1.1)|.
    """
    assert build().realized_vol_30m == pytest.approx(math.log(1.1))
    assert build().realized_vol_30m == pytest.approx(0.09531017980432486)


def test_realized_vol_30m_is_not_annualized_or_scaled():
    value = build().realized_vol_30m
    unscaled = math.log(1.1)

    for factor in (math.sqrt(252), math.sqrt(390), math.sqrt(252 * 390), 100.0):
        assert value != pytest.approx(unscaled * factor)
    assert value < 1.0  # a raw 30-minute decimal, not a percentage


def test_realized_vol_30m_over_a_constant_ratio_series():
    """A second, independent shape: 31 closes alternating 100/200.

    Every one of the thirty log returns is +/-ln(2), so the closed form is
    sqrt(30) * ln(2) with no reference to the implementation.
    """
    stamps = session_minutes(end=(10, 0))  # 09:30..10:00 -> exactly 31 bars
    bars = [
        bar(s.hour, s.minute, 100.0 if index % 2 == 0 else 200.0)
        for index, s in enumerate(stamps)
    ]
    packet = build(minute_bars=bars, observed_at=et(10, 1, 5))

    assert packet.realized_vol_30m == pytest.approx(math.sqrt(30) * math.log(2))


# --------------------------------------------------------------------------
# 6-8. not enough history is null, never a shorter-horizon substitute
# --------------------------------------------------------------------------


def test_realized_vol_30m_needs_thirty_one_closes():
    thirty_one = session_minutes(end=(10, 0))
    thirty = session_minutes(end=(9, 59))
    assert len(thirty_one) == 31 and len(thirty) == 30

    def vol(stamps):
        bars = [bar(s.hour, s.minute, 100.0 + index) for index, s in enumerate(stamps)]
        return build(
            minute_bars=bars,
            observed_at=stamps[-1] + timedelta(minutes=1, seconds=5),
        ).realized_vol_30m

    assert vol(thirty_one) is not None
    assert vol(thirty) is None


def test_insufficient_fifteen_minute_history_is_null():
    """Fourteen minutes of session is not a fifteen-minute return."""
    bars = [bar(s.hour, s.minute, 100.0) for s in session_minutes(end=(9, 44))]
    packet = build(minute_bars=bars, observed_at=et(9, 45, 5))

    assert packet.return_15m is None
    assert packet.return_60m is None
    # The features that *are* computable still are.
    assert packet.return_since_open is not None


def test_insufficient_sixty_minute_history_is_null():
    """Fifty-nine minutes of session yields the 15m return but not the 60m one."""
    bars = [bar(s.hour, s.minute, 100.0) for s in session_minutes(end=(10, 28))]
    packet = build(minute_bars=bars, observed_at=et(10, 29, 5))

    assert packet.return_15m is not None
    assert packet.return_60m is None


# --------------------------------------------------------------------------
# 9-12. only completed bars, only this regular session
# --------------------------------------------------------------------------


def test_pre_market_bars_are_excluded():
    """A 09:29 bar must not become the "fifteen minutes earlier" close."""
    premarket = [bar(9, 29, 999.0), bar(9, 15, 999.0), bar(4, 0, 999.0)]
    session = [
        bar(s.hour, s.minute, 100.0, open_=200.0 if s.minute == 30 else 100.0)
        for s in session_minutes(end=(9, 44))
    ]
    packet = build(minute_bars=premarket + session, observed_at=et(9, 45, 5))

    # 09:44 - 15min = 09:29, which is pre-market, so there is no such close.
    assert packet.return_15m is None
    # The pre-market open is never the session open.
    assert packet.return_since_open == pytest.approx(100.0 / 200.0 - 1)
    assert packet.overnight_gap_pct == pytest.approx(200.0 / 250.0 - 1)


def test_pre_market_bars_do_not_pad_the_volatility_window():
    premarket = [bar(9, 30 - offset, 100.0) for offset in range(1, 31)]
    session = [bar(s.hour, s.minute, 100.0) for s in session_minutes(end=(9, 44))]

    packet = build(minute_bars=premarket + session, observed_at=et(9, 45, 5))

    # 45 bars supplied, but only 15 of them are regular session.
    assert packet.realized_vol_30m is None


def test_after_hours_bars_are_excluded():
    """The 16:00 bar Alpaca returns for an inclusive end is outside the session."""
    session = [
        bar(s.hour, s.minute, 100.0, open_=200.0 if (s.hour, s.minute) == (9, 30) else 100.0)
        for s in session_minutes(end=(15, 59))
    ]
    session[-1] = bar(15, 59, 110.0)
    after_hours = [bar(16, 0, 999.0), bar(16, 30, 999.0), bar(19, 59, 999.0)]

    packet = build(
        minute_bars=session + after_hours, observed_at=et(17, 0), market_is_open=False
    )

    # The latest usable close is 15:59's 110.0, not any 999.0.
    assert packet.return_15m == pytest.approx(110.0 / 100.0 - 1)
    assert packet.return_since_open == pytest.approx(110.0 / 200.0 - 1)
    # Age is measured from 15:59, not from the 19:59 after-hours bar.
    assert packet.bar_age_seconds == pytest.approx(2 * 3600 - 59 * 60)


def test_the_currently_forming_minute_bar_is_excluded():
    """The spec's own example: at 10:00:05 the newest usable bar begins 09:59."""
    bars = [
        bar(s.hour, s.minute, 100.0 + index)
        for index, s in enumerate(session_minutes(end=(10, 0)))
    ]
    # 09:30 is 100.0 ... 09:44 is 114.0 ... 09:59 is 129.0 ... 10:00 is 130.0.
    packet = build(minute_bars=bars, observed_at=et(10, 0, 5))

    assert packet.bar_age_seconds == pytest.approx(65.0)  # 10:00:05 - 09:59:00
    assert packet.return_15m == pytest.approx(129.0 / 114.0 - 1)  # 09:59 over 09:44


def test_a_bar_is_complete_exactly_sixty_seconds_after_it_begins():
    bars = [bar(s.hour, s.minute, 100.0) for s in session_minutes(end=(10, 0))]

    at_the_boundary = build(minute_bars=bars, observed_at=et(10, 1, 0))
    one_second_earlier = build(minute_bars=bars, observed_at=et(10, 0, 59))

    assert at_the_boundary.bar_age_seconds == pytest.approx(60.0)  # the 10:00 bar
    assert one_second_earlier.bar_age_seconds == pytest.approx(119.0)  # still 09:59


def test_previous_session_bars_do_not_fill_current_session_history():
    """Yesterday had a full session; today has five minutes. Today wins, alone."""
    yesterday = [
        bar(s.hour, s.minute, 100.0, day=PREVIOUS_SESSION)
        for s in session_minutes(end=(15, 59), day=PREVIOUS_SESSION)
    ]
    today = [
        bar(s.hour, s.minute, 100.0, open_=200.0 if s.minute == 30 else 100.0)
        for s in session_minutes(end=(9, 34))
    ]

    packet = build(minute_bars=yesterday + today, observed_at=et(9, 35, 5))

    assert packet.return_15m is None
    assert packet.return_60m is None
    assert packet.realized_vol_30m is None
    # Today's own five bars are still used where they suffice.
    assert packet.return_since_open == pytest.approx(100.0 / 200.0 - 1)


def test_a_session_with_no_completed_bars_nulls_every_intraday_feature():
    """Observing before 09:30 is the honest all-null case, not an error."""
    packet = build(minute_bars=[], observed_at=et(4, 42), market_is_open=False)

    assert packet.return_15m is None
    assert packet.return_60m is None
    assert packet.return_since_open is None
    assert packet.overnight_gap_pct is None
    assert packet.realized_vol_30m is None
    assert packet.bar_age_seconds is None
    assert isinstance(packet, FeaturePacket)


# --------------------------------------------------------------------------
# 13-14. ordering and duplicates
# --------------------------------------------------------------------------


def test_bars_supplied_out_of_order_are_sorted_before_calculation():
    ordered = main_bars()
    baseline = build(minute_bars=ordered)

    for supplied in (ordered[::-1], ordered[30:] + ordered[:30]):
        packet = build(minute_bars=supplied)
        assert packet.return_15m == pytest.approx(baseline.return_15m)
        assert packet.return_60m == pytest.approx(baseline.return_60m)
        assert packet.realized_vol_30m == pytest.approx(baseline.realized_vol_30m)
        assert packet.bar_age_seconds == pytest.approx(baseline.bar_age_seconds)


def test_duplicate_timestamps_resolve_to_the_last_bar_supplied():
    """Deterministic by rule: a later bar for a timestamp supersedes an earlier one."""
    bars = main_bars()

    last_wins = build(minute_bars=bars + [bar(10, 35, 120.0)])
    other_order = build(minute_bars=bars[:-1] + [bar(10, 35, 120.0), bar(10, 35, 110.0)])

    assert last_wins.return_15m == pytest.approx(120.0 / 100.0 - 1)
    assert other_order.return_15m == pytest.approx(110.0 / 100.0 - 1)


def test_duplicate_timestamps_do_not_inflate_the_volatility_window():
    """Thirty distinct closes duplicated is still thirty closes, not sixty."""
    stamps = session_minutes(end=(9, 59))  # 30 bars: one short of the window
    bars = [bar(s.hour, s.minute, 100.0 + index) for index, s in enumerate(stamps)]

    packet = build(minute_bars=bars + bars, observed_at=et(10, 0, 5))

    assert packet.realized_vol_30m is None


# --------------------------------------------------------------------------
# 15-17. spread_bps
# --------------------------------------------------------------------------


def test_spread_bps_exact_calculation():
    # mid = 100.00, spread = 0.20 -> 0.20 / 100.00 * 10_000 = 20 bps.
    assert build(bid=99.90, ask=100.10).spread_bps == pytest.approx(20.0)
    # A second, differently-shaped case: mid = 50.0, spread = 0.05 -> 10 bps.
    assert build(bid=49.975, ask=50.025).spread_bps == pytest.approx(10.0)


def test_missing_quote_leaves_spread_null():
    assert build(bid=None, ask=None).spread_bps is None
    assert build(bid=99.90, ask=None).spread_bps is None
    assert build(bid=None, ask=100.10).spread_bps is None


@pytest.mark.parametrize(
    "bid, ask",
    [
        (0.0, 100.10),  # bid must be > 0
        (-1.0, 100.10),
        (100.10, 99.90),  # crossed book: ask < bid
        (100.0, 0.0),  # ask must be >= bid, and the midpoint must be > 0
        (0.0, 0.0),
    ],
)
def test_invalid_bid_ask_leaves_spread_null(bid, ask):
    assert build(bid=bid, ask=ask).spread_bps is None


def test_a_locked_book_is_a_zero_spread_not_an_error():
    assert build(bid=100.0, ask=100.0).spread_bps == pytest.approx(0.0)


def test_spread_is_context_only_and_does_not_disturb_the_signals():
    wide = build(bid=1.0, ask=199.0)

    assert wide.spread_bps is not None
    assert wide.return_15m == pytest.approx(0.10)


# --------------------------------------------------------------------------
# 18. the previous regular session close
# --------------------------------------------------------------------------


def test_missing_previous_close_leaves_the_overnight_gap_null():
    assert build(daily_bars=[]).overnight_gap_pct is None
    assert build(daily_bars=[daily(PREVIOUS_SESSION, None)]).overnight_gap_pct is None
    assert build(daily_bars=[daily(PREVIOUS_SESSION, 0.0)]).overnight_gap_pct is None
    # ...while everything not built from it survives.
    assert build(daily_bars=[]).return_since_open is not None


def test_the_current_sessions_own_daily_bar_is_not_the_previous_close():
    """A daily bar already exists for today mid-session. It is not "previous"."""
    with_today = DAILY_BARS + [daily(SESSION, 111.0)]

    assert build(daily_bars=with_today).overnight_gap_pct == pytest.approx(200.0 / 250.0 - 1)


def test_the_newest_prior_daily_bar_wins_regardless_of_order():
    scrambled = list(reversed(DAILY_BARS))

    assert build(daily_bars=scrambled).overnight_gap_pct == pytest.approx(200.0 / 250.0 - 1)


def test_missing_0930_bar_nulls_both_features_built_on_the_session_open():
    """Without the 09:30 bar there is no session open, and nothing invents one."""
    bars = [bar(s.hour, s.minute, 100.0) for s in session_minutes(start=(9, 31), end=(10, 35))]
    packet = build(minute_bars=bars)

    assert packet.return_since_open is None
    assert packet.overnight_gap_pct is None
    assert packet.return_15m is not None  # unaffected


# --------------------------------------------------------------------------
# invalid prices
# --------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [0.0, -5.0, None])
def test_an_invalid_close_is_treated_as_absent_data(bad):
    bars = main_bars()
    bars[-16] = bar(10, 20, bad)  # the "fifteen minutes earlier" bar

    packet = build(minute_bars=bars)

    assert packet.return_15m is None
    assert packet.return_60m == pytest.approx(110.0 / 88.0 - 1)  # unaffected


def test_an_invalid_session_open_nulls_the_features_built_on_it():
    bars = main_bars()
    bars[0] = bar(9, 30, 100.0, open_=0.0)

    packet = build(minute_bars=bars)

    assert packet.return_since_open is None
    assert packet.overnight_gap_pct is None
    assert packet.return_15m is not None


# --------------------------------------------------------------------------
# context fields
# --------------------------------------------------------------------------


def test_context_fields_when_the_market_is_open():
    packet = build()

    assert packet.symbol == "SPY"
    assert packet.observed_at == OBSERVED_AT
    assert packet.market_is_open is True
    assert packet.data_feed == "iex"
    # 09:30:00 -> 10:36:05 is 66 minutes and 5 seconds.
    assert packet.minutes_since_open == pytest.approx(66 + 5 / 60)
    # 10:36:05 -> 16:00:00 is 323 minutes and 55 seconds.
    assert packet.minutes_to_close == pytest.approx(323 + 55 / 60)
    # The 10:35 bar begins at 10:35:00, so at 10:36:05 it is 65 seconds old.
    assert packet.bar_age_seconds == pytest.approx(65.0)


def test_session_minutes_are_null_when_the_market_is_closed():
    packet = build(observed_at=et(17, 0), market_is_open=False)

    assert packet.minutes_since_open is None
    assert packet.minutes_to_close is None
    # ...but the completed session is still measurable.
    assert packet.bar_age_seconds is not None


def test_session_minutes_are_null_when_openness_is_unknown():
    assert build(market_is_open=None).minutes_since_open is None
    assert build(market_is_open=None).minutes_to_close is None


def test_session_minutes_are_null_when_the_clock_disagrees_with_the_session():
    """Told "open" at 04:42, the honest answer is null, not a negative number."""
    packet = build(minute_bars=[], observed_at=et(4, 42), market_is_open=True)

    assert packet.minutes_since_open is None
    assert packet.minutes_to_close is None


def test_bar_age_is_measured_from_the_start_of_the_minute():
    """Alpaca stamps a minute bar with the left edge of its interval."""
    bars = [bar(s.hour, s.minute, 100.0) for s in session_minutes(end=(10, 0))]

    assert build(minute_bars=bars, observed_at=et(10, 1, 30)).bar_age_seconds == pytest.approx(90.0)
    assert build(minute_bars=bars, observed_at=et(10, 5)).bar_age_seconds == pytest.approx(300.0)


def test_observed_at_in_another_timezone_resolves_to_the_same_session():
    """14:36:05Z is 10:36:05 New York; the session is the same one."""
    utc = OBSERVED_AT.astimezone(ZoneInfo("UTC"))

    assert build(observed_at=utc).return_15m == pytest.approx(0.10)
    assert build(observed_at=utc).minutes_since_open == pytest.approx(66 + 5 / 60)


# --------------------------------------------------------------------------
# 20-23. shape, purity and read-only guarantees
# --------------------------------------------------------------------------


def test_the_packet_holds_exactly_the_agreed_fields():
    assert set(FeaturePacket.model_fields) == {
        "symbol",
        "observed_at",
        "market_is_open",
        "data_feed",
        "minutes_since_open",
        "minutes_to_close",
        "spread_bps",
        "bar_age_seconds",
        "return_15m",
        "return_60m",
        "return_since_open",
        "overnight_gap_pct",
        "realized_vol_30m",
    }


def test_the_packet_holds_no_alpaca_sdk_object():
    packet = build()

    for value in packet.model_dump().values():
        module = type(value).__module__
        assert not module.startswith("alpaca"), f"{value!r} came from {module}"
        assert isinstance(value, (str, bool, float, int, datetime, type(None)))

    # It also survives a round trip through plain JSON.
    assert FeaturePacket.model_validate_json(packet.model_dump_json()) == packet


def test_the_features_module_never_imports_alpaca():
    source = Path(features_module.__file__).read_text(encoding="utf-8")
    assert "alpaca" not in source.lower()

    for value in vars(features_module).values():
        module = getattr(value, "__module__", "") or ""
        assert not module.startswith("alpaca")


def test_feature_calculation_makes_no_network_call(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("features must not open a socket")

    monkeypatch.setattr(socket, "socket", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)

    assert build().return_15m == pytest.approx(0.10)


def test_the_features_module_exposes_no_trading_or_execution_helper():
    forbidden = (
        "submit", "cancel", "replace", "close_position", "close_all", "exercise",
        "order", "buy_call", "buy_put", "position", "size", "risk", "decide",
    )
    offenders = [
        name for name in dir(features_module) if any(word in name.lower() for word in forbidden)
    ]
    assert offenders == []


def test_the_packet_is_frozen_and_closed_to_stray_fields():
    packet = build()

    with pytest.raises(Exception):
        packet.return_15m = 0.5
    with pytest.raises(Exception):
        FeaturePacket(**{**packet.model_dump(), "rsi_14": 55.0})


def test_the_packet_carries_no_indicator_beyond_the_five_agreed_signals():
    banned = ("rsi", "macd", "sma", "ema", "vwap", "volume", "vix", "iv",
              "delta", "gamma", "theta", "vega", "sentiment", "news")
    for name in FeaturePacket.model_fields:
        assert not any(word in name.lower() for word in banned), name


# --------------------------------------------------------------------------
# minutes_to_close comes from the clock, never from an assumed 16:00
# --------------------------------------------------------------------------


def test_minutes_to_close_follows_the_clock_not_an_assumed_1600():
    """A half day closes at 13:00 New York. Assuming 16:00 over-reports by 180."""
    packet = build(observed_at=et(12, 0, 5), session_close_at=et(13, 0))

    # 12:00:05 -> 13:00:00 is 59 minutes and 55 seconds.
    assert packet.minutes_to_close == pytest.approx(59 + 55 / 60)
    # A hardcoded 16:00 would have said 239:55.
    assert packet.minutes_to_close < 60


def test_minutes_since_open_is_unchanged_by_an_early_close():
    """Half days still open at 09:30; only the close moves."""
    packet = build(observed_at=et(12, 0, 5), session_close_at=et(13, 0))

    assert packet.minutes_since_open == pytest.approx(150 + 5 / 60)


def test_session_minutes_are_null_without_a_close_time_from_the_clock():
    """An unknown close time is not a 16:00 one."""
    packet = build(session_close_at=None)

    assert packet.minutes_to_close is None
    assert packet.minutes_since_open is None
    # Nothing else is disturbed.
    assert packet.return_15m == pytest.approx(0.10)


def test_a_close_time_belonging_to_another_session_is_not_used():
    """Once a session has ended the clock's next close is the following day's."""
    tomorrow = build(session_close_at=et(16, 0, day=date(2026, 8, 25)))

    assert tomorrow.minutes_to_close is None
    assert tomorrow.minutes_since_open is None


def test_after_an_early_close_the_session_minutes_go_null():
    packet = build(observed_at=et(14, 0), session_close_at=et(13, 0))

    assert packet.minutes_to_close is None
    assert packet.minutes_since_open is None


def test_the_close_time_may_arrive_in_any_timezone():
    """17:00Z is 13:00 New York, the same early close."""
    as_utc = et(13, 0).astimezone(ZoneInfo("UTC"))
    packet = build(observed_at=et(12, 0, 5), session_close_at=as_utc)

    assert packet.minutes_to_close == pytest.approx(59 + 55 / 60)


# --------------------------------------------------------------------------
# realized_vol_30m needs thirty consecutive one-minute intervals
# --------------------------------------------------------------------------


def gapped_session(missing, end=(10, 1)):
    """Session bars from 09:30 to ``end`` with ``missing`` absent, as a feed gap."""
    stamps = session_minutes(end=end)
    stamps.remove(missing)
    return stamps, [bar(s.hour, s.minute, 100.0 + index) for index, s in enumerate(stamps)]


def test_a_gap_inside_the_volatility_window_yields_null():
    """A missing minute would put a two-minute move in the sum as a one-minute one."""
    stamps, bars = gapped_session(et(9, 45))
    assert len(stamps) == 31  # the right count, the wrong spacing

    packet = build(minute_bars=bars, observed_at=et(10, 2, 5))

    assert packet.realized_vol_30m is None


def test_a_gap_at_the_leading_edge_of_the_volatility_window_yields_null():
    stamps, bars = gapped_session(et(9, 31))
    assert len(stamps) == 31

    packet = build(minute_bars=bars, observed_at=et(10, 2, 5))

    assert packet.realized_vol_30m is None


def test_a_gap_at_the_trailing_edge_of_the_volatility_window_yields_null():
    stamps, bars = gapped_session(et(10, 0))
    assert len(stamps) == 31

    packet = build(minute_bars=bars, observed_at=et(10, 2, 5))

    assert packet.realized_vol_30m is None


def test_a_gap_older_than_the_volatility_window_does_not_matter():
    """Only the thirty-one closes actually used have to be consecutive."""
    stamps = session_minutes(end=(10, 5))
    stamps.remove(et(9, 32))  # three minutes before the window begins
    bars = [
        bar(s.hour, s.minute, 110.0 if s == stamps[-1] else 100.0) for s in stamps
    ]

    packet = build(minute_bars=bars, observed_at=et(10, 6, 5))

    # 09:35..10:05 are consecutive, so the closed form still holds.
    assert packet.realized_vol_30m == pytest.approx(math.log(1.1))


def test_an_invalid_close_inside_the_window_leaves_a_gap_and_yields_null():
    """Dropping a bad close leaves a hole, and a hole is a gap."""
    bars = main_bars()
    bars[-16] = bar(10, 20, 0.0)  # inside the last thirty-one closes

    assert build(minute_bars=bars).realized_vol_30m is None


def test_a_consecutive_window_is_still_measured_exactly():
    """The spacing check must not disturb the ordinary case."""
    assert build().realized_vol_30m == pytest.approx(math.log(1.1))


# --------------------------------------------------------------------------
# an early close really ends the session for every feature
# --------------------------------------------------------------------------


def half_day_bars():
    """09:30..12:59 as a real half-day session, then post-close bars 13:00..13:30.

    Closes are 100.0 except 09:30 opening at 200.0, 11:59 at 88.0 (sixty
    minutes before the last in-session bar) and 12:59 at 110.0. Every bar from
    13:00 onward is 999.0, so if any of them leaked into a feature the value
    would be unmistakable.
    """
    bars = []
    for stamp in session_minutes(end=(12, 59)):
        hm = (stamp.hour, stamp.minute)
        close = {(11, 59): 88.0, (12, 59): 110.0}.get(hm, 100.0)
        open_ = 200.0 if hm == (9, 30) else close
        bars.append(bar(stamp.hour, stamp.minute, close, open_=open_))

    # Contiguous, so a broken filter would produce numbers rather than nulls.
    bars += [bar(s.hour, s.minute, 999.0) for s in session_minutes(start=(13, 0), end=(13, 30))]
    return bars


def test_bars_at_or_after_an_early_close_never_enter_a_feature():
    """Market open, this session closes 13:00, and the feed hands us 13:00-13:30."""
    packet = build(
        minute_bars=half_day_bars(),
        observed_at=et(13, 31, 5),  # late enough that every 999.0 bar is complete
        market_is_open=True,
        session_close_at=et(13, 0),
    )

    # The last in-session bar is 12:59 at 110.0, not 13:30 at 999.0.
    assert packet.return_15m == pytest.approx(110.0 / 100.0 - 1)  # 12:59 over 12:44
    assert packet.return_60m == pytest.approx(110.0 / 88.0 - 1)  # 12:59 over 11:59
    assert packet.return_since_open == pytest.approx(110.0 / 200.0 - 1)
    # 12:29..12:58 at 100.0 then 12:59 at 110.0: the same closed form as always.
    assert packet.realized_vol_30m == pytest.approx(math.log(1.1))
    # Age is measured from 12:59:00, so 32 minutes and 5 seconds.
    assert packet.bar_age_seconds == pytest.approx(32 * 60 + 5)


def test_a_broken_early_close_filter_would_be_visible():
    """The same bars with a 16:00 close: proof the fixture discriminates.

    Every assertion above flips, so the test cannot be passing by accident.
    """
    leaked = build(
        minute_bars=half_day_bars(),
        observed_at=et(13, 31, 5),
        market_is_open=True,
        session_close_at=et(16, 0),
    )

    assert leaked.return_15m == pytest.approx(0.0)  # 999.0 over 999.0
    assert leaked.return_since_open == pytest.approx(999.0 / 200.0 - 1)
    assert leaked.realized_vol_30m == pytest.approx(0.0)
    assert leaked.bar_age_seconds == pytest.approx(65.0)  # the 13:30 bar


def test_a_normal_close_of_1600_changes_nothing():
    """The clock says 16:00 on an ordinary day, which is what was assumed before."""
    packet = build(session_close_at=et(16, 0))

    assert packet.return_15m == pytest.approx(0.10)
    assert packet.return_60m == pytest.approx(0.25)
    assert packet.realized_vol_30m == pytest.approx(math.log(1.1))


def test_an_unusable_close_time_falls_back_to_the_regular_close():
    """A next-day close cannot bound today, so the 16:00 regular close stands in."""
    tomorrow = et(16, 0, day=date(2026, 8, 25))

    assert build(session_close_at=tomorrow).return_15m == pytest.approx(0.10)
    assert build(session_close_at=None).return_15m == pytest.approx(0.10)
