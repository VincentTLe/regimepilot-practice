"""History tests: the Alpaca boundary of Phase 2B.

Every client is a fake, so no network call is made and no real credential is
touched. What is under test here is the *request* we send and the normalization
we do to the reply -- not arithmetic, which lives in test_features.py.
"""

import traceback
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest
from alpaca.data.enums import DataFeed
from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest

from regimepilot import history as history_module
from regimepilot.features import DATA_FEED_IEX, FeaturePacket
from regimepilot.history import (
    HistoryError,
    fetch_latest_quote,
    fetch_market_clock,
    fetch_recent_daily_bars,
    fetch_session_minute_bars,
    observe_features,
)
from regimepilot.models import OhlcvBar

NY = ZoneInfo("America/New_York")

SESSION = date(2026, 8, 24)
PREVIOUS_SESSION = date(2026, 8, 21)

# 10:36:05 New York, i.e. 14:36:05Z. Its session is 13:30Z -> 20:00Z.
OBSERVED_AT = datetime(2026, 8, 24, 10, 36, 5, tzinfo=NY)
SESSION_OPEN_UTC = datetime(2026, 8, 24, 13, 30, tzinfo=timezone.utc)
SESSION_CLOSE_UTC = datetime(2026, 8, 24, 20, 0, tzinfo=timezone.utc)


def stored(moment):
    """How alpaca-py holds a request timestamp.

    ``BaseTimeseriesDataRequest.__init__`` converts a timezone-aware value to
    UTC and then drops the tzinfo, so the instant survives but the attribute
    reads back naive. Asserting against the raw attribute without this would be
    asserting against the SDK's storage habit rather than against our request.
    """
    return moment.astimezone(timezone.utc).replace(tzinfo=None)

API_KEY = "SUPER-SECRET-KEY"
SECRET_KEY = "SUPER-SECRET-SECRET"


def et(hour, minute, second=0, day=SESSION):
    return datetime(day.year, day.month, day.day, hour, minute, second, tzinfo=NY)


class FakeBar:
    """Mimics an alpaca Bar: prices as floats, timestamp at the start of the minute."""

    def __init__(self, timestamp, close, open_=None):
        self.symbol = "SPY"
        self.timestamp = timestamp
        self.open = close if open_ is None else open_
        self.high = close
        self.low = close
        self.close = close
        self.volume = 1000.0
        # Fields Phase 2B must not carry into the packet.
        self.trade_count = 42.0
        self.vwap = close


class FakeBarSet:
    def __init__(self, bars):
        self.data = {"SPY": list(bars)}


class FakeQuote:
    def __init__(self, bid=99.90, ask=100.10):
        self.symbol = "SPY"
        self.bid_price = bid
        self.ask_price = ask
        self.bid_size = 5.0
        self.ask_size = 7.0
        self.timestamp = et(10, 36, 4)


class FakeClock:
    def __init__(self, is_open=True, next_close=None):
        self.is_open = is_open
        self.timestamp = OBSERVED_AT
        self.next_open = et(9, 30, day=date(2026, 8, 25))
        self.next_close = et(16, 0) if next_close is None else next_close


def session_bars():
    """09:30..09:45 inclusive. 09:30 opens at 200.0; 09:45 closes at 110.0."""
    bars = []
    stamp = et(9, 30)
    while stamp <= et(9, 45):
        close = 110.0 if stamp == et(9, 45) else 100.0
        open_ = 200.0 if stamp == et(9, 30) else close
        bars.append(FakeBar(stamp, close, open_))
        stamp += timedelta(minutes=1)
    return bars


DAILY_BARS = [
    FakeBar(datetime(2026, 8, 20, 4, 0, tzinfo=timezone.utc), 240.0),
    FakeBar(datetime(2026, 8, 21, 4, 0, tzinfo=timezone.utc), 250.0),
]


class FakeDataClient:
    """Records every request it is handed, and reaches no network."""

    def __init__(self, *, minute_bars=..., daily_bars=..., quote=...):
        self._minute_bars = session_bars() if minute_bars is ... else minute_bars
        self._daily_bars = DAILY_BARS if daily_bars is ... else daily_bars
        self._quote = FakeQuote() if quote is ... else quote
        self.bar_requests = []
        self.quote_requests = []
        # Deliberately carries credentials so the leak tests are meaningful.
        self.api_key = API_KEY
        self.secret_key = SECRET_KEY

    def get_stock_bars(self, request):
        self.bar_requests.append(request)
        daily = str(request.timeframe.value).endswith("Day")
        return FakeBarSet(self._daily_bars if daily else self._minute_bars)

    def get_stock_latest_quote(self, request):
        self.quote_requests.append(request)
        return {} if self._quote is None else {"SPY": self._quote}


class FakeTradingClient:
    """Only a clock. Every mutating method explodes if it is ever reached."""

    def __init__(self, is_open=True, next_close=None):
        self._is_open = is_open
        self._next_close = next_close
        self.api_key = API_KEY
        self.secret_key = SECRET_KEY

    def get_clock(self):
        return FakeClock(self._is_open, self._next_close)

    def __getattr__(self, name):
        raise AssertionError(f"Phase 2B must not call trading_client.{name}")


def minute_request(data):
    return next(r for r in data.bar_requests if not str(r.timeframe.value).endswith("Day"))


def daily_request(data):
    return next(r for r in data.bar_requests if str(r.timeframe.value).endswith("Day"))


# --------------------------------------------------------------------------
# 22. the request explicitly names IEX
# --------------------------------------------------------------------------


def test_minute_bar_request_explicitly_uses_the_iex_feed():
    data = FakeDataClient()
    fetch_session_minute_bars(data, observed_at=OBSERVED_AT)

    request = data.bar_requests[0]
    assert isinstance(request, StockBarsRequest)
    assert request.feed is DataFeed.IEX
    assert request.feed is not None  # never left to the account default


def test_daily_bar_request_explicitly_uses_the_iex_feed():
    data = FakeDataClient()
    fetch_recent_daily_bars(data, observed_at=OBSERVED_AT)

    assert data.bar_requests[0].feed is DataFeed.IEX


def test_quote_request_explicitly_uses_the_iex_feed():
    data = FakeDataClient()
    fetch_latest_quote(data)

    request = data.quote_requests[0]
    assert isinstance(request, StockLatestQuoteRequest)
    assert request.feed is DataFeed.IEX


def test_every_request_of_a_full_observation_names_iex():
    data = FakeDataClient()
    observe_features(FakeTradingClient(), data, now=OBSERVED_AT)

    assert len(data.bar_requests) == 2
    for request in data.bar_requests + data.quote_requests:
        assert request.feed is DataFeed.IEX


def test_the_feed_string_in_the_packet_matches_the_sdk_enum():
    """features.py cannot import the SDK, so this is where the two are tied."""
    assert DATA_FEED_IEX == DataFeed.IEX.value == "iex"


# --------------------------------------------------------------------------
# the shape of the historical request
# --------------------------------------------------------------------------


def test_minute_request_asks_for_one_minute_spy_bars_of_this_session_only():
    data = FakeDataClient()
    fetch_session_minute_bars(data, observed_at=OBSERVED_AT)

    request = data.bar_requests[0]
    assert request.symbol_or_symbols in ("SPY", ["SPY"])
    assert request.timeframe.value == "1Min"
    # Exactly this session's regular hours, so the window cannot reach the
    # previous session even before features.py filters.
    assert request.start == stored(SESSION_OPEN_UTC)
    assert request.end == stored(SESSION_CLOSE_UTC)


def test_the_minute_window_is_sent_over_the_wire_as_an_explicit_utc_instant():
    """What matters is the instant Alpaca receives, not how the SDK stores it."""
    data = FakeDataClient()
    fetch_session_minute_bars(data, observed_at=OBSERVED_AT)

    fields = data.bar_requests[0].to_request_fields()
    assert fields["start"] == "2026-08-24T13:30:00+00:00"  # 09:30 New York
    assert fields["end"] == "2026-08-24T20:00:00+00:00"  # 16:00 New York
    assert fields["symbols"] == "SPY"


def test_the_minute_window_follows_the_session_not_the_calendar_day():
    """An observation after the close still belongs to that day's session."""
    data = FakeDataClient()
    fetch_session_minute_bars(data, observed_at=et(23, 30))

    assert data.bar_requests[0].start == stored(SESSION_OPEN_UTC)
    assert data.bar_requests[0].end == stored(SESSION_CLOSE_UTC)


def test_daily_request_looks_back_far_enough_to_clear_a_long_weekend():
    data = FakeDataClient()
    fetch_recent_daily_bars(data, observed_at=OBSERVED_AT)

    request = data.bar_requests[0]
    assert request.timeframe.value == "1Day"
    assert request.end == stored(OBSERVED_AT)
    assert request.start <= stored(OBSERVED_AT) - timedelta(days=5)


# --------------------------------------------------------------------------
# normalization: nothing from the SDK survives
# --------------------------------------------------------------------------


def test_minute_bars_come_back_as_our_own_normalized_model():
    bars = fetch_session_minute_bars(FakeDataClient(), observed_at=OBSERVED_AT)

    assert len(bars) == 16
    assert all(isinstance(bar, OhlcvBar) for bar in bars)
    assert bars[0].open == pytest.approx(200.0)
    assert bars[-1].close == pytest.approx(110.0)
    # Timestamps are normalized to UTC.
    assert bars[0].timestamp == SESSION_OPEN_UTC
    # trade_count and vwap exist upstream but have nowhere to live here.
    assert set(OhlcvBar.model_fields) == {
        "timestamp", "open", "high", "low", "close", "volume"
    }


def test_a_quote_comes_back_as_two_plain_floats():
    bid, ask = fetch_latest_quote(FakeDataClient())

    assert (bid, ask) == (pytest.approx(99.90), pytest.approx(100.10))
    assert type(bid) is float and type(ask) is float


def test_a_missing_quote_is_two_nulls_not_an_error():
    assert fetch_latest_quote(FakeDataClient(quote=None)) == (None, None)


def test_an_empty_bar_reply_is_an_empty_list_not_an_error():
    data = FakeDataClient(minute_bars=[], daily_bars=[])

    assert fetch_session_minute_bars(data, observed_at=OBSERVED_AT) == []
    assert fetch_recent_daily_bars(data, observed_at=OBSERVED_AT) == []


def test_market_state_is_read_from_the_clock():
    assert fetch_market_clock(FakeTradingClient(is_open=True)).is_open is True
    assert fetch_market_clock(FakeTradingClient(is_open=False)).is_open is False


def test_one_clock_request_carries_both_openness_and_the_close_time():
    """No second API call: next_close rides along with is_open."""
    clock = fetch_market_clock(FakeTradingClient(is_open=True, next_close=et(13, 0)))

    assert clock.is_open is True
    assert clock.next_close == datetime(2026, 8, 24, 17, 0, tzinfo=timezone.utc)


def test_the_packet_takes_minutes_to_close_from_the_clocks_early_close():
    """A half day: the clock says 13:00, so nothing may report 16:00."""
    trading = FakeTradingClient(is_open=True, next_close=et(13, 0))
    packet = observe_features(trading, FakeDataClient(), now=et(12, 0, 5))

    # 12:00:05 -> 13:00:00 is 59 minutes and 55 seconds, not 239:55.
    assert packet.minutes_to_close == pytest.approx(59 + 55 / 60)


def test_the_packet_takes_minutes_to_close_from_a_normal_close_too():
    trading = FakeTradingClient(is_open=True)  # next_close 16:00
    packet = observe_features(trading, FakeDataClient(), now=et(12, 0, 5))

    assert packet.minutes_to_close == pytest.approx(239 + 55 / 60)


def test_a_clock_without_a_close_time_leaves_the_session_minutes_null():
    trading = FakeTradingClient(is_open=True, next_close="not a datetime")
    packet = observe_features(trading, FakeDataClient(), now=et(12, 0, 5))

    assert packet.minutes_to_close is None
    assert packet.minutes_since_open is None


# --------------------------------------------------------------------------
# 19. an API failure raises, and fabricates nothing
# --------------------------------------------------------------------------


class Boom(RuntimeError):
    """An upstream error that quotes the outbound request, as HTTP clients do."""

    def __init__(self):
        super().__init__(f"401 unauthorized for key={API_KEY} secret={SECRET_KEY}")


@pytest.mark.parametrize(
    "client, method",
    [
        ("data", "get_stock_bars"),
        ("data", "get_stock_latest_quote"),
        ("trading", "get_clock"),
    ],
)
def test_an_api_failure_raises_history_error_and_fabricates_nothing(client, method):
    trading, data = FakeTradingClient(), FakeDataClient()
    target = trading if client == "trading" else data

    def explode(*args, **kwargs):
        raise Boom()

    object.__setattr__(target, method, explode)

    with pytest.raises(HistoryError):
        observe_features(trading, data, now=OBSERVED_AT)


def test_a_bar_failure_never_becomes_an_empty_but_successful_packet():
    """The difference that matters: no data is null, a broken call is an error."""
    data = FakeDataClient()

    def explode(request):
        raise Boom()

    data.get_stock_bars = explode

    with pytest.raises(HistoryError):
        fetch_session_minute_bars(data, observed_at=OBSERVED_AT)


def test_the_history_error_names_the_step_without_leaking_credentials():
    data = FakeDataClient()

    def explode(request):
        raise Boom()

    data.get_stock_bars = explode

    with pytest.raises(HistoryError) as caught:
        fetch_session_minute_bars(data, observed_at=OBSERVED_AT)

    message = str(caught.value)
    assert "minute bars" in message
    assert "Boom" in message  # the exception type is useful and safe

    rendered = "".join(
        traceback.format_exception(type(caught.value), caught.value, caught.value.__traceback__)
    )
    for blob in (message, rendered):
        assert API_KEY not in blob
        assert SECRET_KEY not in blob


# --------------------------------------------------------------------------
# end to end, still without a network
# --------------------------------------------------------------------------


def test_a_full_observation_produces_a_feature_packet_with_hand_checked_values():
    packet = observe_features(FakeTradingClient(), FakeDataClient(), now=OBSERVED_AT)

    assert isinstance(packet, FeaturePacket)
    assert packet.symbol == "SPY"
    assert packet.data_feed == "iex"
    assert packet.market_is_open is True
    # 09:45 closes at 110.0, 09:30 closes at 100.0 -> exactly +10%.
    assert packet.return_15m == pytest.approx(110.0 / 100.0 - 1)
    # The 09:30 bar opens at 200.0.
    assert packet.return_since_open == pytest.approx(110.0 / 200.0 - 1)
    # 200.0 against Friday's 250.0 close.
    assert packet.overnight_gap_pct == pytest.approx(200.0 / 250.0 - 1)
    # Sixteen bars is not sixty minutes, and not thirty-one closes.
    assert packet.return_60m is None
    assert packet.realized_vol_30m is None
    # mid 100.00, spread 0.20.
    assert packet.spread_bps == pytest.approx(20.0)


def test_an_observation_with_no_session_data_is_all_null_not_an_error():
    """Observing before 09:30, which is exactly the closed-market case."""
    data = FakeDataClient(minute_bars=[], daily_bars=[], quote=None)
    packet = observe_features(FakeTradingClient(is_open=False), data, now=et(4, 42))

    assert isinstance(packet, FeaturePacket)
    for field in ("return_15m", "return_60m", "return_since_open",
                  "overnight_gap_pct", "realized_vol_30m",
                  "spread_bps", "bar_age_seconds",
                  "minutes_since_open", "minutes_to_close"):
        assert getattr(packet, field) is None, field
    assert packet.market_is_open is False


def test_the_packet_serializes_without_any_sdk_object_or_credential():
    data = FakeDataClient()
    packet = observe_features(FakeTradingClient(), data, now=OBSERVED_AT)

    serialized = packet.model_dump_json()
    for blob in (serialized, history_module.format_summary(packet)):
        assert API_KEY not in blob
        assert SECRET_KEY not in blob
    assert FeaturePacket.model_validate_json(serialized) == packet


# --------------------------------------------------------------------------
# 23. read-only
# --------------------------------------------------------------------------


def test_the_history_module_exposes_no_trading_or_execution_helper():
    forbidden = (
        "submit", "cancel", "replace", "close_position", "close_all",
        "exercise", "order", "buy_call", "buy_put", "place_",
    )
    offenders = [
        name for name in dir(history_module) if any(word in name.lower() for word in forbidden)
    ]
    assert offenders == []


def test_a_full_observation_touches_no_mutating_trading_method():
    """FakeTradingClient raises on any attribute but get_clock."""
    packet = observe_features(FakeTradingClient(), FakeDataClient(), now=OBSERVED_AT)

    assert isinstance(packet, FeaturePacket)


def test_the_history_module_names_no_execution_endpoint():
    from pathlib import Path

    source = Path(history_module.__file__).read_text(encoding="utf-8").lower()
    for word in ("submit_order", "cancel_order", "replace_order", "close_position",
                 "exercise_option", "orderrequest", "orderside"):
        assert word not in source
