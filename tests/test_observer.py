"""Observer tests. Every Alpaca client is replaced with a fake, so no network
call is ever made and no real credential is ever touched."""

import json
import traceback
from datetime import date, datetime, timezone
from enum import Enum

import pytest

from regimepilot.models import (
    AccountSnapshot,
    ObservationPacket,
    OhlcvBar,
    OptionContractSummary,
)
from regimepilot.observer import (
    MAX_OPTION_CONTRACT_PAGES,
    ObserverError,
    format_summary,
    normalize_contract,
    normalize_underlying,
    observe,
)

NOW = datetime(2026, 8, 25, 14, 30, tzinfo=timezone.utc)

API_KEY = "SUPER-SECRET-KEY"
SECRET_KEY = "SUPER-SECRET-SECRET"
ACCOUNT_ID = "11112222-3333-4444-5555-666677778888"


class FakeContractType(Enum):
    """Mimics alpaca's ContractType, which normalization must unwrap."""

    CALL = "call"
    PUT = "put"


class FakeAssetStatus(Enum):
    ACTIVE = "active"


class FakeClock:
    def __init__(self, is_open=True):
        self.is_open = is_open
        self.next_open = datetime(2026, 8, 26, 13, 30, tzinfo=timezone.utc)
        self.next_close = datetime(2026, 8, 25, 20, 0, tzinfo=timezone.utc)


class FakeAccount:
    """Alpaca returns account money as strings, so these are strings too."""

    def __init__(self, account_id=ACCOUNT_ID):
        self.id = account_id
        self.equity = "100000.55"
        self.cash = "99000.25"
        self.buying_power = "200000.10"
        self.options_buying_power = "98000.75"
        self.options_trading_level = 2


class FakeTrade:
    def __init__(self, price=643.21):
        self.price = price
        self.timestamp = datetime(2026, 8, 25, 14, 29, 58, tzinfo=timezone.utc)


class FakeQuote:
    def __init__(self, bid=643.10, ask=643.25):
        self.bid_price = bid
        self.ask_price = ask
        self.timestamp = datetime(2026, 8, 25, 14, 29, 59, tzinfo=timezone.utc)


class FakeBar:
    def __init__(self, timestamp, close=643.0, volume=1_000_000):
        self.timestamp = timestamp
        self.open = close - 1
        self.high = close + 2
        self.low = close - 3
        self.close = close
        self.volume = volume


class FakeBarSet:
    def __init__(self, bars):
        self.data = {"SPY": list(bars)}


class FakeContract:
    def __init__(
        self,
        symbol="SPY260828C00640000",
        contract_type=FakeContractType.CALL,
        strike="640",
        expiration=date(2026, 8, 28),
    ):
        self.symbol = symbol
        self.type = contract_type
        self.strike_price = strike
        self.expiration_date = expiration
        self.status = FakeAssetStatus.ACTIVE
        self.tradable = True
        # Fields Phase 2A must NOT copy into the packet.
        self.open_interest = "12345"
        self.close_price = "3.40"


class FakeContractsResponse:
    def __init__(self, contracts, next_page_token=None):
        self.option_contracts = list(contracts)
        self.next_page_token = next_page_token


DAILY_BARS = [
    FakeBar(datetime(2026, 8, 22, 4, 0, tzinfo=timezone.utc), close=640.0),
    FakeBar(datetime(2026, 8, 21, 4, 0, tzinfo=timezone.utc), close=638.0),
    FakeBar(datetime(2026, 8, 25, 4, 0, tzinfo=timezone.utc), close=643.0),
]


class FakeTradingClient:
    """Stands in for alpaca TradingClient. Records requests, hits no network."""

    def __init__(self, *, contracts=None, pages=None, account_id=ACCOUNT_ID, is_open=True):
        self._is_open = is_open
        self._account_id = account_id
        # ``pages`` models a paginated reply; ``contracts`` a single page.
        self._pages = pages if pages is not None else [(list(contracts or []), None)]
        self.option_requests = []
        # Deliberately carries credentials so the leak tests below are meaningful.
        self.api_key = API_KEY
        self.secret_key = SECRET_KEY

    def get_clock(self):
        return FakeClock(self._is_open)

    def get_account(self):
        return FakeAccount(self._account_id)

    def get_option_contracts(self, request):
        self.option_requests.append(request)
        contracts, token = self._pages[len(self.option_requests) - 1]
        return FakeContractsResponse(contracts, token)


class FakeDataClient:
    """``None`` for trade/quote/minute_bar models a feed with nothing to say."""

    def __init__(self, *, trade=..., quote=..., minute_bar=..., daily_bars=...):
        self._trade = FakeTrade() if trade is ... else trade
        self._quote = FakeQuote() if quote is ... else quote
        self._minute_bar = (
            FakeBar(datetime(2026, 8, 25, 14, 29, tzinfo=timezone.utc))
            if minute_bar is ...
            else minute_bar
        )
        self._daily_bars = DAILY_BARS if daily_bars is ... else daily_bars

    @staticmethod
    def _wrap(value):
        return {} if value is None else {"SPY": value}

    def get_stock_latest_trade(self, request):
        return self._wrap(self._trade)

    def get_stock_latest_quote(self, request):
        return self._wrap(self._quote)

    def get_stock_latest_bar(self, request):
        return self._wrap(self._minute_bar)

    def get_stock_bars(self, request):
        return FakeBarSet(self._daily_bars)


def observe_with(trading=None, data=None):
    return observe(trading or FakeTradingClient(), data or FakeDataClient(), now=NOW)


# --------------------------------------------------------------------------
# 1. successful full normalization
# --------------------------------------------------------------------------


def test_successful_full_normalization():
    trading = FakeTradingClient(
        contracts=[
            FakeContract(expiration=date(2026, 9, 4)),
            FakeContract(expiration=date(2026, 8, 28)),
            FakeContract(expiration=date(2026, 9, 1)),
        ]
    )
    packet = observe_with(trading)

    assert isinstance(packet, ObservationPacket)
    assert packet.observed_at == NOW

    assert packet.market.is_open is True
    assert packet.market.next_open == datetime(2026, 8, 26, 13, 30, tzinfo=timezone.utc)
    assert packet.market.next_close == datetime(2026, 8, 25, 20, 0, tzinfo=timezone.utc)

    assert packet.account.account_id_masked == "****8888"
    assert packet.account.equity == pytest.approx(100000.55)
    assert packet.account.cash == pytest.approx(99000.25)
    assert packet.account.buying_power == pytest.approx(200000.10)
    assert packet.account.options_buying_power == pytest.approx(98000.75)
    assert packet.account.options_trading_level == 2

    underlying = packet.underlying
    assert underlying.symbol == "SPY"
    assert underlying.latest_trade_price == pytest.approx(643.21)
    assert underlying.latest_trade_timestamp is not None
    assert underlying.bid_price == pytest.approx(643.10)
    assert underlying.ask_price == pytest.approx(643.25)
    assert underlying.quote_timestamp is not None

    # Newest daily bar wins regardless of the order the feed returned them.
    assert underlying.daily_bar.close == pytest.approx(643.0)
    assert underlying.previous_daily_bar.close == pytest.approx(640.0)
    assert underlying.minute_bar.timestamp == datetime(
        2026, 8, 25, 14, 29, tzinfo=timezone.utc
    )

    assert packet.option_universe.contract_count == 3
    assert packet.option_universe.earliest_expiration == date(2026, 8, 28)
    assert packet.option_universe.latest_expiration == date(2026, 9, 4)


def test_packet_holds_only_our_own_models():
    """No Alpaca object may survive the observer."""
    packet = observe_with(FakeTradingClient(contracts=[FakeContract()]))

    assert isinstance(packet.account, AccountSnapshot)
    assert isinstance(packet.underlying.minute_bar, OhlcvBar)
    assert all(
        isinstance(contract, OptionContractSummary)
        for contract in packet.option_universe.contracts
    )
    # Phase 2A observes; it does not score. Nothing may creep in later either.
    assert set(OptionContractSummary.model_fields) == {
        "symbol",
        "option_type",
        "strike_price",
        "expiration_date",
        "status",
        "tradable",
    }


def test_naive_and_offset_timestamps_are_normalized_to_utc():
    packet = observe(
        FakeTradingClient(),
        FakeDataClient(),
        now=datetime(2026, 8, 25, 9, 30),  # naive; treated as UTC
    )
    assert packet.observed_at == datetime(2026, 8, 25, 9, 30, tzinfo=timezone.utc)


# --------------------------------------------------------------------------
# 2-5. a successful call that carries no data yields null, never a guess
# --------------------------------------------------------------------------


def test_missing_stock_trade():
    packet = observe_with(data=FakeDataClient(trade=None))

    assert packet.underlying.latest_trade_price is None
    assert packet.underlying.latest_trade_timestamp is None
    # Everything else still observed.
    assert packet.underlying.bid_price == pytest.approx(643.10)
    assert packet.underlying.minute_bar is not None


def test_missing_stock_quote():
    packet = observe_with(data=FakeDataClient(quote=None))

    assert packet.underlying.bid_price is None
    assert packet.underlying.ask_price is None
    assert packet.underlying.quote_timestamp is None
    assert packet.underlying.latest_trade_price == pytest.approx(643.21)


def test_missing_minute_bar():
    packet = observe_with(data=FakeDataClient(minute_bar=None))

    assert packet.underlying.minute_bar is None
    assert packet.underlying.daily_bar is not None


def test_missing_daily_bar():
    packet = observe_with(data=FakeDataClient(daily_bars=[]))

    assert packet.underlying.daily_bar is None
    assert packet.underlying.previous_daily_bar is None
    assert packet.underlying.minute_bar is not None


def test_single_daily_bar_leaves_previous_null():
    only_one = [FakeBar(datetime(2026, 8, 25, 4, 0, tzinfo=timezone.utc))]
    packet = observe_with(data=FakeDataClient(daily_bars=only_one))

    assert packet.underlying.daily_bar is not None
    assert packet.underlying.previous_daily_bar is None


def test_everything_missing_still_produces_a_valid_packet():
    data = FakeDataClient(trade=None, quote=None, minute_bar=None, daily_bars=[])
    packet = observe_with(data=data)

    assert isinstance(packet, ObservationPacket)
    assert packet.underlying.latest_trade_price is None
    assert packet.underlying.daily_bar is None
    assert packet.option_universe.contract_count == 0


def test_bar_with_missing_fields_nulls_only_those_fields():
    bar = FakeBar(datetime(2026, 8, 25, 4, 0, tzinfo=timezone.utc))
    bar.volume = None
    bar.high = None

    snapshot = normalize_underlying(daily_bars=[bar])

    assert snapshot.daily_bar.volume is None
    assert snapshot.daily_bar.high is None
    assert snapshot.daily_bar.close == pytest.approx(643.0)


# --------------------------------------------------------------------------
# 6-7. option contracts
# --------------------------------------------------------------------------


def test_empty_option_contract_list_is_a_valid_universe():
    packet = observe_with(FakeTradingClient(contracts=[]))

    assert packet.option_universe.contract_count == 0
    assert packet.option_universe.earliest_expiration is None
    assert packet.option_universe.latest_expiration is None
    assert packet.option_universe.contracts == ()
    # An empty universe is not an error.
    assert isinstance(packet, ObservationPacket)


def test_option_contract_normalization():
    contract = normalize_contract(
        FakeContract(
            symbol="SPY260828P00635000",
            contract_type=FakeContractType.PUT,
            strike="635.5",
            expiration=date(2026, 8, 28),
        )
    )

    assert contract.symbol == "SPY260828P00635000"
    assert contract.option_type == "put"  # enum unwrapped to its value
    assert contract.strike_price == pytest.approx(635.5)  # string coerced
    assert contract.expiration_date == date(2026, 8, 28)
    assert contract.status == "active"
    assert contract.tradable is True

    # Pricing and interest data exist upstream but must not be carried over.
    dumped = contract.model_dump()
    assert "open_interest" not in dumped
    assert "close_price" not in dumped


def test_option_contract_accepts_a_string_expiration():
    contract = normalize_contract(FakeContract(expiration="2026-09-04"))
    assert contract.expiration_date == date(2026, 9, 4)


def test_option_contracts_are_paged_to_exhaustion():
    """A truncated page would make contract_count quietly wrong."""
    trading = FakeTradingClient(
        pages=[
            ([FakeContract(expiration=date(2026, 8, 28))], "token-2"),
            ([FakeContract(expiration=date(2026, 9, 4))], None),
        ]
    )
    packet = observe_with(trading)

    assert len(trading.option_requests) == 2
    assert trading.option_requests[0].page_token is None
    assert trading.option_requests[1].page_token == "token-2"
    assert packet.option_universe.contract_count == 2
    assert packet.option_universe.earliest_expiration == date(2026, 8, 28)
    assert packet.option_universe.latest_expiration == date(2026, 9, 4)


def test_option_request_uses_the_phase_one_expiration_window():
    trading = FakeTradingClient()
    observe_with(trading)

    request = trading.option_requests[0]
    assert list(request.underlying_symbols) == ["SPY"]
    assert request.expiration_date_gte == date(2026, 8, 28)
    assert request.expiration_date_lte == date(2026, 9, 8)


# --------------------------------------------------------------------------
# 8-9. the account id and the credentials
# --------------------------------------------------------------------------


def test_account_id_remains_masked():
    packet = observe_with(FakeTradingClient(account_id=ACCOUNT_ID))

    assert packet.account.account_id_masked == "****8888"
    assert ACCOUNT_ID not in packet.model_dump_json()
    # The unmasked id has nowhere to live on our model.
    assert "account_id" not in AccountSnapshot.model_fields


def test_secrets_never_appear_in_the_serialized_packet_or_summary():
    trading = FakeTradingClient(contracts=[FakeContract()])
    packet = observe_with(trading)

    serialized = packet.model_dump_json()
    summary = format_summary(packet)

    for blob in (serialized, json.dumps(packet.model_dump(mode="json")), summary):
        assert API_KEY not in blob
        assert SECRET_KEY not in blob
        assert ACCOUNT_ID not in blob

    assert "****8888" in serialized  # only the masked tail survives


def test_summary_stays_compact_and_hides_the_contract_list():
    contracts = [FakeContract(symbol=f"SPY260828C0064{i:04d}") for i in range(50)]
    packet = observe_with(FakeTradingClient(contracts=contracts))

    summary = format_summary(packet)

    assert len(summary.splitlines()) <= 12
    assert "50 contracts" in summary
    for contract in packet.option_universe.contracts:
        assert contract.symbol not in summary
    # ...but the full list is still there programmatically.
    assert len(packet.option_universe.contracts) == 50


# --------------------------------------------------------------------------
# 10. an API failure produces no observation at all
# --------------------------------------------------------------------------


class Boom(RuntimeError):
    """An upstream error that quotes the outbound request, as HTTP clients do."""

    def __init__(self):
        super().__init__(f"401 unauthorized for key={API_KEY} secret={SECRET_KEY}")


@pytest.mark.parametrize(
    "client, method",
    [
        ("trading", "get_clock"),
        ("trading", "get_account"),
        ("trading", "get_option_contracts"),
        ("data", "get_stock_latest_trade"),
        ("data", "get_stock_latest_quote"),
        ("data", "get_stock_latest_bar"),
        ("data", "get_stock_bars"),
    ],
)
def test_api_failure_raises_and_fabricates_nothing(client, method):
    trading, data = FakeTradingClient(), FakeDataClient()
    target = trading if client == "trading" else data

    def explode(*args, **kwargs):
        raise Boom()

    setattr(target, method, explode)

    with pytest.raises(ObserverError) as caught:
        observe(trading, data, now=NOW)

    # No packet, no partial packet, no invented value.
    assert caught.value.args and isinstance(caught.value.args[0], str)


def test_api_failure_message_names_the_step_without_leaking_credentials():
    trading = FakeTradingClient()
    trading.get_account = lambda: (_ for _ in ()).throw(Boom())

    with pytest.raises(ObserverError) as caught:
        observe(trading, FakeDataClient(), now=NOW)

    message = str(caught.value)
    assert "account" in message
    assert "Boom" in message  # the type is useful and safe

    # Neither the message nor anything reachable from the traceback may quote
    # the upstream text, so a printed traceback cannot leak a key.
    rendered = "".join(
        traceback.format_exception(type(caught.value), caught.value, caught.value.__traceback__)
    )
    for blob in (message, rendered):
        assert API_KEY not in blob
        assert SECRET_KEY not in blob


def test_observer_exposes_no_order_or_position_mutation():
    """Phase 2A stays read-only. Guard against an execution helper sneaking in."""
    import regimepilot.observer as module

    forbidden = ("submit", "cancel", "replace", "close_position", "close_all", "exercise")
    offenders = [name for name in dir(module) if any(word in name.lower() for word in forbidden)]
    assert offenders == []


# --------------------------------------------------------------------------
# 11. the option window is a New York calendar date, not a UTC one
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "now, earliest, latest",
    [
        # 10:30 New York on Tuesday 2026-08-25. The UTC date and the New York
        # date agree, which is the ordinary daytime case.
        (
            datetime(2026, 8, 25, 14, 30, tzinfo=timezone.utc),
            date(2026, 8, 28),
            date(2026, 9, 8),
        ),
        # 20:30 New York on the *same* Tuesday. UTC has already rolled over to
        # Wednesday; the US options calendar has not, so the window must be
        # identical to the one above.
        (
            datetime(2026, 8, 26, 0, 30, tzinfo=timezone.utc),
            date(2026, 8, 28),
            date(2026, 9, 8),
        ),
        # 23:30 New York on Tuesday 2026-01-13, in EST. Only a real timezone
        # conversion lands on the 13th here: a hardcoded -4 would read this
        # instant as the 14th and shift both bounds by a day.
        (
            datetime(2026, 1, 14, 4, 30, tzinfo=timezone.utc),
            date(2026, 1, 16),
            date(2026, 1, 27),
        ),
    ],
    ids=["midsession-edt", "after-utc-midnight-edt", "after-utc-midnight-est"],
)
def test_option_expiration_window_uses_the_new_york_calendar_date(now, earliest, latest):
    """DTE is counted from the market's date, not from whatever date UTC is on."""
    trading = FakeTradingClient()
    observe(trading, FakeDataClient(), now=now)

    request = trading.option_requests[0]
    assert request.expiration_date_gte == earliest
    assert request.expiration_date_lte == latest


# --------------------------------------------------------------------------
# 12. the observed universe cannot be mutated after the fact
# --------------------------------------------------------------------------


def test_observed_contracts_cannot_be_appended_to_or_removed_from():
    """An observation is a record. Nothing may add a contract to it later."""
    packet = observe_with(FakeTradingClient(contracts=[FakeContract()]))
    contracts = packet.option_universe.contracts

    assert not hasattr(contracts, "append")
    assert not hasattr(contracts, "remove")
    with pytest.raises(TypeError):
        contracts[0] = FakeContract()
    with pytest.raises(TypeError):
        del contracts[0]


def test_the_option_universe_is_frozen_and_closed_to_stray_fields():
    packet = observe_with(FakeTradingClient(contracts=[FakeContract()]))
    universe = packet.option_universe

    with pytest.raises(Exception):
        universe.contracts = ()
    with pytest.raises(Exception):
        universe.contract_count = 99
    with pytest.raises(Exception):
        packet.option_universe = universe


def test_the_universe_serializes_and_round_trips_with_its_contracts_intact():
    trading = FakeTradingClient(
        contracts=[
            FakeContract(symbol="SPY260828C00640000", expiration=date(2026, 8, 28)),
            FakeContract(symbol="SPY260904P00635000", expiration=date(2026, 9, 4)),
        ]
    )
    packet = observe_with(trading)

    payload = json.loads(packet.model_dump_json())
    serialized = payload["option_universe"]["contracts"]

    assert [contract["symbol"] for contract in serialized] == [
        "SPY260828C00640000",
        "SPY260904P00635000",
    ]

    restored = ObservationPacket.model_validate(payload)
    assert restored.option_universe == packet.option_universe
    assert not hasattr(restored.option_universe.contracts, "append")


def test_the_universe_count_and_bounds_agree_with_the_contracts_it_holds():
    trading = FakeTradingClient(
        contracts=[
            FakeContract(symbol="a", expiration=date(2026, 9, 4)),
            FakeContract(symbol="b", expiration=date(2026, 8, 28)),
            FakeContract(symbol="c", expiration=date(2026, 9, 1)),
        ]
    )
    universe = observe_with(trading).option_universe

    assert universe.contract_count == 3
    assert len(universe.contracts) == 3
    assert universe.earliest_expiration == date(2026, 8, 28)
    assert universe.latest_expiration == date(2026, 9, 4)


# --------------------------------------------------------------------------
# 13. a universe too large to page is an error, never a partial packet
# --------------------------------------------------------------------------


def _endless_pages(count):
    """``count`` pages, every one of which promises another after it."""
    return [([FakeContract(symbol=f"SPY-{index}")], f"token-{index + 2}") for index in range(count)]


def test_a_universe_larger_than_the_page_cap_raises_instead_of_truncating():
    trading = FakeTradingClient(pages=_endless_pages(MAX_OPTION_CONTRACT_PAGES))

    with pytest.raises(ObserverError) as caught:
        observe(trading, FakeDataClient(), now=NOW)

    # Every allowed page was read before giving up, and no packet came back.
    assert len(trading.option_requests) == MAX_OPTION_CONTRACT_PAGES

    message = str(caught.value)
    assert "option contracts" in message
    # The message must say *why* it stopped, not merely that something failed.
    assert "page" in message.lower()


def test_the_page_cap_error_leaks_no_credential_or_upstream_text():
    trading = FakeTradingClient(pages=_endless_pages(MAX_OPTION_CONTRACT_PAGES))

    with pytest.raises(ObserverError) as caught:
        observe(trading, FakeDataClient(), now=NOW)

    rendered = "".join(
        traceback.format_exception(type(caught.value), caught.value, caught.value.__traceback__)
    )
    for blob in (str(caught.value), rendered):
        assert API_KEY not in blob
        assert SECRET_KEY not in blob
        assert ACCOUNT_ID not in blob
        assert "token-" not in blob


def test_a_universe_ending_on_the_final_allowed_page_is_a_complete_observation():
    """The cap is a limit on pages read, not a limit one short of it."""
    pages = _endless_pages(MAX_OPTION_CONTRACT_PAGES - 1)
    pages.append(([FakeContract(symbol="last", expiration=date(2026, 9, 4))], None))
    trading = FakeTradingClient(pages=pages)

    packet = observe(trading, FakeDataClient(), now=NOW)

    assert len(trading.option_requests) == MAX_OPTION_CONTRACT_PAGES
    assert packet.option_universe.contract_count == MAX_OPTION_CONTRACT_PAGES
    assert packet.option_universe.contracts[-1].symbol == "last"
