"""Chain tests: the Alpaca option-data boundary of Phase 4A.

Every client is a fake, so no network call is made and no real credential is
touched. Option snapshots are the SDK's own ``OptionsSnapshot`` models built
offline from API-shaped dicts, so the join sees the exact reply shape
production sees -- a plain ``{symbol: OptionsSnapshot}`` dict.
"""

import json
import traceback
from datetime import date, datetime, timedelta, timezone
from enum import Enum

import pytest
from alpaca.data.enums import OptionsFeed
from alpaca.data.models.snapshots import OptionsSnapshot
from alpaca.trading.enums import AssetStatus, ContractType

from regimepilot import chain as chain_module
from regimepilot.chain import (
    MAX_DAYS_TO_EXPIRATION,
    MAX_OPTION_CONTRACT_PAGES,
    MIN_DAYS_TO_EXPIRATION,
    QUERY_STRIKE_WINDOW,
    ChainError,
    build_option_data_client,
    format_summary,
    main,
    observe_chain,
)
from regimepilot.config import ConfigError, settings_from_mapping
from regimepilot.features import quote_age_seconds
from regimepilot.models import ChainPacket, ContractCandidate

# 10:35 New York on Wednesday 2026-08-26.
NOW = datetime(2026, 8, 26, 14, 35, tzinfo=timezone.utc)

API_KEY = "SUPER-SECRET-KEY"
SECRET_KEY = "SUPER-SECRET-SECRET"


class FakeContractType(Enum):
    """Mimics alpaca's ContractType, which normalization must unwrap."""

    CALL = "call"
    PUT = "put"


class FakeAssetStatus(Enum):
    ACTIVE = "active"


class FakeContract:
    """One row of the /v2/options/contracts reply. Strike arrives as a string."""

    def __init__(
        self,
        symbol="SPY260901C00765000",
        contract_type=FakeContractType.CALL,
        strike="765",
        expiration=date(2026, 9, 1),
        tradable=True,
    ):
        self.symbol = symbol
        self.type = contract_type
        self.strike_price = strike
        self.expiration_date = expiration
        self.status = FakeAssetStatus.ACTIVE
        self.tradable = tradable
        # Fields Phase 4A must NOT copy into the packet.
        self.open_interest = "12345"
        self.close_price = "3.40"


class FakeContractsResponse:
    def __init__(self, contracts, next_page_token=None):
        self.option_contracts = list(contracts)
        self.next_page_token = next_page_token


class FakeClock:
    """Alpaca's market clock; ``timestamp`` is the server's own idea of now."""

    def __init__(self, timestamp=NOW, is_open=True):
        self.timestamp = timestamp
        self.is_open = is_open
        self.next_open = datetime(2026, 8, 27, 13, 30, tzinfo=timezone.utc)
        self.next_close = datetime(2026, 8, 26, 20, 0, tzinfo=timezone.utc)


class FakeTradingClient:
    """Stands in for alpaca TradingClient. Records requests, hits no network."""

    def __init__(self, *, contracts=None, pages=None, clock=...):
        self._pages = pages if pages is not None else [(list(contracts or []), None)]
        self._clock = FakeClock() if clock is ... else clock
        self.option_requests = []
        self.clock_requests = 0
        # Deliberately carries credentials so the leak tests are meaningful.
        self.api_key = API_KEY
        self.secret_key = SECRET_KEY

    def get_clock(self):
        self.clock_requests += 1
        return self._clock

    def get_option_contracts(self, request):
        self.option_requests.append(request)
        contracts, token = self._pages[len(self.option_requests) - 1]
        return FakeContractsResponse(contracts, token)


class FakeQuote:
    def __init__(self, bid=764.90, ask=765.10):
        self.symbol = "SPY"
        self.bid_price = bid
        self.ask_price = ask
        self.bid_size = 5.0
        self.ask_size = 7.0
        self.timestamp = datetime(2026, 8, 26, 14, 34, 59, tzinfo=timezone.utc)


class FakeDataClient:
    """``quote=None`` models a feed with nothing to say about SPY."""

    def __init__(self, *, quote=...):
        self._quote = FakeQuote() if quote is ... else quote
        self.quote_requests = []
        self.api_key = API_KEY
        self.secret_key = SECRET_KEY

    def get_stock_latest_quote(self, request):
        self.quote_requests.append(request)
        return {} if self._quote is None else {"SPY": self._quote}


def quote_payload(bid, ask, stamp, *, greeks=True):
    """One option snapshot exactly as the indicative feed returns it."""
    payload = {
        "latestQuote": {
            "ap": ask, "as": 61, "ax": "P", "bp": bid, "bs": 51, "bx": "T", "c": "A",
            "t": stamp,
        },
        "latestTrade": {"c": "a", "p": (bid + ask) / 2, "s": 1, "t": stamp, "x": "A"},
    }
    if greeks:
        payload["greeks"] = {
            "delta": 0.5503, "gamma": 0.0417, "rho": 0.0343, "theta": -0.6677, "vega": 0.2748,
        }
        payload["impliedVolatility"] = 0.1366
    return payload


def sdk_snapshot(symbol, bid, ask, stamp="2026-08-26T14:34:58.500000000Z", *, greeks=True):
    """The real SDK model that ``get_option_snapshot`` returns per symbol."""
    return OptionsSnapshot(symbol=symbol, raw_data=quote_payload(bid, ask, stamp, greeks=greeks))


class FakeOptionClient:
    """Stands in for OptionHistoricalDataClient. Answers only symbols it knows."""

    def __init__(self, snapshots=(), *, fail=False):
        self._snapshots = {snapshot.symbol: snapshot for snapshot in snapshots}
        self.snapshot_requests = []
        self.fail = fail
        self.api_key = API_KEY
        self.secret_key = SECRET_KEY

    def get_option_snapshot(self, request):
        self.snapshot_requests.append(request)
        if self.fail:
            raise RuntimeError(f"401 unauthorized for key={API_KEY} secret={SECRET_KEY}")
        requested = request.symbol_or_symbols
        symbols = [requested] if isinstance(requested, str) else list(requested)
        return {s: self._snapshots[s] for s in symbols if s in self._snapshots}


CONTRACTS = [
    FakeContract("SPY260904C00766000", strike="766", expiration=date(2026, 9, 4)),
    FakeContract("SPY260901C00765000", strike="765", expiration=date(2026, 9, 1)),
]
SNAPSHOTS = [
    sdk_snapshot("SPY260901C00765000", 4.26, 4.50),
    sdk_snapshot("SPY260904C00766000", 5.80, 5.95, "2026-08-26T14:30:00Z", greeks=False),
]


def observe_with(action="BUY_CALL", *, trading=None, data=None, option=None, now=NOW):
    return observe_chain(
        trading or FakeTradingClient(contracts=CONTRACTS),
        data or FakeDataClient(),
        option or FakeOptionClient(SNAPSHOTS),
        action=action,
        now=now,
    )


# --------------------------------------------------------------------------
# 1. HOLD means there is nothing to look at
# --------------------------------------------------------------------------


def test_hold_action_observes_nothing_and_calls_no_api():
    trading, data, option = FakeTradingClient(), FakeDataClient(), FakeOptionClient()

    packet = observe_with("HOLD", trading=trading, data=data, option=option)

    assert isinstance(packet, ChainPacket)
    assert packet.action == "HOLD"
    assert packet.observed_at == NOW
    assert packet.underlying_mid is None
    assert packet.candidates == ()
    assert packet.quotes_read_at is None
    assert trading.option_requests == []
    assert trading.clock_requests == 0
    assert data.quote_requests == []
    assert option.snapshot_requests == []


# --------------------------------------------------------------------------
# 2. the join: contract identity + indicative quote, nothing judged
# --------------------------------------------------------------------------


def test_candidates_join_contract_identity_with_their_indicative_quote():
    packet = observe_with("BUY_CALL")

    assert packet.symbol == "SPY"
    assert packet.action == "BUY_CALL"
    assert packet.option_feed == "indicative"
    assert packet.underlying_mid == pytest.approx(765.0)

    # Sorted by expiration then strike, whatever order Alpaca returned.
    assert [c.symbol for c in packet.candidates] == [
        "SPY260901C00765000",
        "SPY260904C00766000",
    ]

    first = packet.candidates[0]
    assert isinstance(first, ContractCandidate)
    assert first.option_type == "call"
    assert first.strike_price == 765.0
    assert first.expiration_date == date(2026, 9, 1)
    assert first.days_to_expiration == 6  # from the New York date 2026-08-26
    assert first.status == "active"
    assert first.tradable is True
    assert first.bid == 4.26
    assert first.ask == 4.50
    assert first.quote_at == datetime(2026, 8, 26, 14, 34, 58, 500000, tzinfo=timezone.utc)

    # A snapshot without greeks or IV is still a perfectly good quote.
    second = packet.candidates[1]
    assert second.days_to_expiration == 9
    assert second.bid == 5.80
    assert second.ask == 5.95
    assert second.quote_at == datetime(2026, 8, 26, 14, 30, tzinfo=timezone.utc)


def test_the_candidate_carries_only_the_agreed_fields():
    """No greek, no IV, no open interest, no volume: 4A observes, it does not rank."""
    assert set(ContractCandidate.model_fields) == {
        "symbol",
        "option_type",
        "strike_price",
        "expiration_date",
        "days_to_expiration",
        "status",
        "tradable",
        "bid",
        "ask",
        "quote_at",
    }


# --------------------------------------------------------------------------
# 3-4. the contract request is narrowed on the server, from the New York date
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "action, contract_type",
    [("BUY_CALL", ContractType.CALL), ("BUY_PUT", ContractType.PUT)],
)
def test_contract_request_is_narrowed_server_side(action, contract_type):
    trading = FakeTradingClient(contracts=CONTRACTS)
    observe_with(action, trading=trading)

    assert len(trading.option_requests) == 1
    request = trading.option_requests[0]
    assert list(request.underlying_symbols) == ["SPY"]
    assert request.root_symbol == "SPY"
    assert request.status == AssetStatus.ACTIVE
    assert request.type == contract_type
    assert request.expiration_date_gte == date(2026, 8, 31)
    assert request.expiration_date_lte == date(2026, 9, 5)
    # Alpaca-py types strike bounds as strings; mid 765.00 +/- the query window.
    assert request.strike_price_gte == "760.00"
    assert request.strike_price_lte == "770.00"
    assert request.limit is not None and request.limit > 0
    assert request.page_token is None


def test_the_query_window_constants_are_the_approved_ones():
    """5-10 calendar days was approved; +/-$5 is a temporary query bound only."""
    assert (MIN_DAYS_TO_EXPIRATION, MAX_DAYS_TO_EXPIRATION) == (5, 10)
    assert QUERY_STRIKE_WINDOW == 5.0


@pytest.mark.parametrize(
    "now, earliest, latest",
    [
        # 10:35 New York on Wednesday 2026-08-26: UTC and New York agree.
        (datetime(2026, 8, 26, 14, 35, tzinfo=timezone.utc), date(2026, 8, 31), date(2026, 9, 5)),
        # 20:30 New York on the same Wednesday: UTC is already Thursday.
        (datetime(2026, 8, 27, 0, 30, tzinfo=timezone.utc), date(2026, 8, 31), date(2026, 9, 5)),
        # 23:30 New York on Tuesday 2026-01-13, in EST (a -4 offset would be wrong).
        (datetime(2026, 1, 14, 4, 30, tzinfo=timezone.utc), date(2026, 1, 18), date(2026, 1, 23)),
    ],
    ids=["midsession-edt", "after-utc-midnight-edt", "after-utc-midnight-est"],
)
def test_expiration_window_uses_the_new_york_calendar_date(now, earliest, latest):
    trading = FakeTradingClient(contracts=CONTRACTS)
    observe_with(trading=trading, now=now)

    request = trading.option_requests[0]
    assert request.expiration_date_gte == earliest
    assert request.expiration_date_lte == latest


# --------------------------------------------------------------------------
# 5-6. the snapshot request names the feed, and a silent symbol stays null
# --------------------------------------------------------------------------


def test_snapshot_request_names_the_indicative_feed_and_only_the_contract_symbols():
    option = FakeOptionClient(SNAPSHOTS)
    observe_with(option=option)

    assert len(option.snapshot_requests) == 1
    request = option.snapshot_requests[0]
    assert request.feed == OptionsFeed.INDICATIVE
    assert sorted(request.symbol_or_symbols) == ["SPY260901C00765000", "SPY260904C00766000"]


def test_a_symbol_absent_from_the_snapshot_reply_keeps_null_quote_fields():
    """The SDK drops symbols the feed returned as null; the contract still exists."""
    option = FakeOptionClient([sdk_snapshot("SPY260901C00765000", 4.26, 4.50)])
    packet = observe_with(option=option)

    quiet = next(c for c in packet.candidates if c.symbol == "SPY260904C00766000")
    assert quiet.strike_price == 766.0
    assert quiet.bid is None
    assert quiet.ask is None
    assert quiet.quote_at is None


# --------------------------------------------------------------------------
# 7. no SPY quote means no strike window, so no contract request at all
# --------------------------------------------------------------------------


def test_a_missing_underlying_quote_yields_no_candidates_and_no_contract_request():
    trading, option = FakeTradingClient(contracts=CONTRACTS), FakeOptionClient(SNAPSHOTS)
    packet = observe_with(trading=trading, data=FakeDataClient(quote=None), option=option)

    assert packet.underlying_mid is None
    assert packet.candidates == ()
    assert trading.option_requests == []
    assert option.snapshot_requests == []


def test_a_crossed_underlying_quote_is_not_a_usable_mid():
    packet = observe_with(data=FakeDataClient(quote=FakeQuote(bid=765.20, ask=765.10)))

    assert packet.underlying_mid is None
    assert packet.candidates == ()


# --------------------------------------------------------------------------
# 8-10. paging and batching stay within what the endpoints allow
# --------------------------------------------------------------------------


def test_contracts_are_paged_to_exhaustion():
    trading = FakeTradingClient(
        pages=[
            ([FakeContract("SPY260901C00765000", strike="765", expiration=date(2026, 9, 1))], "token-2"),
            ([FakeContract("SPY260901C00766000", strike="766", expiration=date(2026, 9, 1))], None),
        ]
    )
    packet = observe_with(trading=trading)

    assert len(trading.option_requests) == 2
    assert trading.option_requests[0].page_token is None
    assert trading.option_requests[1].page_token == "token-2"
    assert [c.strike_price for c in packet.candidates] == [765.0, 766.0]


def test_a_runaway_contract_window_is_an_error_not_a_slice():
    """More pages than the cap means an incomplete slice; that is a failure."""
    trading = FakeTradingClient(
        pages=[([FakeContract()], f"token-{n}") for n in range(MAX_OPTION_CONTRACT_PAGES + 1)]
    )

    with pytest.raises(ChainError) as caught:
        observe_with(trading=trading)

    assert str(MAX_OPTION_CONTRACT_PAGES) in str(caught.value)
    assert "token-" not in str(caught.value)
    assert len(trading.option_requests) == MAX_OPTION_CONTRACT_PAGES


def test_snapshot_symbols_are_requested_in_batches_of_one_hundred():
    contracts = [
        FakeContract(f"SPY260901C{765000 + n:08d}", strike=str(765 + n), expiration=date(2026, 9, 1))
        for n in range(150)
    ]
    option = FakeOptionClient([sdk_snapshot(c.symbol, 1.0, 1.1) for c in contracts])
    packet = observe_with(trading=FakeTradingClient(contracts=contracts), option=option)

    assert [len(r.symbol_or_symbols) for r in option.snapshot_requests] == [100, 50]
    assert len(packet.candidates) == 150
    assert all(c.bid == 1.0 for c in packet.candidates)


def test_an_empty_window_makes_no_snapshot_request():
    option = FakeOptionClient(SNAPSHOTS)
    packet = observe_with(trading=FakeTradingClient(contracts=[]), option=option)

    assert packet.underlying_mid == pytest.approx(765.0)
    assert packet.candidates == ()
    assert option.snapshot_requests == []


# --------------------------------------------------------------------------
# 11-12. a broken request is an error that never echoes a credential
# --------------------------------------------------------------------------


def assert_credential_safe(caught):
    message = str(caught.value)
    rendered = "".join(
        traceback.format_exception(type(caught.value), caught.value, caught.value.__traceback__)
    )
    for blob in (message, rendered):
        assert API_KEY not in blob
        assert SECRET_KEY not in blob


def test_a_snapshot_failure_is_a_chain_error_naming_only_the_step_and_type():
    with pytest.raises(ChainError) as caught:
        observe_with(option=FakeOptionClient(fail=True))

    assert "option snapshots" in str(caught.value)
    assert "RuntimeError" in str(caught.value)
    assert_credential_safe(caught)


def test_a_contract_failure_is_a_chain_error_naming_only_the_step_and_type():
    trading = FakeTradingClient()

    def explode(request):
        raise RuntimeError(f"boom key={API_KEY}")

    trading.get_option_contracts = explode

    with pytest.raises(ChainError) as caught:
        observe_with(trading=trading)

    assert "option contracts" in str(caught.value)
    assert_credential_safe(caught)


def test_an_underlying_quote_failure_is_a_chain_error_too():
    data = FakeDataClient()

    def explode(request):
        raise RuntimeError(f"boom secret={SECRET_KEY}")

    data.get_stock_latest_quote = explode

    with pytest.raises(ChainError) as caught:
        observe_with(data=data)

    assert "latest quote" in str(caught.value)
    assert_credential_safe(caught)


# --------------------------------------------------------------------------
# 13. the option data client sits behind the same paper guard as the others
# --------------------------------------------------------------------------


def paper_env(value):
    return {
        "ALPACA_API_KEY": API_KEY,
        "ALPACA_SECRET_KEY": SECRET_KEY,
        "ALPACA_PAPER": value,
    }


def test_build_option_data_client_refuses_non_paper_settings():
    with pytest.raises(ConfigError):
        build_option_data_client(settings_from_mapping(paper_env("false")))


def test_build_option_data_client_builds_from_paper_settings_without_a_network_call():
    client = build_option_data_client(settings_from_mapping(paper_env("true")))
    assert client is not None


# --------------------------------------------------------------------------
# 14-15. the summary shows exactly what the thresholds will be chosen from
# --------------------------------------------------------------------------


def test_format_summary_lists_each_candidate_with_spread_and_quote_age():
    rendered = format_summary(observe_with())

    assert "BUY_CALL" in rendered
    assert "indicative" in rendered
    assert "765.00" in rendered  # underlying mid and the ATM strike
    # 765 call: 4.26 / 4.50 -> spread 0.24 = 547.9 bps, quoted 1.5 s before observed_at.
    assert "4.26" in rendered and "4.50" in rendered
    assert "0.24" in rendered
    assert "547.9" in rendered
    assert "1.5" in rendered
    # 766 call: quoted five minutes earlier.
    assert "300.0" in rendered
    assert "2026-09-01" in rendered and "2026-09-04" in rendered
    assert "yes" in rendered


def test_quote_age_is_measured_against_the_alpaca_clock_not_the_local_one():
    """The observing machine's clock was found 14 s slow; ages must not depend on it."""
    server_now = NOW + timedelta(seconds=14)
    trading = FakeTradingClient(contracts=CONTRACTS, clock=FakeClock(timestamp=server_now))
    # Quoted 12.5 s after local "now": the future by the local clock, 1.5 s old by Alpaca's.
    option = FakeOptionClient(
        [sdk_snapshot("SPY260901C00765000", 4.26, 4.50, "2026-08-26T14:35:12.500000000Z")]
    )
    packet = observe_with(trading=trading, option=option)

    assert trading.clock_requests == 1
    assert packet.observed_at == NOW
    assert packet.quotes_read_at == server_now
    candidate = next(c for c in packet.candidates if c.symbol == "SPY260901C00765000")
    assert quote_age_seconds(candidate.quote_at, packet.quotes_read_at) == pytest.approx(1.5)

    rendered = format_summary(packet)
    assert "1.5" in rendered
    assert "+14.0" in rendered  # the skew is shown, never silently absorbed


def test_a_clock_without_a_timestamp_leaves_quotes_read_at_null_and_falls_back():
    trading = FakeTradingClient(contracts=CONTRACTS, clock=FakeClock(timestamp=None))
    packet = observe_with(trading=trading)

    assert packet.quotes_read_at is None
    assert "300.0" in format_summary(packet)  # measured against observed_at instead


def test_format_summary_renders_a_quiet_symbol_with_dashes_not_zeros():
    option = FakeOptionClient([sdk_snapshot("SPY260901C00765000", 4.26, 4.50)])
    rendered = format_summary(observe_with(option=option))

    quiet_line = next(line for line in rendered.splitlines() if "2026-09-04" in line)
    assert "0.00" not in quiet_line
    assert "-" in quiet_line


def test_format_summary_explains_hold_and_missing_quote():
    hold = format_summary(observe_with("HOLD"))
    assert "HOLD" in hold
    assert "no chain" in hold

    quiet = format_summary(observe_with(data=FakeDataClient(quote=None)))
    assert "no usable SPY quote" in quiet


# --------------------------------------------------------------------------
# 16-17. the packet is a record: serializable, credential-free, immutable
# --------------------------------------------------------------------------


def test_the_packet_serializes_without_credentials_and_round_trips():
    packet = observe_with()

    serialized = packet.model_dump_json()
    for blob in (serialized, format_summary(packet)):
        assert API_KEY not in blob
        assert SECRET_KEY not in blob
    assert ChainPacket.model_validate_json(serialized) == packet
    assert json.loads(serialized)["candidates"][0]["symbol"] == "SPY260901C00765000"


def test_the_packet_is_frozen_and_closed_to_stray_fields():
    packet = observe_with()

    with pytest.raises(Exception):
        packet.candidates = ()
    with pytest.raises(Exception):
        packet.candidates[0].bid = 0.0
    with pytest.raises(Exception):
        ChainPacket(**{**packet.model_dump(), "extra": True})
    assert not hasattr(packet.candidates, "append")


# --------------------------------------------------------------------------
# 18. the command needs a direction, and never gets one from the LLM
# --------------------------------------------------------------------------


@pytest.mark.parametrize("argv", [[], ["--json"], ["--action", "HOLD"], ["--action=SELL_CALL"]])
def test_main_refuses_to_run_without_a_buy_direction(argv, capsys):
    assert main(argv) == 1
    assert "usage" in capsys.readouterr().err


# --------------------------------------------------------------------------
# 19-20. read-only
# --------------------------------------------------------------------------


def test_the_chain_module_exposes_no_trading_or_execution_helper():
    forbidden = (
        "submit", "cancel", "replace", "close_position", "close_all", "exercise",
        "order", "buy_call", "buy_put", "place_", "position", "size", "risk", "decide",
    )
    offenders = [
        name for name in dir(chain_module) if any(word in name.lower() for word in forbidden)
    ]
    assert offenders == []


def test_the_chain_module_names_no_execution_endpoint():
    from pathlib import Path

    source = Path(chain_module.__file__).read_text(encoding="utf-8").lower()
    for word in ("submit_order", "cancel_order", "replace_order", "close_position",
                 "exercise_option", "orderrequest", "orderside"):
        assert word not in source
