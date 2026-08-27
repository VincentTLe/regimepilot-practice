"""Execution tests: the fresh pre-order re-check and the paper order boundary.

Every Alpaca client is a fake, so no network call is made and no real
credential is touched. Option snapshots are the SDK's own ``OptionsSnapshot``
models built offline, so the fresh quote is read from the exact reply shape
production sees. Nothing here ever reaches a real endpoint: the fakes explode
on any method the module is not allowed to call.
"""

import traceback
from datetime import date, datetime, timedelta, timezone
from enum import Enum
from pathlib import Path

import pytest
from alpaca.data.models.snapshots import OptionsSnapshot
from alpaca.trading.enums import OrderSide, OrderType, PositionIntent, TimeInForce
from alpaca.trading.requests import LimitOrderRequest

from regimepilot import execution as execution_module
from regimepilot.execution import ExecutionError, observe_execution_state, submit_paper_order
from regimepilot.models import ExecutionState, OrderPlan, OrderReceipt, SelectedContract

# 10:35 New York on Wednesday 2026-08-26 by the local clock. The server clock
# is fourteen seconds ahead, as it was found to be on the observing machine.
NOW = datetime(2026, 8, 26, 14, 35, tzinfo=timezone.utc)
SERVER_NOW = NOW + timedelta(seconds=14)
NEXT_CLOSE = datetime(2026, 8, 26, 20, 0, tzinfo=timezone.utc)
SUBMITTED_AT = SERVER_NOW + timedelta(seconds=1)
TODAY = date(2026, 8, 26)

API_KEY = "SUPER-SECRET-KEY"
SECRET_KEY = "SUPER-SECRET-SECRET"
ACCOUNT_ID = "11112222-3333-4444-5555-666677778888"
ORDER_ID = "aaaabbbb-cccc-dddd-eeee-ffff00001111"

SYMBOL = "SPY260902C00765000"
CLIENT_ORDER_ID = "regimepilot-x"


# --------------------------------------------------------------------------
# fakes
# --------------------------------------------------------------------------


class FakeAssetClass(Enum):
    US_EQUITY = "us_equity"
    US_OPTION = "us_option"


class FakeSide(Enum):
    LONG = "long"
    BUY = "buy"


class FakeOrderStatus(Enum):
    """Mimics alpaca's OrderStatus, which the receipt must unwrap to text."""

    NEW = "new"
    ACCEPTED = "accepted"
    FILLED = "filled"


class FakeAccount:
    """Alpaca returns account money as strings, so these are strings too."""

    def __init__(self, account_id=ACCOUNT_ID, equity="100000.55", options_buying_power="98000.75"):
        self.id = account_id
        self.equity = equity
        self.options_buying_power = options_buying_power


class FakePosition:
    def __init__(self, symbol=SYMBOL, asset_class=FakeAssetClass.US_OPTION, qty="1"):
        self.symbol = symbol
        self.asset_class = asset_class
        self.qty = qty
        self.side = FakeSide.LONG


class FakeOpenOrder:
    def __init__(self, symbol=SYMBOL, asset_class=FakeAssetClass.US_OPTION, order_id=ORDER_ID):
        self.id = order_id
        self.symbol = symbol
        self.asset_class = asset_class
        self.qty = "1"
        self.side = FakeSide.BUY
        self.status = FakeOrderStatus.NEW
        self.legs = None


class FakeClock:
    """Alpaca's market clock; ``timestamp`` is the server's own idea of now."""

    def __init__(self, timestamp=SERVER_NOW, is_open=True, next_close=NEXT_CLOSE):
        self.timestamp = timestamp
        self.is_open = is_open
        self.next_open = datetime(2026, 8, 27, 13, 30, tzinfo=timezone.utc)
        self.next_close = next_close


class FakeSubmittedOrder:
    """What ``submit_order`` / ``get_order_by_id`` return. Money and qty are strings."""

    def __init__(
        self,
        order_id=ORDER_ID,
        status=FakeOrderStatus.NEW,
        filled_qty="0",
        filled_avg_price=None,
        submitted_at=SUBMITTED_AT,
    ):
        self.id = order_id
        self.client_order_id = CLIENT_ORDER_ID
        self.status = status
        self.submitted_at = submitted_at
        self.filled_qty = filled_qty
        self.filled_avg_price = filled_avg_price
        self.qty = "1"
        self.limit_price = "5.49"


class FakeTradingClient:
    """Stands in for alpaca TradingClient with exactly the methods this module
    may call. Any other method explodes if it is ever reached. ``log`` records
    every call in order, and may be shared with the option client."""

    def __init__(
        self,
        *,
        account=...,
        positions=(),
        orders=(),
        clock=...,
        order=...,
        read_back=...,
        log=None,
    ):
        self._account = FakeAccount() if account is ... else account
        self._positions = list(positions)
        self._orders = list(orders)
        self._clock = FakeClock() if clock is ... else clock
        self._order = FakeSubmittedOrder() if order is ... else order
        self._read_back = self._order if read_back is ... else read_back
        self.log = [] if log is None else log
        self.submitted_requests = []
        self.read_back_ids = []
        # Deliberately carries credentials so the leak tests are meaningful.
        self.api_key = API_KEY
        self.secret_key = SECRET_KEY

    def get_account(self):
        self.log.append("get_account")
        return self._account

    def get_all_positions(self):
        self.log.append("get_all_positions")
        return self._positions

    def get_orders(self, request):
        self.log.append("get_orders")
        return self._orders

    def get_clock(self):
        self.log.append("get_clock")
        return self._clock

    def submit_order(self, request):
        self.log.append("submit_order")
        self.submitted_requests.append(request)
        return self._order

    def get_order_by_id(self, order_id):
        self.log.append("get_order_by_id")
        self.read_back_ids.append(order_id)
        return self._read_back

    def __getattr__(self, name):
        raise AssertionError(f"execution must not call trading_client.{name}")


def quote_payload(bid, ask, stamp):
    """One option snapshot exactly as the indicative feed returns it."""
    return {
        "latestQuote": {
            "ap": ask, "as": 61, "ax": "P", "bp": bid, "bs": 51, "bx": "T", "c": "A", "t": stamp,
        },
        "latestTrade": {"c": "a", "p": (bid + ask) / 2, "s": 1, "t": stamp, "x": "A"},
    }


def stamp(seconds_before=3.0, *, reference=SERVER_NOW):
    """A quote timestamp string ``seconds_before`` the reference clock."""
    return (reference - timedelta(seconds=seconds_before)).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def sdk_snapshot(symbol=SYMBOL, bid=5.40, ask=5.49, stamp_text=None):
    """The real SDK model that ``get_option_snapshot`` returns per symbol."""
    return OptionsSnapshot(
        symbol=symbol, raw_data=quote_payload(bid, ask, stamp() if stamp_text is None else stamp_text)
    )


class FakeOptionClient:
    """Stands in for OptionHistoricalDataClient. Answers only symbols it knows."""

    def __init__(self, snapshots=(), *, fail=False, log=None):
        self._snapshots = {snapshot.symbol: snapshot for snapshot in snapshots}
        self.fail = fail
        self.log = [] if log is None else log
        self.snapshot_requests = []
        self.api_key = API_KEY
        self.secret_key = SECRET_KEY

    def get_option_snapshot(self, request):
        self.log.append("get_option_snapshot")
        self.snapshot_requests.append(request)
        if self.fail:
            raise Boom()
        requested = request.symbol_or_symbols
        symbols = [requested] if isinstance(requested, str) else list(requested)
        return {s: self._snapshots[s] for s in symbols if s in self._snapshots}

    def __getattr__(self, name):
        raise AssertionError(f"execution must not call option_client.{name}")


class Boom(RuntimeError):
    """An upstream error that quotes the outbound request, as HTTP clients do."""

    def __init__(self):
        super().__init__(f"401 unauthorized for key={API_KEY} secret={SECRET_KEY}")


def explode(*args, **kwargs):
    raise Boom()


def selected_contract(**overrides):
    fields = dict(
        symbol=SYMBOL,
        option_type="call",
        strike_price=765.0,
        expiration_date=date(2026, 9, 2),
        days_to_expiration=7,
        bid=5.40,
        ask=5.50,
        mid=5.45,
        spread_bps=183.5,
        quote_at=NOW - timedelta(minutes=1),
        quote_age_seconds=1.0,
        underlying_mid=765.0,
    )
    fields.update(overrides)
    return SelectedContract(**fields)


def plan(**overrides):
    fields = dict(
        symbol=SYMBOL, qty=1, limit_price=5.49, max_premium_usd=549.0, client_order_id=CLIENT_ORDER_ID
    )
    fields.update(overrides)
    return OrderPlan(**fields)


def observe_with(*, trading=None, option=None, selected=None, now=NOW):
    return observe_execution_state(
        trading or FakeTradingClient(),
        option or FakeOptionClient([sdk_snapshot()]),
        selected=selected or selected_contract(),
        now=now,
    )


def assert_credential_safe(caught):
    message = str(caught.value)
    rendered = "".join(
        traceback.format_exception(type(caught.value), caught.value, caught.value.__traceback__)
    )
    for blob in (message, rendered):
        assert API_KEY not in blob
        assert SECRET_KEY not in blob


# --------------------------------------------------------------------------
# 1-2. the fresh re-check: account, quote, then the server clock
# --------------------------------------------------------------------------


def test_fresh_state_carries_account_quote_and_server_clock():
    """Quoted 3 s before the server clock, which is 11 s *after* the local
    clock: measured locally it would be a future stamp, so this also proves
    the age is taken on Alpaca's clock."""
    state = observe_with()

    assert isinstance(state, ExecutionState)
    assert state.observed_at == NOW
    assert state.account.account_id_masked == "****8888"
    assert state.account.options_buying_power == pytest.approx(98000.75)
    assert state.account.has_open_spy_option_position is False
    assert state.account.has_open_spy_option_order is False
    assert state.market_is_open is True
    assert state.minutes_to_close == pytest.approx((NEXT_CLOSE - SERVER_NOW).total_seconds() / 60)

    quote = state.quote
    assert quote.symbol == SYMBOL
    assert quote.bid == 5.40
    assert quote.ask == 5.49
    assert quote.quote_at == SERVER_NOW - timedelta(seconds=3)
    assert quote.server_time == SERVER_NOW
    assert quote.reject_reason is None


def test_the_clock_is_read_once_after_the_fresh_quote_and_nothing_is_submitted():
    log = []
    trading = FakeTradingClient(log=log)
    option = FakeOptionClient([sdk_snapshot()], log=log)

    observe_with(trading=trading, option=option)

    assert log.index("get_option_snapshot") < log.index("get_clock")
    assert log.count("get_clock") == 1
    assert log.count("get_option_snapshot") == 1
    assert "submit_order" not in log
    assert "get_order_by_id" not in log
    (request,) = option.snapshot_requests
    assert list(request.symbol_or_symbols) == [SYMBOL]


def test_an_existing_spy_option_position_and_order_reach_the_state():
    trading = FakeTradingClient(positions=[FakePosition()], orders=[FakeOpenOrder()])

    state = observe_with(trading=trading)

    assert state.account.has_open_spy_option_position is True
    assert state.account.has_open_spy_option_order is True


# --------------------------------------------------------------------------
# 3-5. the fresh quote is judged by the selector's rules
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "snapshot, reason",
    [
        (sdk_snapshot(stamp_text=stamp(30)), "stale_quote"),
        (sdk_snapshot(bid=5.50, ask=5.40), "invalid_quote"),
        (sdk_snapshot(stamp_text=stamp(-5)), "invalid_quote"),
        (sdk_snapshot(bid=5.00, ask=5.40), "wide_spread"),
    ],
    ids=["stale", "crossed", "future-stamped", "wide-spread"],
)
def test_a_quote_the_selector_would_refuse_is_refused_here_too(snapshot, reason):
    state = observe_with(option=FakeOptionClient([snapshot]))

    assert state.quote.reject_reason == reason
    assert state.quote.bid is not None and state.quote.ask is not None


def test_a_silent_feed_is_no_quote_not_an_error():
    state = observe_with(option=FakeOptionClient([]))

    assert state.quote.symbol == SYMBOL
    assert state.quote.bid is None
    assert state.quote.ask is None
    assert state.quote.quote_at is None
    assert state.quote.server_time == SERVER_NOW
    assert state.quote.reject_reason == "no_quote"


def test_days_to_expiration_is_recomputed_from_the_server_date():
    """A contract that expires today must not be ordered, whatever the
    selection said about its DTE when it was chosen."""
    selected = selected_contract(expiration_date=TODAY, days_to_expiration=7)

    state = observe_with(selected=selected)

    assert state.quote.reject_reason == "invalid_contract"


def test_an_incomplete_clock_leaves_fields_null_never_guessed():
    trading = FakeTradingClient(clock=FakeClock(timestamp=None, is_open=None, next_close=None))
    # Without a server clock the age falls back to observed_at, the local clock.
    option = FakeOptionClient([sdk_snapshot(stamp_text=stamp(3, reference=NOW))])

    state = observe_with(trading=trading, option=option)

    assert state.market_is_open is None
    assert state.minutes_to_close is None
    assert state.quote.server_time is None
    assert state.quote.reject_reason is None


# --------------------------------------------------------------------------
# 6. every failed read is an error that never echoes a credential
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "client, method, step",
    [
        ("trading", "get_account", "account"),
        ("trading", "get_all_positions", "positions"),
        ("trading", "get_orders", "open orders"),
        ("trading", "get_clock", "market clock"),
        ("option", "get_option_snapshot", "option snapshots"),
    ],
)
def test_a_failed_read_raises_and_names_only_the_step_and_type(client, method, step):
    trading, option = FakeTradingClient(), FakeOptionClient([sdk_snapshot()])
    setattr(trading if client == "trading" else option, method, explode)

    with pytest.raises(ExecutionError) as caught:
        observe_with(trading=trading, option=option)

    assert step in str(caught.value)
    assert "Boom" in str(caught.value)
    assert_credential_safe(caught)
    assert "submit_order" not in trading.log


def test_a_snapshot_failure_is_an_execution_error_too():
    with pytest.raises(ExecutionError) as caught:
        observe_with(option=FakeOptionClient(fail=True))

    assert "option snapshots" in str(caught.value)
    assert_credential_safe(caught)


def test_the_state_serializes_without_credentials_and_round_trips():
    state = observe_with()

    serialized = state.model_dump_json()
    assert API_KEY not in serialized
    assert SECRET_KEY not in serialized
    assert ACCOUNT_ID not in serialized
    assert ExecutionState.model_validate_json(serialized) == state


# --------------------------------------------------------------------------
# 7. the paper order: exactly one single-leg buy-to-open day limit
# --------------------------------------------------------------------------


def test_submit_sends_one_limit_day_buy_to_open_order_and_reads_it_back_once():
    filled = FakeSubmittedOrder(status=FakeOrderStatus.FILLED, filled_qty="1", filled_avg_price="5.45")
    trading = FakeTradingClient(read_back=filled)

    receipt = submit_paper_order(trading, plan())

    assert trading.log == ["submit_order", "get_order_by_id"]
    (request,) = trading.submitted_requests
    assert isinstance(request, LimitOrderRequest)
    assert request.symbol == SYMBOL
    assert request.qty == 1
    assert request.side == OrderSide.BUY
    assert request.type == OrderType.LIMIT
    assert request.time_in_force == TimeInForce.DAY
    assert request.limit_price == 5.49
    assert request.client_order_id == CLIENT_ORDER_ID
    assert request.position_intent == PositionIntent.BUY_TO_OPEN
    assert request.order_class is None
    assert request.notional is None
    assert request.extended_hours is None
    assert request.legs is None
    assert trading.read_back_ids == [ORDER_ID]

    assert isinstance(receipt, OrderReceipt)
    assert receipt.submitted is True
    assert receipt.order_id == ORDER_ID
    assert receipt.client_order_id == CLIENT_ORDER_ID
    assert receipt.status == "filled"
    assert receipt.submitted_at == SUBMITTED_AT
    assert receipt.filled_qty == 1.0
    assert receipt.filled_avg_price == 5.45
    assert receipt.error is None


def test_an_unfilled_order_is_still_a_submitted_receipt():
    receipt = submit_paper_order(FakeTradingClient(), plan())

    assert receipt.submitted is True
    assert receipt.status == "new"
    assert receipt.filled_qty == 0.0
    assert receipt.filled_avg_price is None
    assert receipt.error is None


def test_a_refused_submission_is_a_receipt_naming_only_the_type():
    trading = FakeTradingClient()
    trading.submit_order = explode

    receipt = submit_paper_order(trading, plan())

    assert receipt.submitted is False
    assert receipt.order_id is None
    assert receipt.client_order_id == CLIENT_ORDER_ID
    assert receipt.error == "failed to submit order: Boom"
    assert API_KEY not in receipt.model_dump_json()
    assert SECRET_KEY not in receipt.model_dump_json()
    assert API_KEY not in traceback.format_exc()
    assert "get_order_by_id" not in trading.log


def test_a_failed_read_back_keeps_the_submission_and_names_the_read_back():
    trading = FakeTradingClient(order=FakeSubmittedOrder(status=FakeOrderStatus.ACCEPTED))
    trading.get_order_by_id = explode

    receipt = submit_paper_order(trading, plan())

    assert receipt.submitted is True
    assert receipt.order_id == ORDER_ID
    assert receipt.status == "accepted"
    assert receipt.submitted_at == SUBMITTED_AT
    assert receipt.error == "failed to read back order: Boom"
    assert API_KEY not in receipt.model_dump_json()
    assert SECRET_KEY not in receipt.model_dump_json()
    assert trading.log == ["submit_order"]


@pytest.mark.parametrize(
    "overrides",
    [{"side": "sell"}, {"order_type": "market"}, {"time_in_force": "gtc"}, {"qty": 0}],
    ids=["sell", "market", "gtc", "zero-qty"],
)
def test_a_plan_that_is_not_a_buy_day_limit_is_refused_before_any_call(overrides):
    """The model forbids these; ``model_construct`` bypasses it, as a bug might."""
    fields = {**plan().model_dump(), **overrides}
    bad_plan = OrderPlan.model_construct(**fields)
    trading = FakeTradingClient()

    with pytest.raises(ExecutionError):
        submit_paper_order(trading, bad_plan)

    assert trading.log == []


# --------------------------------------------------------------------------
# 8. only this module may submit an order
# --------------------------------------------------------------------------


def test_only_the_execution_module_names_the_order_endpoint():
    package = Path(execution_module.__file__).parent
    words = ("submit_order", "LimitOrderRequest")
    offenders = {}
    for source in sorted(package.glob("*.py")):
        text = source.read_text(encoding="utf-8")
        hits = [word for word in words if word in text]
        if hits and source.name != "execution.py":
            offenders[source.name] = hits

    assert offenders == {}
    execution_text = (package / "execution.py").read_text(encoding="utf-8")
    assert all(word in execution_text for word in words)
