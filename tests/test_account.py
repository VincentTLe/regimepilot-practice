"""Account-state tests. Every Alpaca client is replaced with a fake, so no
network call is ever made and no real credential is ever touched."""

import traceback
from datetime import datetime, timezone
from enum import Enum

import pytest

from regimepilot.account import (
    OPEN_ORDER_LIMIT,
    AccountError,
    format_summary,
    is_spy_option,
    normalize_position,
    observe_account,
    occ_root,
)
from regimepilot.models import AccountState, OpenOrderSummary, PositionSummary

NOW = datetime(2026, 8, 26, 14, 30, tzinfo=timezone.utc)

API_KEY = "SUPER-SECRET-KEY"
SECRET_KEY = "SUPER-SECRET-SECRET"
ACCOUNT_ID = "11112222-3333-4444-5555-666677778888"
ORDER_ID = "aaaabbbb-cccc-dddd-eeee-ffff00001111"

SPY_CALL = "SPY260902C00765000"
SPY_PUT = "SPY260902P00766000"
QQQ_CALL = "QQQ260902C00500000"


class FakeAssetClass(Enum):
    """Mimics alpaca's AssetClass, which normalization must unwrap."""

    US_EQUITY = "us_equity"
    US_OPTION = "us_option"


class FakeSide(Enum):
    LONG = "long"
    SHORT = "short"
    BUY = "buy"
    SELL = "sell"


class FakeStatus(Enum):
    NEW = "new"


class FakeAccount:
    """Alpaca returns account money as strings, so these are strings too."""

    def __init__(self, account_id=ACCOUNT_ID, equity="100000.55", options_buying_power="98000.75"):
        self.id = account_id
        self.equity = equity
        self.options_buying_power = options_buying_power
        # Fields Phase 5A must NOT copy into the state.
        self.cash = "99000.25"
        self.account_number = "PA3ZZZZZZZZZ"


class FakePosition:
    def __init__(
        self,
        symbol=SPY_CALL,
        asset_class=FakeAssetClass.US_OPTION,
        qty="1",
        side=FakeSide.LONG,
    ):
        self.symbol = symbol
        self.asset_class = asset_class
        self.qty = qty
        self.side = side
        # Management facts the portfolio agent reads (as text, like Alpaca).
        self.avg_entry_price = "3.40"
        self.market_value = "350.00"
        self.cost_basis = "340.00"
        self.current_price = "3.50"
        self.unrealized_pl = "10.00"
        self.unrealized_plpc = "0.0294"
        self.qty_available = "1"
        # Fields the state must NOT copy.
        self.asset_id = "asset-uuid"
        self.exchange = "OPRA"


class FakeOrder:
    def __init__(
        self,
        symbol=SPY_CALL,
        asset_class=FakeAssetClass.US_OPTION,
        qty="1",
        side=FakeSide.BUY,
        status=FakeStatus.NEW,
        order_id=ORDER_ID,
        legs=None,
    ):
        self.id = order_id
        self.symbol = symbol
        self.asset_class = asset_class
        self.qty = qty
        self.side = side
        self.status = status
        self.legs = legs
        self.client_order_id = "client-order-1"


class FakeTradingClient:
    """Stands in for alpaca TradingClient with exactly the three read methods
    Phase 5A may call. Any other method explodes if it is ever reached."""

    def __init__(self, *, account=..., positions=(), orders=()):
        self._account = FakeAccount() if account is ... else account
        self._positions = list(positions)
        self._orders = list(orders)
        self.order_requests = []
        # Deliberately carries credentials so the leak tests are meaningful.
        self.api_key = API_KEY
        self.secret_key = SECRET_KEY

    def get_account(self):
        return self._account

    def get_all_positions(self):
        return self._positions

    def get_orders(self, request):
        self.order_requests.append(request)
        return self._orders

    def __getattr__(self, name):
        raise AssertionError(f"Phase 5A must not call trading_client.{name}")


class Boom(RuntimeError):
    """An upstream error that quotes the outbound request, as HTTP clients do."""

    def __init__(self):
        super().__init__(f"401 unauthorized for key={API_KEY} secret={SECRET_KEY}")


def explode(*args, **kwargs):
    raise Boom()


def observe_with(**kwargs):
    return observe_account(FakeTradingClient(**kwargs), now=NOW)


# --------------------------------------------------------------------------
# 1. an empty account is a confirmed empty state, not an unknown one
# --------------------------------------------------------------------------


def test_empty_account_is_a_confirmed_empty_state():
    state = observe_with()

    assert isinstance(state, AccountState)
    assert state.observed_at == NOW
    assert state.account_id_masked == "****8888"
    assert state.equity == pytest.approx(100000.55)
    assert state.options_buying_power == pytest.approx(98000.75)
    assert state.positions == ()
    assert state.open_orders == ()
    assert state.has_open_spy_option_position is False
    assert state.has_open_spy_option_order is False


# --------------------------------------------------------------------------
# 2-5. which positions count as a SPY option
# --------------------------------------------------------------------------


@pytest.mark.parametrize("symbol", [SPY_CALL, SPY_PUT], ids=["call", "put"])
def test_a_spy_option_position_sets_the_position_flag_only(symbol):
    state = observe_with(positions=[FakePosition(symbol=symbol)])

    assert state.has_open_spy_option_position is True
    assert state.has_open_spy_option_order is False
    assert state.positions == (
        PositionSummary(
            symbol=symbol, asset_class="us_option", side="long", qty=1.0, is_spy_option=True,
            avg_entry_price=3.4, cost_basis=340.0, current_price=3.5, market_value=350.0,
            unrealized_pl=10.0, unrealized_plpc=0.0294, qty_available=1.0,
        ),
    )


def test_a_short_spy_option_position_still_counts():
    state = observe_with(positions=[FakePosition(side=FakeSide.SHORT, qty="-1")])

    assert state.has_open_spy_option_position is True
    assert state.positions[0].side == "short"
    assert state.positions[0].qty == pytest.approx(-1.0)


@pytest.mark.parametrize("symbol", ["AAPL", "SPY"], ids=["other-stock", "the-spy-etf-itself"])
def test_a_stock_position_is_recorded_but_is_not_a_spy_option(symbol):
    state = observe_with(
        positions=[FakePosition(symbol=symbol, asset_class=FakeAssetClass.US_EQUITY, qty="10")]
    )

    assert state.has_open_spy_option_position is False
    assert state.positions[0].symbol == symbol
    assert state.positions[0].asset_class == "us_equity"
    assert state.positions[0].is_spy_option is False


@pytest.mark.parametrize(
    "symbol",
    [QQQ_CALL, "SPYX260902C00765000", "SPXW260902C05000000"],
    ids=["qqq", "root-starting-with-spy", "spxw"],
)
def test_an_option_on_another_root_is_not_a_spy_option(symbol):
    state = observe_with(positions=[FakePosition(symbol=symbol)])

    assert state.has_open_spy_option_position is False
    assert state.positions[0].is_spy_option is False


# --------------------------------------------------------------------------
# 6-7. open orders, kept apart from positions
# --------------------------------------------------------------------------


def test_an_open_spy_option_order_sets_the_order_flag_only():
    state = observe_with(orders=[FakeOrder(symbol=SPY_PUT)])

    assert state.has_open_spy_option_order is True
    assert state.has_open_spy_option_position is False
    assert state.open_orders == (
        OpenOrderSummary(
            order_id=ORDER_ID,
            symbol=SPY_PUT,
            asset_class="us_option",
            side="buy",
            qty=1.0,
            status="new",
            is_spy_option=True,
        ),
    )


def test_a_multi_leg_order_is_judged_by_its_legs():
    """An mleg parent carries no symbol of its own; alpaca puts the legs under it."""
    parent = FakeOrder(
        symbol=None,
        asset_class=None,
        side=None,
        legs=[FakeOrder(symbol=SPY_CALL, order_id="leg-1"), FakeOrder(symbol=SPY_PUT, order_id="leg-2")],
    )
    state = observe_with(orders=[parent])

    assert state.has_open_spy_option_order is True
    assert len(state.open_orders) == 1
    assert state.open_orders[0].symbol is None
    assert state.open_orders[0].is_spy_option is True


def test_an_unrelated_open_order_is_recorded_but_not_flagged():
    state = observe_with(
        orders=[FakeOrder(symbol="AAPL", asset_class=FakeAssetClass.US_EQUITY, qty="10")]
    )

    assert state.has_open_spy_option_order is False
    assert state.open_orders[0].symbol == "AAPL"
    assert state.open_orders[0].is_spy_option is False


def test_positions_and_open_orders_are_never_collapsed_into_one_flag():
    state = observe_with(
        positions=[FakePosition(symbol=SPY_CALL)],
        orders=[FakeOrder(symbol="AAPL", asset_class=FakeAssetClass.US_EQUITY)],
    )

    assert state.has_open_spy_option_position is True
    assert state.has_open_spy_option_order is False


def test_open_orders_are_requested_with_the_server_side_filters():
    client = FakeTradingClient()
    observe_account(client, now=NOW)

    (request,) = client.order_requests
    assert request.status == "open"
    assert request.limit == OPEN_ORDER_LIMIT == 500
    assert request.nested is True


# --------------------------------------------------------------------------
# 8-10. balance normalization and missing optional fields
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [("100000.55", 100000.55), (250000, 250000.0), (None, None), ("not-a-number", None)],
)
def test_equity_is_a_float_or_null_never_a_guess(raw, expected):
    state = observe_with(account=FakeAccount(equity=raw))
    assert state.equity == expected


@pytest.mark.parametrize(
    "raw, expected",
    [("98000.75", 98000.75), (0, 0.0), (None, None), ("", None)],
)
def test_options_buying_power_is_a_float_or_null_never_a_guess(raw, expected):
    state = observe_with(account=FakeAccount(options_buying_power=raw))
    assert state.options_buying_power == expected


def test_missing_optional_fields_become_null_without_failing():
    class BareAccount:
        id = ACCOUNT_ID

    class BareOrder:
        id = ORDER_ID
        symbol = "AAPL"

    state = observe_account(
        FakeTradingClient(account=BareAccount(), orders=[BareOrder()]), now=NOW
    )

    assert state.equity is None
    assert state.options_buying_power is None
    order = state.open_orders[0]
    assert order.asset_class is None
    assert order.side is None
    assert order.qty is None
    assert order.status is None
    assert order.is_spy_option is False


# --------------------------------------------------------------------------
# 11-12. a failed read is an error, never an empty account, and leaks nothing
# --------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["get_account", "get_all_positions", "get_orders"])
def test_a_failed_read_raises_and_never_reads_as_an_empty_account(method):
    client = FakeTradingClient(positions=[FakePosition()], orders=[FakeOrder()])
    setattr(client, method, explode)

    with pytest.raises(AccountError) as caught:
        observe_account(client, now=NOW)

    assert caught.value.args and isinstance(caught.value.args[0], str)


def test_the_error_names_the_step_and_type_without_leaking_credentials():
    client = FakeTradingClient()
    client.get_all_positions = explode

    with pytest.raises(AccountError) as caught:
        observe_account(client, now=NOW)

    message = str(caught.value)
    assert "positions" in message
    assert "Boom" in message  # the type is useful and safe

    rendered = "".join(
        traceback.format_exception(type(caught.value), caught.value, caught.value.__traceback__)
    )
    for blob in (message, rendered):
        assert API_KEY not in blob
        assert SECRET_KEY not in blob


def test_secrets_and_the_account_id_never_appear_in_the_output():
    state = observe_with(positions=[FakePosition()], orders=[FakeOrder()])

    for blob in (state.model_dump_json(), format_summary(state)):
        assert API_KEY not in blob
        assert SECRET_KEY not in blob
        assert ACCOUNT_ID not in blob
        assert "PA3ZZZZZZZZZ" not in blob
    assert "****8888" in state.model_dump_json()


# --------------------------------------------------------------------------
# 13. a reply this module cannot understand is unknown, not empty
# --------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["get_account", "get_all_positions", "get_orders"])
def test_a_null_reply_is_an_error_not_an_empty_account(method):
    client = FakeTradingClient()
    setattr(client, method, lambda *args, **kwargs: None)

    with pytest.raises(AccountError):
        observe_account(client, now=NOW)


def test_a_client_without_the_read_methods_is_an_error():
    class Empty:
        pass

    with pytest.raises(AccountError):
        observe_account(Empty(), now=NOW)


@pytest.mark.parametrize(
    "symbol",
    ["SPY1260902C00765000", "", None],
    ids=["adjusted-root", "empty", "missing"],
)
def test_an_option_position_with_an_unrecognized_symbol_is_an_error(symbol):
    with pytest.raises(AccountError) as caught:
        observe_with(positions=[FakePosition(symbol=symbol)])

    assert "positions" in str(caught.value)


@pytest.mark.parametrize("symbol", ["SPY1260902C00765000", None], ids=["adjusted-root", "missing"])
def test_an_option_order_with_an_unrecognized_symbol_is_an_error(symbol):
    with pytest.raises(AccountError) as caught:
        observe_with(orders=[FakeOrder(symbol=symbol)])

    assert "open orders" in str(caught.value)


@pytest.mark.parametrize("symbol", ["SPY1260902C00765000", None], ids=["adjusted-root", "missing"])
def test_an_option_leg_with_an_unrecognized_symbol_is_an_error(symbol):
    parent = FakeOrder(symbol=None, asset_class=None, side=None, legs=[FakeOrder(symbol=symbol, order_id="leg-1")])

    with pytest.raises(AccountError) as caught:
        observe_with(orders=[parent])

    assert "open orders" in str(caught.value)


def test_a_multi_leg_order_without_its_legs_is_an_error():
    with pytest.raises(AccountError) as caught:
        observe_with(orders=[FakeOrder(symbol=None, asset_class=None, legs=None)])

    assert "open orders" in str(caught.value)


def test_a_reply_that_fills_the_order_limit_is_an_error_not_a_list():
    orders = [
        FakeOrder(symbol="AAPL", asset_class=FakeAssetClass.US_EQUITY, order_id=f"order-{i}")
        for i in range(OPEN_ORDER_LIMIT)
    ]

    with pytest.raises(AccountError) as caught:
        observe_with(orders=orders)

    assert "open orders" in str(caught.value)


# --------------------------------------------------------------------------
# 14. the state is frozen, closed, serializable and self-consistent
# --------------------------------------------------------------------------


def test_the_state_is_frozen_closed_to_stray_fields_and_round_trips():
    state = observe_with(positions=[FakePosition()], orders=[FakeOrder(symbol=SPY_PUT)])

    with pytest.raises(Exception):
        state.equity = 1.0
    with pytest.raises(Exception):
        state.positions[0].is_spy_option = False
    with pytest.raises(Exception):
        AccountState(observed_at=NOW, account_id_masked="****8888", extra_field=1)

    restored = AccountState.model_validate_json(state.model_dump_json())
    assert restored == state
    assert not hasattr(restored.positions, "append")


def test_the_flags_must_agree_with_the_lines_they_summarize():
    base = {"observed_at": NOW, "account_id_masked": "****8888"}
    spy_position = PositionSummary(symbol=SPY_CALL, is_spy_option=True)
    spy_order = OpenOrderSummary(order_id=ORDER_ID, symbol=SPY_CALL, is_spy_option=True)

    with pytest.raises(ValueError):
        AccountState(**base, has_open_spy_option_position=True)
    with pytest.raises(ValueError):
        AccountState(**base, positions=(spy_position,), has_open_spy_option_position=False)
    with pytest.raises(ValueError):
        AccountState(**base, has_open_spy_option_order=True)
    with pytest.raises(ValueError):
        AccountState(**base, open_orders=(spy_order,), has_open_spy_option_order=False)

    consistent = AccountState(
        **base,
        positions=(spy_position,),
        open_orders=(spy_order,),
        has_open_spy_option_position=True,
        has_open_spy_option_order=True,
    )
    assert consistent.has_open_spy_option_position is True


# --------------------------------------------------------------------------
# 15. read-only guard
# --------------------------------------------------------------------------


def test_the_module_exposes_no_order_or_position_mutation():
    """Phase 5A stays read-only. Guard against an execution helper sneaking in."""
    import regimepilot.account as module

    forbidden = ("submit", "cancel", "replace", "close_position", "close_all", "exercise")
    offenders = [name for name in dir(module) if any(word in name.lower() for word in forbidden)]
    assert offenders == []


# --------------------------------------------------------------------------
# 16. the SPY-option rule itself
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "symbol, root",
    [
        (SPY_CALL, "SPY"),
        (SPY_PUT, "SPY"),
        (QQQ_CALL, "QQQ"),
        ("SPXW260902C05000000", "SPXW"),
        ("SPY1260902C00765000", None),  # adjusted root: not the plain contract
        ("spy260902c00765000", None),  # nothing is upper-cased
        ("SPY   260902C00765000", None),  # padded OCC form is not what alpaca sends
        (" SPY260902C00765000", None),  # nothing is stripped
        ("SPY260902C0076500", None),  # strike too short
        ("SPY", None),
        ("", None),
        (None, None),
    ],
)
def test_occ_root_reads_only_the_compact_alpaca_form(symbol, root):
    assert occ_root(symbol) == root


@pytest.mark.parametrize(
    "symbol, expected",
    [
        (SPY_CALL, ("SPY", "2026-09-02", "call", 765.0)),
        (SPY_PUT, ("SPY", "2026-09-02", "put", 766.0)),
        ("SPXW260902C05000000", ("SPXW", "2026-09-02", "call", 5000.0)),
        ("SPY260902C00765500", ("SPY", "2026-09-02", "call", 765.5)),
        ("SPY261399C00765000", None),  # month 13 is not a date
        ("SPY1260902C00765000", None),
    ],
)
def test_parse_occ_symbol_decodes_expiration_type_and_strike(symbol, expected):
    from regimepilot.account import parse_occ_symbol

    parsed = parse_occ_symbol(symbol)
    if expected is None:
        assert parsed is None
    else:
        assert (parsed.root, str(parsed.expiration_date), parsed.option_type, parsed.strike_price) == expected


def test_normalize_position_keeps_the_management_facts_as_floats():
    summary = normalize_position(FakePosition())

    assert summary.avg_entry_price == 3.40
    assert summary.cost_basis == 340.0
    assert summary.current_price == 3.50
    assert summary.market_value == 350.0
    assert summary.unrealized_pl == 10.0
    assert summary.unrealized_plpc == 0.0294
    assert summary.qty_available == 1.0
    assert "asset_id" not in summary.model_dump() and "exchange" not in summary.model_dump()


@pytest.mark.parametrize(
    "symbol, asset_class, expected",
    [
        (SPY_CALL, "us_option", True),
        (SPY_CALL, FakeAssetClass.US_OPTION, True),
        (SPY_CALL, None, True),  # no asset class (an mleg leg parent): the root decides
        ("SPY", "us_equity", False),  # the ETF itself
        (SPY_CALL, "us_equity", False),  # the SDK's asset class is trusted
        (QQQ_CALL, "us_option", False),
        (None, None, False),
    ],
)
def test_is_spy_option_needs_an_option_asset_class_and_the_spy_root(symbol, asset_class, expected):
    assert is_spy_option(symbol, asset_class) is expected


# --------------------------------------------------------------------------
# 17. the summary shows every line and both flags
# --------------------------------------------------------------------------


def test_summary_lists_every_position_and_order_with_both_flags():
    state = observe_with(
        positions=[
            FakePosition(),
            FakePosition(symbol="AAPL", asset_class=FakeAssetClass.US_EQUITY, qty="10"),
        ],
        orders=[FakeOrder(symbol=SPY_PUT)],
    )

    summary = format_summary(state)

    assert "****8888" in summary
    assert "100,000.55" in summary
    assert "98,000.75" in summary
    for symbol in (SPY_CALL, "AAPL", SPY_PUT):
        assert symbol in summary
    assert "position YES" in summary
    assert "open order YES" in summary
