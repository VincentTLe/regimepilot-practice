from datetime import date
from types import SimpleNamespace

import pytest

import broker
from models import LegPlan, OrderPlan
from tests.fakes import FakeTradingClient, fake_position

API_KEY = "SUPER-SECRET-KEY"
SECRET = "SUPER-SECRET-VALUE"


def env(**overrides):
    base = {"ALPACA_API_KEY": API_KEY, "ALPACA_SECRET_KEY": SECRET, "ALPACA_PAPER": "true"}
    base.update(overrides)
    return base


# --- paper-only guards ---

def test_parse_bool_is_strict():
    assert broker.parse_bool("TRUE", name="x") is True
    assert broker.parse_bool("0", name="x") is False
    with pytest.raises(broker.ConfigError):
        broker.parse_bool("maybe", name="x")


@pytest.mark.parametrize(
    "bad",
    [
        env(ALPACA_PAPER="false"),
        env(ALPACA_LIVE="true"),
        env(ALPACA_LIVE_TRADING="1"),
        env(APCA_LIVE="yes"),
        env(ALPACA_BASE_URL="https://api.alpaca.markets"),
        env(APCA_API_BASE_URL="https://api.alpaca.markets/v2"),
    ],
)
def test_load_config_refuses_live_signals(bad):
    with pytest.raises(broker.ConfigError):
        broker.load_config(bad)


def test_paper_endpoint_is_not_a_live_signal():
    config = broker.load_config(env(ALPACA_BASE_URL="https://paper-api.alpaca.markets"))
    assert config.api_key == API_KEY


def test_load_config_requires_credentials():
    with pytest.raises(broker.ConfigError):
        broker.load_config({"ALPACA_PAPER": "true"})


def test_config_repr_never_leaks_credentials():
    config = broker.load_config(env())
    for text in (repr(config), str(config)):
        assert API_KEY not in text and SECRET not in text


def test_symbols_parse_dedupes_and_uppercases():
    config = broker.load_config(env(SYMBOLS=" spy, qqq ,SPY,nvda "))
    assert config.symbols == ("SPY", "QQQ", "NVDA")


def test_bad_bar_timeframe_is_a_config_error():
    with pytest.raises(broker.ConfigError):
        broker.load_config(env(BAR_TIMEFRAME="fifteen"))


# --- account state ---

def test_fetch_account_state_parses_legs_and_flags_the_rest():
    order_leg = SimpleNamespace(symbol="SPY260911C00655000", legs=None)
    parent = SimpleNamespace(symbol=None, legs=[order_leg])
    trading = FakeTradingClient(
        positions=[
            fake_position("SPY260911C00650000", 1, 6.0, side="long"),
            fake_position("SPY260911C00655000", 1, 3.5, side="short"),
            fake_position("WEIRD-SYMBOL", 1, 1.0),
            SimpleNamespace(symbol="AAPL", qty="10", side="long", avg_entry_price="150", asset_class="us_equity"),
        ],
        orders=[parent],
    )
    state = broker.fetch_account_state(trading, ("SPY",))
    assert state.equity == 100_000.0 and state.options_level == 3
    assert [leg.qty for leg in state.legs] == [1, -1]  # short side is negative
    assert state.unparsed_positions == ("WEIRD-SYMBOL",)
    assert "SPY260911C00655000" in state.open_order_symbols  # nested order legs collected


def test_fetch_account_state_read_failure_raises_not_empty():
    class Exploding(FakeTradingClient):
        def get_all_positions(self):
            raise RuntimeError(API_KEY)  # must never leak

    with pytest.raises(broker.BrokerError) as excinfo:
        broker.fetch_account_state(Exploding(), ("SPY",))
    assert API_KEY not in str(excinfo.value)
    assert "RuntimeError" in str(excinfo.value)


# --- the one submitting function ---

def entry_plan(**overrides):
    base = dict(
        kind="enter",
        underlying="SPY",
        qty=1,
        limit_price=2.8,
        legs=(
            LegPlan(symbol="SPY260911C00650000", side="buy", intent="buy_to_open"),
            LegPlan(symbol="SPY260911C00655000", side="sell", intent="sell_to_open"),
        ),
        client_order_id="sp-x-enter-SPY",
    )
    base.update(overrides)
    return OrderPlan(**base)


def test_submit_valid_entry_plan():
    trading = FakeTradingClient()
    receipt = broker.submit_paper_order(trading, entry_plan())
    assert receipt.submitted is True and receipt.order_id == "order-1"
    request = trading.submitted[0]
    assert str(request.order_class.value) == "mleg"
    assert len(request.legs) == 2 and request.qty == 1


def test_submit_valid_exit_plan_with_credit():
    plan = entry_plan(
        kind="exit",
        limit_price=-2.8,
        legs=(
            LegPlan(symbol="SPY260911C00650000", side="sell", intent="sell_to_close"),
            LegPlan(symbol="SPY260911C00655000", side="buy", intent="buy_to_close"),
        ),
    )
    receipt = broker.submit_paper_order(FakeTradingClient(), plan)
    assert receipt.submitted is True


@pytest.mark.parametrize(
    "bad",
    [
        entry_plan(time_in_force="gtc"),
        entry_plan(order_class="simple"),
        entry_plan(qty=0),
        entry_plan(limit_price=0.0),  # entry must pay a debit
        entry_plan(limit_price=-1.0),
        entry_plan(client_order_id=""),
        entry_plan(legs=(  # duplicate symbols
            LegPlan(symbol="SAME", side="buy", intent="buy_to_open"),
            LegPlan(symbol="SAME", side="sell", intent="sell_to_open"),
        )),
        entry_plan(legs=(  # exit shapes on an entry
            LegPlan(symbol="A", side="sell", intent="sell_to_close"),
            LegPlan(symbol="B", side="buy", intent="buy_to_close"),
        )),
    ],
)
def test_submit_refuses_malformed_plans(bad):
    trading = FakeTradingClient()
    with pytest.raises(broker.BrokerError):
        broker.submit_paper_order(trading, bad)
    assert trading.submitted == []


def test_submit_refusal_reports_type_name_only():
    trading = FakeTradingClient(submit_error=RuntimeError(f"boom {SECRET}"))
    receipt = broker.submit_paper_order(trading, entry_plan())
    assert receipt.submitted is False
    assert receipt.error == "RuntimeError"
    assert SECRET not in (receipt.error or "")
