"""One mocked cycle through the whole MVP pipeline, module by real module.

Only the market-data reads that Phase 2B/3B already cover elsewhere are
stubbed (features and news). Everything from the paper-account read through
gates, the stub proposal, the chain, the selector, the fresh re-check, the
risk decision and the paper submission runs for real against fake Alpaca
clients that record every call. This is the test that says "the vertical
slice is wired"; the per-module tests say each piece is right.
"""

import json
from datetime import date, datetime, timedelta, timezone
from enum import Enum

import pytest
from alpaca.data.models.snapshots import OptionsSnapshot

from regimepilot import evidence as evidence_module
from regimepilot.models import CycleRecord
from regimepilot.runner import append_record, run_cycle

from test_evidence import build_features, build_news

# Wednesday 2026-08-26, 10:35 New York. The server clock runs two seconds
# ahead of the cycle's start, and the fresh option quote is stamped at start.
NOW = datetime(2026, 8, 26, 14, 35, tzinfo=timezone.utc)
SERVER_NOW = NOW + timedelta(seconds=2)
QUOTE_STAMP = "2026-08-26T14:35:00.000000000Z"
STALE_STAMP = "2026-08-26T14:34:20.000000000Z"

API_KEY = "SUPER-SECRET-KEY"
SECRET_KEY = "SUPER-SECRET-SECRET"
ORDER_ID = "aaaabbbb-cccc-dddd-eeee-ffff00001111"

# Strike 765 expiring 2026-09-02: exactly 7 DTE from the session date, at the money.
SPY_CALL = "SPY260902C00765000"
SPY_CALL_766 = "SPY260902C00766000"


class FakeEnum(Enum):
    US_OPTION = "us_option"
    LONG = "long"
    BUY = "buy"
    NEW = "new"
    ACCEPTED = "accepted"
    CALL = "call"
    ACTIVE = "active"


class FakeAccount:
    id = "11112222-3333-4444-5555-666677778888"
    equity = "100000.55"
    options_buying_power = "98000.75"


class FakePosition:
    def __init__(self, symbol=SPY_CALL):
        self.symbol = symbol
        self.asset_class = FakeEnum.US_OPTION
        self.qty = "1"
        self.side = FakeEnum.LONG


class FakeOpenOrder:
    def __init__(self, symbol=SPY_CALL):
        self.id = "open-order-1"
        self.symbol = symbol
        self.asset_class = FakeEnum.US_OPTION
        self.qty = "1"
        self.side = FakeEnum.BUY
        self.status = FakeEnum.NEW
        self.legs = None


class FakeClock:
    def __init__(self, timestamp=SERVER_NOW, is_open=True):
        self.timestamp = timestamp
        self.is_open = is_open
        self.next_open = datetime(2026, 8, 27, 13, 30, tzinfo=timezone.utc)
        self.next_close = datetime(2026, 8, 26, 20, 0, tzinfo=timezone.utc)


class FakeContract:
    def __init__(self, symbol, strike, expiration=date(2026, 9, 2)):
        self.symbol = symbol
        self.type = FakeEnum.CALL
        self.strike_price = strike
        self.expiration_date = expiration
        self.status = FakeEnum.ACTIVE
        self.tradable = True


class FakeContractsResponse:
    def __init__(self, contracts):
        self.option_contracts = list(contracts)
        self.next_page_token = None


class FakeSubmittedOrder:
    """What ``submit_order`` and ``get_order_by_id`` return. Money is text."""

    def __init__(self, request, *, status=FakeEnum.ACCEPTED, filled_qty="0", filled_avg_price=None):
        self.id = ORDER_ID
        self.client_order_id = request.client_order_id
        self.symbol = request.symbol
        self.qty = str(request.qty)
        self.limit_price = str(request.limit_price)
        self.status = status
        self.submitted_at = SERVER_NOW + timedelta(seconds=1)
        self.filled_qty = filled_qty
        self.filled_avg_price = filled_avg_price


class FakeTradingClient:
    """Every trading-API call the cycle makes, recorded, in one fake."""

    def __init__(self, *, positions=(), orders=(), clock=None, submit_fails=False):
        self._positions = list(positions)
        self._orders = list(orders)
        self._clock = clock or FakeClock()
        self.submit_fails = submit_fails
        self.calls = []
        self.submitted = []
        self.api_key = API_KEY
        self.secret_key = SECRET_KEY

    def get_account(self):
        self.calls.append("get_account")
        return FakeAccount()

    def get_all_positions(self):
        self.calls.append("get_all_positions")
        return self._positions

    def get_orders(self, request):
        self.calls.append("get_orders")
        return self._orders

    def get_clock(self):
        self.calls.append("get_clock")
        return self._clock

    def get_option_contracts(self, request):
        self.calls.append("get_option_contracts")
        return FakeContractsResponse(
            [FakeContract(SPY_CALL, "765"), FakeContract(SPY_CALL_766, "766")]
        )

    def submit_order(self, request):
        self.calls.append("submit_order")
        if self.submit_fails:
            raise RuntimeError(f"403 forbidden for key={API_KEY} secret={SECRET_KEY}")
        self.submitted.append(request)
        return FakeSubmittedOrder(request)

    def get_order_by_id(self, order_id):
        self.calls.append("get_order_by_id")
        assert order_id == ORDER_ID
        return FakeSubmittedOrder(self.submitted[-1], filled_qty="1", filled_avg_price="5.49")

    def __getattr__(self, name):
        raise AssertionError(f"the cycle must not call trading_client.{name}")


class FakeQuote:
    def __init__(self, bid=764.90, ask=765.10):
        self.bid_price = bid
        self.ask_price = ask
        self.timestamp = NOW


class FakeDataClient:
    def __init__(self):
        self.api_key = API_KEY
        self.secret_key = SECRET_KEY

    def get_stock_latest_quote(self, request):
        return {"SPY": FakeQuote()}

    def __getattr__(self, name):
        raise AssertionError(f"the cycle must not call data_client.{name}")


def snapshot(symbol, bid, ask, stamp=QUOTE_STAMP):
    payload = {
        "latestQuote": {"ap": ask, "as": 61, "ax": "P", "bp": bid, "bs": 51, "bx": "T", "c": "A", "t": stamp},
        "latestTrade": {"c": "a", "p": (bid + ask) / 2, "s": 1, "t": stamp, "x": "A"},
    }
    return OptionsSnapshot(symbol=symbol, raw_data=payload)


class FakeOptionClient:
    def __init__(self, *, stamp=QUOTE_STAMP):
        self._snapshots = {
            SPY_CALL: snapshot(SPY_CALL, 5.44, 5.49, stamp),
            SPY_CALL_766: snapshot(SPY_CALL_766, 4.98, 5.04, stamp),
        }
        self.requests = []
        self.api_key = API_KEY
        self.secret_key = SECRET_KEY

    def get_option_snapshot(self, request):
        requested = request.symbol_or_symbols
        symbols = [requested] if isinstance(requested, str) else list(requested)
        self.requests.append(symbols)
        return {s: self._snapshots[s] for s in symbols if s in self._snapshots}


class NoNewsClient:
    """News is stubbed at the evidence layer; the client must never be touched."""

    def __getattr__(self, name):
        raise AssertionError(f"the cycle must not call news_client.{name}")


@pytest.fixture
def market(monkeypatch):
    """Features and news exactly as Phase 3 tests build them: gates pass, momentum up."""
    monkeypatch.setattr(evidence_module, "observe_features", lambda *a, **k: build_features())
    monkeypatch.setattr(evidence_module, "observe_news", lambda *a, **k: build_news())


def cycle(trading=None, option=None, **overrides):
    kwargs = dict(execute=False, stub=True, now=NOW, cycle_id="20260826-143500")
    kwargs.update(overrides)
    return run_cycle(
        trading or FakeTradingClient(),
        FakeDataClient(),
        option or FakeOptionClient(),
        NoNewsClient(),
        settings=None,
        **kwargs,
    )


# --------------------------------------------------------------------------
# 1. dry run: the whole pipeline, one plan, nothing submitted
# --------------------------------------------------------------------------


def test_dry_run_builds_one_plan_and_submits_nothing(market):
    trading = FakeTradingClient()

    record = cycle(trading)

    assert record.outcome == "planned"
    assert record.mode == "dry_run"
    assert record.proposal.action == "BUY_CALL" and record.proposal.model == "stub"
    assert record.selection.status == "selected"
    assert record.selection.selected.symbol == SPY_CALL
    assert record.execution_state.quote.symbol == SPY_CALL
    assert record.execution_state.quote.reject_reason is None
    assert record.execution_state.account.has_open_spy_option_position is False
    assert record.risk.approved is True
    plan = record.risk.plan
    assert (plan.symbol, plan.qty, plan.side, plan.order_type, plan.time_in_force) == (
        SPY_CALL, 1, "buy", "limit", "day",
    )
    assert plan.limit_price == 5.49
    assert plan.max_premium_usd == 549.0
    assert plan.client_order_id == "regimepilot-20260826-143500"
    assert record.receipt is None
    assert "submit_order" not in trading.calls
    # The account was read twice: once for the gate, once fresh before the plan.
    assert trading.calls.count("get_account") == 2


# --------------------------------------------------------------------------
# 2. execute: one paper order, exact shape, receipt with the read-back
# --------------------------------------------------------------------------


def test_execute_submits_exactly_one_day_limit_order_and_records_the_receipt(market):
    trading = FakeTradingClient()

    record = cycle(trading, execute=True)

    assert record.outcome == "submitted"
    assert record.mode == "execute"
    assert trading.calls.count("submit_order") == 1
    request = trading.submitted[0]
    assert request.symbol == SPY_CALL
    assert float(request.qty) == 1
    assert request.side.value == "buy"
    assert request.type.value == "limit"
    assert request.time_in_force.value == "day"
    assert request.limit_price == 5.49
    assert request.client_order_id == "regimepilot-20260826-143500"
    assert request.position_intent.value == "buy_to_open"
    assert request.order_class is None
    assert request.extended_hours is None
    receipt = record.receipt
    assert receipt.submitted is True
    assert receipt.order_id == ORDER_ID
    assert receipt.client_order_id == "regimepilot-20260826-143500"
    assert receipt.status == "accepted"
    assert receipt.filled_qty == 1.0
    assert receipt.filled_avg_price == 5.49
    # The fresh re-check happened before the order, never after.
    assert trading.calls.index("get_clock") < trading.calls.index("submit_order")


def test_a_refused_submission_is_an_error_record_without_a_leak(market):
    trading = FakeTradingClient(submit_fails=True)

    record = cycle(trading, execute=True)

    assert record.outcome == "error"
    assert record.receipt.submitted is False
    assert "RuntimeError" in record.error
    assert API_KEY not in record.model_dump_json()
    assert SECRET_KEY not in record.model_dump_json()


# --------------------------------------------------------------------------
# 3. duplicate protection: position holds the gate, open order refuses the plan
# --------------------------------------------------------------------------


def test_an_existing_spy_option_position_holds_before_any_chain_read(market):
    trading = FakeTradingClient(positions=[FakePosition()])

    record = cycle(trading, execute=True)

    assert record.outcome == "hold"
    assert record.proposal.gate_skipped is True
    assert "already_in_position" in record.proposal.thesis
    assert "get_option_contracts" not in trading.calls
    assert "submit_order" not in trading.calls


def test_an_open_spy_option_order_refuses_the_plan_before_submission(market):
    trading = FakeTradingClient(orders=[FakeOpenOrder()])

    record = cycle(trading, execute=True)

    assert record.outcome == "rejected"
    assert record.risk.reason == "existing_spy_option_order"
    assert record.risk.plan is None
    assert "submit_order" not in trading.calls


# --------------------------------------------------------------------------
# 4. --action never bypasses the gates; a stale fresh quote never becomes an order
# --------------------------------------------------------------------------


def test_a_forced_action_still_holds_when_the_gates_fail(monkeypatch):
    monkeypatch.setattr(
        evidence_module, "observe_features", lambda *a, **k: build_features(market_is_open=False)
    )
    monkeypatch.setattr(evidence_module, "observe_news", lambda *a, **k: build_news())
    trading = FakeTradingClient()

    record = cycle(trading, execute=True, forced_action="BUY_CALL")

    assert record.outcome == "hold"
    assert record.forced_action == "BUY_CALL"
    assert record.proposal.action == "HOLD"
    assert "market_closed" in record.proposal.thesis
    assert "get_option_contracts" not in trading.calls
    assert "submit_order" not in trading.calls


def test_a_forced_action_replaces_only_the_proposal_when_the_gates_pass(market):
    trading = FakeTradingClient()

    record = cycle(trading, forced_action="BUY_CALL")

    assert record.outcome == "planned"
    assert record.proposal.action == "BUY_CALL"
    assert record.proposal.model == "forced"
    assert record.risk.approved is True


def test_a_stale_fresh_quote_refuses_the_plan(market):
    trading = FakeTradingClient()

    record = cycle(trading, FakeOptionClient(stamp=STALE_STAMP), execute=True)

    # The selector already refuses the stale chain quote, so no contract is chosen;
    # either way nothing is submitted.
    assert record.outcome in ("no_contract", "rejected")
    assert "submit_order" not in trading.calls


def test_a_quote_that_goes_stale_between_selection_and_ordering_is_refused(market):
    class GoesStale(FakeOptionClient):
        def get_option_snapshot(self, request):
            reply = super().get_option_snapshot(request)
            if len(self.requests) == 1:
                return reply
            return {s: snapshot(s, 5.44, 5.49, STALE_STAMP) for s in reply}

    trading = FakeTradingClient()

    record = cycle(trading, GoesStale(), execute=True)

    assert record.selection.status == "selected"
    assert record.execution_state.quote.reject_reason == "stale_quote"
    assert record.outcome == "rejected"
    assert record.risk.reason == "unacceptable_quote"
    assert "submit_order" not in trading.calls


# --------------------------------------------------------------------------
# 5. the journal keeps the whole record
# --------------------------------------------------------------------------


def test_the_journal_line_round_trips_a_submitted_cycle(market, tmp_path):
    record = cycle(FakeTradingClient(), execute=True)
    path = tmp_path / "logs" / "cycles.jsonl"

    append_record(record, path)
    append_record(record, path)

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    loaded = CycleRecord.model_validate_json(lines[0])
    assert loaded == record
    row = json.loads(lines[0])
    assert row["outcome"] == "submitted"
    assert row["receipt"]["order_id"] == ORDER_ID
    assert row["risk"]["plan"]["limit_price"] == 5.49
    assert API_KEY not in lines[0]
