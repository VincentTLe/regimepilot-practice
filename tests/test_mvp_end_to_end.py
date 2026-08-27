"""One mocked portfolio lifecycle through the real modules, cycle by cycle.

Only the market-data reads that Phase 2B/3B cover elsewhere are stubbed
(features and news). Everything else runs for real against one stateful fake
Alpaca account: the paper-account read, the entry gates, the portfolio
context, the stub decision, the chain, the selector, the fresh re-check, the
entry and exit risk rules, the paper submission and the journal memory.

The story: flat → open a CALL → the position is seen next cycle and held →
a PUT is opened beside it → the PUT alone is closed with SELL_TO_CLOSE → the
CALL remains and a new entry is still possible → the journal remembers why
the CALL was opened.
"""

import json
from datetime import date, datetime, timedelta, timezone
from enum import Enum

import pytest
from alpaca.data.models.snapshots import OptionsSnapshot

from regimepilot import evidence as evidence_module
from regimepilot.memory import load_position_memory
from regimepilot.models import CycleRecord
from regimepilot.runner import append_record, run_cycle

from test_evidence import build_features, build_news

# Wednesday 2026-08-26, 10:35 New York. The server clock runs two seconds
# ahead of the cycle's start, and every option quote is stamped at start.
NOW = datetime(2026, 8, 26, 14, 35, tzinfo=timezone.utc)
SERVER_NOW = NOW + timedelta(seconds=2)
QUOTE_STAMP = "2026-08-26T14:35:00.000000000Z"

API_KEY = "SUPER-SECRET-KEY"
SECRET_KEY = "SUPER-SECRET-SECRET"

# 7 DTE from the session date, at the money on both sides.
SPY_CALL = "SPY260902C00765000"
SPY_CALL_766 = "SPY260902C00766000"
SPY_PUT = "SPY260902P00765000"
SPY_PUT_764 = "SPY260902P00764000"

QUOTES = {
    SPY_CALL: (5.44, 5.49),
    SPY_CALL_766: (4.98, 5.04),
    SPY_PUT: (5.10, 5.16),
    SPY_PUT_764: (4.70, 4.76),
}


class FakeEnum(Enum):
    US_OPTION = "us_option"
    LONG = "long"
    BUY = "buy"
    SELL = "sell"
    NEW = "new"
    ACCEPTED = "accepted"
    CALL = "call"
    PUT = "put"
    ACTIVE = "active"


class FakeAccount:
    id = "11112222-3333-4444-5555-666677778888"
    equity = "100000.55"
    options_buying_power = "98000.75"


class FakePosition:
    """An Alpaca position, money as text, built from the order that opened it."""

    def __init__(self, symbol, entry_price, *, qty="1"):
        self.symbol = symbol
        self.asset_class = FakeEnum.US_OPTION
        self.qty = qty
        self.qty_available = qty
        self.side = FakeEnum.LONG
        self.avg_entry_price = f"{entry_price:.2f}"
        self.cost_basis = f"{entry_price * 100 * float(qty):.2f}"
        mark = QUOTES[symbol][0] + 0.10
        self.current_price = f"{mark:.2f}"
        self.market_value = f"{mark * 100 * float(qty):.2f}"
        self.unrealized_pl = f"{(mark - entry_price) * 100 * float(qty):.2f}"
        self.unrealized_plpc = f"{(mark - entry_price) / entry_price:.4f}"


class FakeOpenOrder:
    def __init__(self, symbol, side=FakeEnum.SELL):
        self.id = "open-order-1"
        self.symbol = symbol
        self.asset_class = FakeEnum.US_OPTION
        self.qty = "1"
        self.side = side
        self.status = FakeEnum.NEW
        self.legs = None


class FakeClock:
    def __init__(self, timestamp=SERVER_NOW, is_open=True):
        self.timestamp = timestamp
        self.is_open = is_open
        self.next_open = datetime(2026, 8, 27, 13, 30, tzinfo=timezone.utc)
        self.next_close = datetime(2026, 8, 26, 20, 0, tzinfo=timezone.utc)


class FakeContract:
    def __init__(self, symbol, strike, contract_type, expiration=date(2026, 9, 2)):
        self.symbol = symbol
        self.type = contract_type
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

    def __init__(self, request, order_id, *, filled_qty="0", filled_avg_price=None):
        self.id = order_id
        self.client_order_id = request.client_order_id
        self.symbol = request.symbol
        self.qty = str(request.qty)
        self.limit_price = str(request.limit_price)
        self.status = FakeEnum.ACCEPTED
        self.submitted_at = SERVER_NOW + timedelta(seconds=1)
        self.filled_qty = filled_qty
        self.filled_avg_price = filled_avg_price


class FakeTradingClient:
    """One stateful paper account: positions and open orders the test mutates
    between cycles to play the fills. Every call is recorded."""

    def __init__(self):
        self.positions = []
        self.orders = []
        self.calls = []
        self.submitted = []
        self.api_key = API_KEY
        self.secret_key = SECRET_KEY

    def get_account(self):
        self.calls.append("get_account")
        return FakeAccount()

    def get_all_positions(self):
        self.calls.append("get_all_positions")
        return list(self.positions)

    def get_orders(self, request):
        self.calls.append("get_orders")
        return list(self.orders)

    def get_clock(self):
        self.calls.append("get_clock")
        return FakeClock()

    def get_option_contracts(self, request):
        self.calls.append("get_option_contracts")
        if request.type.value == "call":
            contracts = [FakeContract(SPY_CALL, "765", FakeEnum.CALL), FakeContract(SPY_CALL_766, "766", FakeEnum.CALL)]
        else:
            contracts = [FakeContract(SPY_PUT, "765", FakeEnum.PUT), FakeContract(SPY_PUT_764, "764", FakeEnum.PUT)]
        return FakeContractsResponse(contracts)

    def submit_order(self, request):
        self.calls.append("submit_order")
        self.submitted.append(request)
        return FakeSubmittedOrder(request, f"order-{len(self.submitted)}")

    def get_order_by_id(self, order_id):
        self.calls.append("get_order_by_id")
        request = self.submitted[int(order_id.split("-")[1]) - 1]
        return FakeSubmittedOrder(request, order_id, filled_qty=str(request.qty), filled_avg_price=str(request.limit_price))

    def __getattr__(self, name):
        raise AssertionError(f"the cycle must not call trading_client.{name}")

    # -- the test plays the fills -------------------------------------------------
    def fill_last(self):
        request = self.submitted[-1]
        if request.side.value == "buy":
            self.positions.append(FakePosition(request.symbol, float(request.limit_price)))
        else:
            self.positions = [p for p in self.positions if p.symbol != request.symbol]

    def submitted_this_cycle(self, before):
        return self.submitted[before:]


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
    def __init__(self):
        self.requests = []
        self.api_key = API_KEY
        self.secret_key = SECRET_KEY

    def get_option_snapshot(self, request):
        requested = request.symbol_or_symbols
        symbols = [requested] if isinstance(requested, str) else list(requested)
        self.requests.append(symbols)
        return {s: snapshot(s, *QUOTES[s]) for s in symbols if s in QUOTES}


class NoNewsClient:
    def __getattr__(self, name):
        raise AssertionError(f"the cycle must not call news_client.{name}")


@pytest.fixture
def market(monkeypatch):
    """Features and news exactly as Phase 3 tests build them: gates pass, momentum up."""
    monkeypatch.setattr(evidence_module, "observe_features", lambda *a, **k: build_features())
    monkeypatch.setattr(evidence_module, "observe_news", lambda *a, **k: build_news())


def cycle(trading, journal, n, **overrides):
    kwargs = dict(execute=True, stub=True, now=NOW + timedelta(minutes=15 * n), cycle_id=f"c{n}", journal_path=journal)
    kwargs.update(overrides)
    before = len(trading.submitted)
    record = run_cycle(trading, FakeDataClient(), FakeOptionClient(), NoNewsClient(), settings=None, **kwargs)
    append_record(record, journal)
    return record, trading.submitted_this_cycle(before)


def plans(record):
    return {(a.kind, a.symbol): a for a in record.actions}


def test_portfolio_lifecycle_enter_hold_add_close_one_and_remember(market, tmp_path):
    trading = FakeTradingClient()
    journal = tmp_path / "logs" / "cycles.jsonl"

    # ---- cycle 1: flat, momentum up -> the stub enters a CALL ------------------
    record, orders = cycle(trading, journal, 1)
    assert record.evidence.portfolio.open_position_count == 0
    assert record.evidence.portfolio.entry_allowed is True
    assert record.decision.new_entry.direction == "CALL"
    assert record.decision.positions == ()
    assert record.outcome == "submitted"
    (opened,) = record.actions
    assert (opened.kind, opened.symbol, opened.outcome) == ("open", SPY_CALL, "submitted")
    plan = opened.risk.plan
    assert (plan.side, plan.position_intent, plan.qty, plan.limit_price) == ("buy", "buy_to_open", 1, 5.49)
    assert plan.client_order_id == "regimepilot-c1-open"
    assert len(orders) == 1 and orders[0].side.value == "buy" and orders[0].position_intent.value == "buy_to_open"
    assert opened.receipt.order_id == "order-1"
    trading.fill_last()

    # ---- cycle 2: the CALL is seen, held, and a duplicate entry is refused ------
    record, orders = cycle(trading, journal, 2)
    portfolio = record.evidence.portfolio
    assert [p.symbol for p in portfolio.positions] == [SPY_CALL]
    held = portfolio.positions[0]
    assert (held.option_type, held.strike_price, held.qty, held.avg_entry_price) == ("call", 765.0, 1.0, 5.49)
    assert held.cost_basis == 549.0 and held.unrealized_pl is not None
    assert held.entry_thesis is not None and "momentum" in held.entry_thesis.lower()
    assert portfolio.entry_allowed is True  # one position never blocks another entry
    assert record.decision.positions[0].action == "HOLD"
    assert orders == []  # the stub re-proposes the same CALL: duplicate_symbol refuses it
    open_action = plans(record)[("open", SPY_CALL)]
    assert (open_action.outcome, open_action.risk.reason) == ("rejected", "duplicate_symbol")

    # ---- cycle 3: a PUT is opened beside the CALL (forced direction) ------------
    record, orders = cycle(trading, journal, 3, forced_enter="PUT")
    put_open = plans(record)[("open", SPY_PUT)]
    assert put_open.outcome == "submitted"
    assert put_open.risk.plan.client_order_id == "regimepilot-c3-open"
    assert put_open.risk.plan.limit_price == 5.16
    assert plans(record).get(("close", SPY_CALL)) is None
    assert len(orders) == 1 and orders[0].symbol == SPY_PUT
    trading.fill_last()

    # ---- cycle 4: close the PUT only; the CALL stays --------------------------
    record, orders = cycle(trading, journal, 4, forced_close=(SPY_PUT,))
    portfolio = record.evidence.portfolio
    assert [p.symbol for p in portfolio.positions] == [SPY_CALL, SPY_PUT]
    verdicts = {d.symbol: d.action for d in record.decision.positions}
    assert verdicts[SPY_CALL] == "HOLD" and verdicts[SPY_PUT] == "CLOSE"
    close = plans(record)[("close", SPY_PUT)]
    assert close.outcome == "submitted"
    plan = close.risk.plan
    assert (plan.side, plan.position_intent, plan.qty, plan.limit_price, plan.notional_usd) == (
        "sell", "sell_to_close", 1, 5.10, 510.0,
    )
    assert plan.client_order_id == "regimepilot-c4-close1"
    sells = [o for o in orders if o.side.value == "sell"]
    assert len(sells) == 1
    assert sells[0].symbol == SPY_PUT and sells[0].position_intent.value == "sell_to_close"
    assert plans(record).get(("close", SPY_CALL)) is None
    assert close.execution_state.quote.symbol == SPY_PUT
    trading.fill_last()

    # ---- cycle 5: one position left, memory intact, entry still possible --------
    record, orders = cycle(trading, journal, 5)
    portfolio = record.evidence.portfolio
    assert [p.symbol for p in portfolio.positions] == [SPY_CALL]
    assert portfolio.entry_allowed is True
    call = portfolio.positions[0]
    assert call.entry_thesis is not None and "momentum" in call.entry_thesis.lower()
    assert call.previous_decision is not None and call.previous_decision.startswith("HOLD")
    assert call.entered_at is not None and call.hours_held is not None

    # ---- the journal tells the whole story ------------------------------------
    lines = journal.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 5
    story = [(json.loads(l)["cycle_id"], json.loads(l)["outcome"]) for l in lines]
    assert story == [("c1", "submitted"), ("c2", "rejected"), ("c3", "submitted"), ("c4", "submitted"), ("c5", "rejected")]
    fourth = CycleRecord.model_validate_json(lines[3])
    assert fourth.evidence.portfolio.open_position_count == 2
    assert fourth.actions[0].receipt.filled_avg_price == 5.10
    memory = load_position_memory(journal)
    assert memory[SPY_CALL].entry_thesis == record.evidence.portfolio.positions[0].entry_thesis
    for line in lines:
        assert API_KEY not in line and SECRET_KEY not in line


def test_a_pending_order_blocks_only_its_own_symbol(market, tmp_path):
    trading = FakeTradingClient()
    trading.positions = [FakePosition(SPY_CALL, 5.49), FakePosition(SPY_PUT, 5.16)]
    trading.orders = [FakeOpenOrder(SPY_PUT, FakeEnum.SELL)]
    journal = tmp_path / "cycles.jsonl"

    record, orders = cycle(trading, journal, 1, forced_close=(SPY_CALL, SPY_PUT))

    results = plans(record)
    assert results[("close", SPY_PUT)].outcome == "rejected"
    assert results[("close", SPY_PUT)].risk.reason == "pending_order_conflict"
    assert results[("close", SPY_CALL)].outcome == "submitted"
    assert [o.symbol for o in orders if o.side.value == "sell"] == [SPY_CALL]
    # A pending SELL does not block a new entry; the stub's CALL is then a duplicate.
    assert record.evidence.portfolio.entry_allowed is True


def test_a_dry_run_never_submits_anything(market, tmp_path):
    trading = FakeTradingClient()
    trading.positions = [FakePosition(SPY_PUT, 5.16)]
    journal = tmp_path / "cycles.jsonl"

    record, orders = cycle(trading, journal, 1, execute=False, forced_close=(SPY_PUT,))

    assert record.mode == "dry_run"
    assert record.outcome == "planned"
    assert orders == []
    assert "submit_order" not in trading.calls
    assert plans(record)[("close", SPY_PUT)].outcome == "planned"
    assert plans(record)[("open", SPY_CALL)].outcome == "planned"


def test_a_failed_entry_gate_still_lets_a_position_be_closed(monkeypatch, tmp_path):
    """Correction 1: too_close_to_close blocks the new entry, never the exit."""
    from test_evidence import et

    monkeypatch.setattr(
        evidence_module, "observe_features", lambda *a, **k: build_features(session_close_at=et(10, 50))
    )
    monkeypatch.setattr(evidence_module, "observe_news", lambda *a, **k: build_news())
    trading = FakeTradingClient()
    trading.positions = [FakePosition(SPY_CALL, 5.49)]
    journal = tmp_path / "cycles.jsonl"

    record, orders = cycle(trading, journal, 1, forced_close=(SPY_CALL,), forced_enter="PUT")

    assert record.evidence.gates.hold_reason == "too_close_to_close"
    assert record.evidence.portfolio.entry_allowed is False
    assert record.decision.new_entry is None
    assert plans(record)[("close", SPY_CALL)].outcome == "submitted"
    assert [o.symbol for o in orders] == [SPY_CALL]
    assert ("open", SPY_PUT) not in plans(record)
