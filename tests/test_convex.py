"""convex: window phases, sizing, exits, contract checks, the choice parser, the foreign-holdings
refusal, and full cycles on the fakes (dry run, armed entry, unfilled cancel, take-profit exit,
time / market exits, gates, stale recheck), plus the single-leg broker path."""

from __future__ import annotations

import json
from datetime import date, datetime, time, timedelta, timezone
from types import SimpleNamespace

import httpx
import pandas as pd
import pytest
from alpaca.trading.enums import OrderClass, OrderSide, PositionIntent
from alpaca.trading.requests import LimitOrderRequest, MarketOrderRequest

import broker
import convex
import settings
from data_models import LegQuote, OpenOption, SingleLegPlan
from tests.fakes import (
    FakeOptionDataClient,
    FakeStockDataClient,
    FakeTradingClient,
    fake_account,
    fake_contract,
    fake_position,
    fake_snapshot,
    quiet_bars,
)

UTC = timezone.utc
FRI = datetime(2026, 9, 4, 13, 35, tzinfo=UTC)  # 09:35 ET: first minute of the entry window
EXP0, EXP1 = date(2026, 9, 4), date(2026, 9, 8)
CALL_774 = "SPY260904C00774000"


@pytest.fixture(autouse=True)
def convex_env(tmp_path, monkeypatch):
    monkeypatch.setattr(convex, "JOURNAL_PATH", tmp_path / "convex.jsonl")
    monkeypatch.setattr(convex.sounds, "play_order_sound", lambda: None)
    monkeypatch.setattr(convex.sounds, "play_fill_sound", lambda: None)
    monkeypatch.setattr(convex.clock_time, "sleep", lambda seconds: None)
    for name, value in {
        "CONVEX_SYMBOLS": ("SPY", "QQQ"), "CONVEX_MAX_EXPIRY_DAYS": 4, "CONVEX_STRIKE_BAND_PCT": 0.02,
        "CONVEX_STRIKES_EACH_SIDE": 3, "CONVEX_CASH_FRACTION": 0.95, "CONVEX_MAX_CONTRACTS": 250,
        "CONVEX_MAX_SPREAD_BPS": 1500, "CONVEX_MAX_QUOTE_AGE_SECONDS": 30, "CONVEX_MIN_OPEN_INTEREST": 100,
        "CONVEX_STOP_FRACTION": 0.5, "CONVEX_TAKE_PROFIT_MULT": 4.0, "CONVEX_ENTRY_START": time(9, 35),
        "CONVEX_ENTRY_END": time(10, 30), "CONVEX_TIME_EXIT": time(10, 45), "CONVEX_MARKET_EXIT": time(10, 55),
        "CONVEX_SESSION_END": time(11, 0), "CONVEX_COOLDOWN_SECONDS": 300, "CONVEX_MAX_ENTRIES_PER_DAY": 2,
    }.items():
        monkeypatch.setattr(settings, name, value)


def clock_at(stamp: datetime, is_open: bool = True) -> SimpleNamespace:
    return SimpleNamespace(timestamp=stamp, is_open=is_open, next_close=stamp + timedelta(hours=6))


def occ(underlying: str, expiration: date, kind: str, strike: float) -> str:
    return f"{underlying}{expiration.strftime('%y%m%d')}{kind}{int(round(strike * 1000)):08d}"


def contracts(underlying: str = "SPY", strikes=range(770, 779)) -> list:
    out = []
    for expiration in (EXP0, EXP1):
        for kind in "CP":
            for strike in strikes:
                out.append(fake_contract(occ(underlying, expiration, kind, strike), float(strike), expiration, 5000))
    return out


def snapshots(bid: float = 2.95, ask: float = 3.05, stamp: datetime = FRI, symbols=None) -> dict:
    symbols = symbols or [c.symbol for c in contracts()] + [c.symbol for c in contracts("QQQ", range(560, 569))]
    return {symbol: fake_snapshot(bid, ask, iv=None, stamp=stamp) for symbol in symbols}


def stock(now: datetime = FRI) -> FakeStockDataClient:
    bars = quiet_bars(60, end=now - timedelta(seconds=300), bar_seconds=300)
    return FakeStockDataClient(
        bars_by_symbol={"SPY": bars, "QQQ": bars},
        quotes_by_symbol={"SPY": (773.9, 774.1), "QQQ": (563.9, 564.1)},
    )


def trading(**kwargs) -> FakeTradingClient:
    kwargs.setdefault("account", fake_account(cash=93_770.4, options_buying_power=93_770.4))
    kwargs.setdefault("clock", clock_at(FRI))
    kwargs.setdefault("contracts", contracts() + contracts("QQQ", range(560, 569)))
    return FakeTradingClient(**kwargs)


def config():
    from dataclasses import replace

    from tests.test_cli import make_config

    return replace(make_config(), llm_api_key="key")  # the convex mode has no manual decider


def chat(content: str) -> httpx.MockTransport:
    return httpx.MockTransport(lambda request: httpx.Response(200, json={"choices": [{"message": {"content": content}}], "model": "test-model"}))


def raising_transport() -> httpx.MockTransport:
    def handler(request):
        raise AssertionError("the model must not be called")

    return httpx.MockTransport(handler)


ENTER_SPY = json.dumps({"action": "enter", "symbol": "SPY", "direction": "CALL", "expiration": "2026-09-04", "strike": 774, "thesis": "go"})


def cycle(trading_client, option_client, *, execute=False, transport=None, state=None, now=FRI, flatten_now=False):
    state = state or convex.ConvexState()
    record = convex.run_cycle(config(), trading_client, stock(now), option_client, execute=execute, state=state,
                              llm_transport=transport, flatten_now=flatten_now)
    return record, state


# --- pure helpers ---

def test_phase_at_boundaries():
    assert convex.phase_at(time(9, 20)) == "pre"
    assert convex.phase_at(time(9, 35)) == "entry"
    assert convex.phase_at(time(10, 30)) == "hold"
    assert convex.phase_at(time(10, 45)) == "time_exit"
    assert convex.phase_at(time(10, 55)) == "market_exit"
    assert convex.phase_at(time(11, 0)) == "done"


def test_size_all_in(monkeypatch):
    assert convex.size_all_in(93_770.4, 93_770.4, 3.05) == (250, None)  # the cap binds
    monkeypatch.setattr(settings, "CONVEX_MAX_CONTRACTS", 1000)
    assert convex.size_all_in(93_770.4, 93_770.4, 3.05) == (292, None)
    assert convex.size_all_in(93_770.4, 20_000, 3.05) == (62, None)  # options buying power binds
    assert convex.size_all_in(None, None, 3.05) == (0, "unknown_cash")
    assert convex.size_all_in(93_770.4, None, 0) == (0, "bad_ask")
    assert convex.size_all_in(200, 200, 3.05) == (0, "insufficient_cash")


def test_exit_reason_tp_stop_time_hold():
    pos = OpenOption(symbol=CALL_774, underlying="SPY", expiration=EXP0, option_type="C", strike=774.0, qty=250, avg_entry_price=3.0)
    assert convex.exit_reason(pos, 12.0, "hold") == "take_profit"
    assert convex.exit_reason(pos, 1.5, "hold") == "stop"
    assert convex.exit_reason(pos, 3.2, "hold") is None
    assert convex.exit_reason(pos, 3.2, "time_exit") == "time"
    assert convex.exit_reason(pos, None, "hold") is None
    assert convex.exit_reason(pos, None, "market_exit") == "time"


def quote(bid, ask, *, stamp=FRI, oi=5000, iv=None) -> LegQuote:
    return LegQuote(symbol=CALL_774, strike=774.0, bid=bid, ask=ask, implied_vol=iv, open_interest=oi, quote_time=stamp)


def test_check_contract_tolerates_missing_iv_but_not_stale_wide_or_thin():
    assert convex.check_contract(quote(2.95, 3.05), FRI) is None
    assert convex.check_contract(quote(2.95, 3.05, stamp=FRI - timedelta(seconds=60)), FRI) == "stale_quote"
    assert convex.check_contract(quote(0.5, 1.5), FRI) == "wide_spread"
    assert convex.check_contract(quote(2.95, 3.05, oi=10), FRI) == "low_open_interest"
    assert convex.check_contract(quote(None, None), FRI) == "no_quote"
    assert convex.check_contract(quote(3.1, 3.0), FRI) == "crossed_quote"


def test_pick_strikes_three_each_side_per_expiry_and_type():
    rows = [{"expiration": e, "type": t, "strike": float(s), "symbol": f"{e}{t}{s}"} for e in (EXP0, EXP1) for t in "CP" for s in range(760, 790)]
    picked = convex.pick_strikes(rows, 774.2)
    assert len(picked) == 4 * 6
    calls_0 = [r["strike"] for r in picked if r["expiration"] == EXP0 and r["type"] == "C"]
    assert calls_0 == [772.0, 773.0, 774.0, 775.0, 776.0, 777.0]


def test_session_facts_from_a_5m_frame():
    stamps = [datetime(2026, 9, 3, 19, 55, tzinfo=UTC), datetime(2026, 9, 4, 13, 30, tzinfo=UTC), datetime(2026, 9, 4, 13, 35, tzinfo=UTC)]
    frame = pd.DataFrame({"open": [770.0, 772.0, 773.0], "high": [771.0, 773.5, 774.5], "low": [769.0, 771.5, 772.5],
                          "close": [770.5, 773.0, 774.0]}, index=pd.DatetimeIndex(stamps))
    facts = convex.session_facts(frame, datetime(2026, 9, 4, 9, 41, tzinfo=convex.ET))
    assert facts["prev_close"] == 770.5 and facts["open"] == 772.0 and facts["high"] == 774.5 and facts["last"] == 774.0
    assert facts["first_bar_direction"] == "up" and facts["minutes_since_open"] == 11
    assert facts["gap_pct"] == round(100 * (772.0 - 770.5) / 770.5, 2)


ELIGIBLE = {("SPY", EXP0, "C", 774.0): CALL_774}


@pytest.mark.parametrize("text", [
    "no json here", "[1, 2]", json.dumps({"action": "buy", "symbol": "SPY"}),
    json.dumps({"action": "enter", "symbol": "IWM", "direction": "CALL", "expiration": "2026-09-04", "strike": 774}),
    json.dumps({"action": "enter", "symbol": "SPY", "direction": "CALL", "expiration": "2026-09-04", "strike": 999}),
    json.dumps({"action": "enter", "symbol": "SPY", "direction": "CALL", "expiration": "tomorrow", "strike": 774}),
    json.dumps({"action": "enter", "symbol": "SPY", "direction": "PUT", "expiration": "2026-09-04", "strike": 774}),  # not eligible
])
def test_parse_convex_choice_rejects_garbage(text):
    assert convex.parse_convex_choice(text, ELIGIBLE, "m") is None


def test_parse_convex_choice_accepts_fenced_json_and_keeps_pass_thesis():
    fenced = "```json\n" + ENTER_SPY + "\n```"
    choice = convex.parse_convex_choice(fenced, ELIGIBLE, "m")
    assert choice is not None and choice.action == "enter" and choice.contract_symbol == CALL_774 and choice.strike == 774.0
    passed = convex.parse_convex_choice(json.dumps({"action": "pass", "thesis": "tape and gap disagree"}), ELIGIBLE, "m")
    assert passed is not None and passed.action == "pass" and passed.thesis == "tape and gap disagree"


def account_state(positions=(), orders=()):
    return broker.fetch_account_state(FakeTradingClient(positions=positions, orders=orders), ("SPY", "QQQ"))


def test_foreign_holdings_refuses_spreads_shorts_and_foreign_orders():
    spread = [fake_position("SPY260911C00650000", 1, 6.0, side="long"), fake_position("SPY260911C00655000", 1, 4.0, side="short")]
    assert any(r.startswith("spread SPY") for r in convex.foreign_holdings(account_state(spread), {}, EXP0))
    assert convex.foreign_holdings(account_state([fake_position("SPY260904P00770000", 1, 1.0, side="short")]), {}, EXP0) == ["short leg SPY260904P00770000"]
    assert convex.foreign_holdings(account_state(), {"o1": "sp-20260904-enter-SPY"}, EXP0) == ["foreign order sp-20260904-enter-SPY"]
    assert convex.foreign_holdings(account_state([fake_position("IWM260904C00296000", 1, 1.0, side="long")]), {}, EXP0) == ["foreign long IWM260904C00296000"]
    own = account_state([fake_position(CALL_774, 250, 3.0, side="long")])
    assert convex.foreign_holdings(own, {"o2": "cx-20260904-enter-SPY"}, EXP0) == []
    assert [p.symbol for p in convex.owned_positions(own, EXP0)] == [CALL_774]


# --- cycles on the fakes ---

def test_run_cycle_refuses_on_foreign_positions():
    spread = [fake_position("SPY260911C00650000", 1, 6.0, side="long"), fake_position("SPY260911C00655000", 1, 4.0, side="short")]
    record, _ = cycle(trading(positions=spread), FakeOptionDataClient(snapshots()), transport=raising_transport())
    assert record["outcome"] == "foreign_positions" and record["foreign"]


def test_dry_run_entry_plans_a_single_leg_and_submits_nothing(tmp_path):
    t = trading()
    record, state = cycle(t, FakeOptionDataClient(snapshots()), transport=chat(ENTER_SPY))
    assert record["outcome"] == "planned" and t.submitted == []
    plan = record["entry"]["receipt"]["plan"]
    assert plan == {"kind": "enter", "symbol": CALL_774, "qty": 250, "side": "buy", "limit_price": 3.05, "client_order_id": f"cx-{record['cycle_id']}-enter-SPY"}
    assert record["entry"]["premium"] == 3.05 * 250 * 100 and record["entry"]["thesis"] == "go"
    assert state.entries_today == 0  # a dry run does not consume the daily entry budget
    assert len((tmp_path / "convex.jsonl").read_text(encoding="utf-8").splitlines()) == 1


def test_execute_entry_submits_a_simple_buy_to_open_and_polls_the_fill():
    t = trading(order_statuses={"order-1": "filled"})
    record, state = cycle(t, FakeOptionDataClient(snapshots()), execute=True, transport=chat(ENTER_SPY))
    assert record["outcome"] == "submitted" and state.entries_today == 1
    request = t.submitted[0]
    assert isinstance(request, LimitOrderRequest) and request.symbol == CALL_774 and request.qty == 250
    assert request.limit_price == 3.05 and request.side == OrderSide.BUY and request.position_intent == PositionIntent.BUY_TO_OPEN
    assert request.order_class == OrderClass.SIMPLE and request.legs is None and request.client_order_id.startswith("cx-")
    assert record["entry"]["fill"] == "filled" and t.canceled == []


def test_unfilled_entry_is_canceled_after_the_timeout(monkeypatch):
    ticks = iter([0.0, 0.0, 100.0, 100.0])
    monkeypatch.setattr(convex.clock_time, "monotonic", lambda: next(ticks))
    t = trading(order_statuses={"order-1": "new"})
    record, _ = cycle(t, FakeOptionDataClient(snapshots()), execute=True, transport=chat(ENTER_SPY))
    assert record["entry"]["fill"] == "canceled_unfilled" and t.canceled == ["order-1"]


def test_take_profit_exit_sells_to_close_at_the_bid():
    t = trading(positions=[fake_position(CALL_774, 250, 3.0, side="long")])
    record, state = cycle(t, FakeOptionDataClient(snapshots(12.1, 12.3)), execute=True, transport=raising_transport())
    assert record["outcome"] == "submitted" and record["exits"][0]["reason"] == "take_profit"
    request = t.submitted[0]
    assert request.side == OrderSide.SELL and request.position_intent == PositionIntent.SELL_TO_CLOSE
    assert request.qty == 250 and request.limit_price == 12.1 and request.symbol == CALL_774
    assert state.last_exit_at == FRI


def test_stop_and_hold_marks():
    t = trading(positions=[fake_position(CALL_774, 250, 3.0, side="long")])
    record, _ = cycle(t, FakeOptionDataClient(snapshots(1.4, 1.5)), transport=raising_transport())
    assert record["exits"][0]["reason"] == "stop" and record["outcome"] == "planned"
    record, _ = cycle(trading(positions=[fake_position(CALL_774, 250, 3.0, side="long")]), FakeOptionDataClient(snapshots(3.1, 3.3)), transport=raising_transport())
    assert record["exits"][0]["reason"] is None and record["outcome"] == "holding" and record["exits"][0]["pnl_pct"] == pytest.approx(6.7, abs=0.1)


def test_time_exit_is_a_limit_then_a_market_order():
    held = [fake_position(CALL_774, 250, 3.0, side="long")]
    at_1045 = datetime(2026, 9, 4, 14, 45, tzinfo=UTC)
    t = trading(positions=held, clock=clock_at(at_1045))
    record, _ = cycle(t, FakeOptionDataClient(snapshots(stamp=at_1045)), execute=True, transport=raising_transport(), now=at_1045)
    assert record["phase"] == "time_exit" and record["exits"][0]["reason"] == "time"
    assert isinstance(t.submitted[0], LimitOrderRequest) and t.submitted[0].limit_price == 2.95
    at_1055 = datetime(2026, 9, 4, 14, 55, tzinfo=UTC)
    t = trading(positions=held, clock=clock_at(at_1055))
    record, _ = cycle(t, FakeOptionDataClient(snapshots(stamp=at_1055)), execute=True, transport=raising_transport(), now=at_1055)
    assert record["phase"] == "market_exit" and isinstance(t.submitted[0], MarketOrderRequest)


def test_flatten_now_forces_the_time_exit_inside_the_window():
    t = trading(positions=[fake_position(CALL_774, 250, 3.0, side="long")])
    record, _ = cycle(t, FakeOptionDataClient(snapshots(3.1, 3.3)), transport=raising_transport(), flatten_now=True)
    assert record["exits"][0]["reason"] == "time" and record["outcome"] == "planned"


def test_pending_order_skips_the_exit_and_blocks_entries():
    order = SimpleNamespace(id="o9", symbol=CALL_774, client_order_id="cx-20260904-133000-enter-SPY", legs=None)
    t = trading(positions=[fake_position(CALL_774, 250, 3.0, side="long")], orders=[order])
    record, _ = cycle(t, FakeOptionDataClient(snapshots(12.1, 12.3)), execute=True, transport=raising_transport())
    assert record["exits"][0]["skipped"] == "pending_order" and t.submitted == [] and record["outcome"] == "holding"


def test_gates_block_entries_without_asking_the_model():
    state = convex.ConvexState(last_exit_at=FRI - timedelta(seconds=60))
    record, _ = cycle(trading(), FakeOptionDataClient(snapshots()), transport=raising_transport(), state=state)
    assert record["outcome"] == "cooldown"
    state = convex.ConvexState(entries_today=2)
    record, _ = cycle(trading(), FakeOptionDataClient(snapshots()), transport=raising_transport(), state=state)
    assert record["outcome"] == "entry_cap"
    early = datetime(2026, 9, 4, 13, 20, tzinfo=UTC)
    record, _ = cycle(trading(clock=clock_at(early)), FakeOptionDataClient(snapshots()), transport=raising_transport(), now=early)
    assert record["outcome"] == "outside_window" and record["phase"] == "pre"
    record, _ = cycle(trading(clock=clock_at(FRI, is_open=False)), FakeOptionDataClient(snapshots()), transport=raising_transport())
    assert record["outcome"] == "market_closed"
    late = datetime(2026, 9, 4, 15, 0, tzinfo=UTC)
    record, _ = cycle(trading(clock=clock_at(late)), FakeOptionDataClient(snapshots()), transport=raising_transport(), now=late)
    assert record["outcome"] == "done"


def test_no_eligible_contract_skips_the_model():
    stale = snapshots(stamp=FRI - timedelta(minutes=5))
    record, _ = cycle(trading(), FakeOptionDataClient(stale), transport=raising_transport())
    assert record["outcome"] == "no_eligible_contract" and record["chain_eligible"] == []


def test_stale_recheck_rejects_the_entry():
    fresh, stale = snapshots(), snapshots(stamp=FRI - timedelta(minutes=5))
    t = trading()
    record, _ = cycle(t, FakeOptionDataClient([fresh, stale]), execute=True, transport=chat(ENTER_SPY))
    assert record["entry"]["rejected"] == "recheck: stale_quote" and t.submitted == [] and record["outcome"] == "hold"


def test_model_pass_is_journaled_with_its_thesis():
    record, _ = cycle(trading(), FakeOptionDataClient(snapshots()), transport=chat(json.dumps({"action": "pass", "thesis": "everything disagrees"})))
    assert record["outcome"] == "pass" and record["llm"]["choice"]["thesis"] == "everything disagrees"


# --- broker single-leg path ---

def plan(**overrides) -> SingleLegPlan:
    base = dict(kind="enter", symbol=CALL_774, underlying="SPY", qty=250, side="buy", intent="buy_to_open",
                limit_price=3.05, client_order_id="cx-1-enter-SPY")
    base.update(overrides)
    return SingleLegPlan(**base)


@pytest.mark.parametrize("bad", [
    dict(qty=0), dict(client_order_id="sp-1-enter-SPY"), dict(limit_price=None), dict(limit_price=-1.0),
    dict(intent="sell_to_open", side="sell"), dict(time_in_force="gtc"), dict(symbol="SPY"),
    dict(kind="exit", side="buy", intent="buy_to_open"),
])
def test_submit_single_leg_refuses_malformed_plans(bad):
    t = FakeTradingClient()
    with pytest.raises(broker.BrokerError):
        broker.submit_single_leg_order(t, plan(**bad))
    assert t.submitted == []


def test_submit_single_leg_market_exit_shape_and_refusal_receipt():
    t = FakeTradingClient()
    receipt = broker.submit_single_leg_order(t, plan(kind="exit", side="sell", intent="sell_to_close", limit_price=None, client_order_id="cx-1-exit-SPY-time"))
    assert receipt.submitted and receipt.order_id == "order-1" and isinstance(t.submitted[0], MarketOrderRequest)
    assert t.submitted[0].position_intent == PositionIntent.SELL_TO_CLOSE and t.submitted[0].order_class == OrderClass.SIMPLE
    refused = broker.submit_single_leg_order(FakeTradingClient(submit_error=RuntimeError("secret body")), plan())
    assert refused.submitted is False and refused.error == "RuntimeError"


def test_fetch_account_state_reads_cash_and_buying_power():
    state = broker.fetch_account_state(FakeTradingClient(account=fake_account(cash=1234.5, options_buying_power=999.0)), ("SPY",))
    assert state.cash == 1234.5 and state.options_buying_power == 999.0
    assert broker.fetch_account_state(FakeTradingClient(), ("SPY",)).cash is None


def test_fetch_contracts_window_parses_both_types_from_the_occ_symbol():
    rows = broker.fetch_contracts_window(trading(), "SPY", 774.0, EXP0, 4, 0.02)
    assert {r["type"] for r in rows} == {"C", "P"} and {r["expiration"] for r in rows} == {EXP0, EXP1}
    row = next(r for r in rows if r["symbol"] == CALL_774)
    assert row["strike"] == 774.0 and row["open_interest"] == 5000 and all(r["symbol"].startswith("SPY") for r in rows)
