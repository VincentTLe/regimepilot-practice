"""Convex mode: ONE long option (call or put) on SPY or QQQ, 0-4 DTE, bought with ~all cash.

    uv run --env-file .env convex.py brief [--ask]                  # the briefing the model sees; --ask also asks it
    uv run --env-file .env convex.py run [--execute] [--loop] [--flatten-now]

Tan's decision 2026-09-03: a lottery ticket with a signal, aimed at +300-500%
before the hackathon deadline (Fri 2026-09-04 11:00 ET). The LLM is the only
discretionary brain: from a briefing the code assembles (spot, 5-minute
indicators, tape, opening range, the near-the-money chain with live quotes) it
picks symbol, direction, expiry and strike, or passes. Deterministic code owns
everything else: paper-only (broker), the refusal to run next to positions or
orders it did not create, the trading window in Eastern time from Alpaca's
clock, cash-based sizing with a contract cap, quote validity, the mechanical
exits (take profit at TAKE_PROFIT_MULT x entry, stop at STOP_FRACTION x entry,
a forced sell-to-close from TIME_EXIT and a market-order fallback from
MARKET_EXIT), and one open position at a time.

Never run this together with `cli.py run` on the same account. Journal:
logs/convex.jsonl. Nothing here touches the spread engine's behaviour.
"""

from __future__ import annotations

import json
import math
import time as clock_time
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import typer
from loguru import logger

import broker
import cli
import decision_layer
import market_data
import options_screener
import pos_and_risk
import settings
import signals
import sounds
import tape
from data_models import AccountState, Config, ConvexChoice, LegQuote, OpenOption, SingleLegPlan, to_json_line

ET = ZoneInfo("America/New_York")
JOURNAL_PATH = Path("logs") / "convex.jsonl"
PREFIX = broker.CONVEX_PREFIX
ENTRY_FILL_TIMEOUT_SECONDS = 45
FILL_POLL_SECONDS = 2
MAX_ASK_DRIFT = 0.10  # refuse to chase a quote that moved more than this since the briefing
BAR_SECONDS = 300
TAPE_MINUTES = 15
TAPE_MIN_TRADES = 50
MIN_OPTIONS_LEVEL = 2  # buying a single long option needs level 2
LLM_MAX_TOKENS = 2000

app = typer.Typer(add_completion=False)

SYSTEM_PROMPT = """You are the only discretionary brain of a one-shot convex bet on a PAPER account.
The account buys ONE long option - a call or a put on one of the listed symbols -
with essentially all of its cash, and mechanical code then manages it: take
profit when the option's mark reaches take_profit_mult times the entry price,
stop when it falls to stop_fraction times the entry price, forced sell-to-close
at time_exit (Eastern). The goal is a multiple of the premium within about an
hour, so this only pays on a real directional move that starts now.

The briefing carries, per symbol: spot; session facts (prev_close, open, gap_pct,
change_pct, range_pct, minutes_since_open, first_bar_direction = the direction of
the first 5-minute bar, i.e. the opening range breakout signal); 5-minute
indicators (rsi, atr, macd_hist, ema_fast_dist / ema_slow_dist = last close minus
a fast/slow trend EMA, positive = above; events = gap/breakout/macd/tape events
on the latest bar); the tape (flow_imbalance = tick-rule buy volume minus sell
volume over their sum for the last minutes of prints, -1..+1; flow_trades = the
prints behind it; l1_imbalance = bid size vs ask size); and the chain: for each
expiration the nearest strikes on both sides of spot with bid, ask, mid,
spread_bps, iv, open_interest, quote_age_s, affordable_qty, and eligible /
reject (the code's quote-quality verdict).

How to choose: direction from agreement between the tape, the first-bar
direction, the EMA anchors and the latest events; symbol = the one where they
agree most clearly; strike ATM or one to two strikes out of the money in the
trade direction (delta plus liquidity); the same-day expiration for maximum
convexity when the move is happening now, the later expiration only when the
thesis needs more time. Choose ONLY rows with eligible: true. Enter unless
there is no eligible contract or the tape, the bars and the gap flatly
contradict each other - a pass is allowed but must state the contradiction.

Reply with strict JSON only:
{"action": "enter" | "pass", "symbol": "<one of the symbols>", "direction": "CALL" | "PUT",
 "expiration": "YYYY-MM-DD", "strike": <number>, "thesis": "<one sentence>"}"""


@dataclass
class ConvexState:
    last_exit_at: datetime | None = None
    entries_today: int = 0
    pending: dict[str, str] = field(default_factory=dict)  # order_id -> label
    cancel_requested: set[str] = field(default_factory=set)


# --- pure helpers -----------------------------------------------------------


def phase_at(now: time) -> str:
    """pre < entry_start | entry < entry_end | hold < time_exit | time_exit < market_exit | market_exit < session_end | done."""
    if now < settings.CONVEX_ENTRY_START:
        return "pre"
    if now < settings.CONVEX_ENTRY_END:
        return "entry"
    if now < settings.CONVEX_TIME_EXIT:
        return "hold"
    if now < settings.CONVEX_MARKET_EXIT:
        return "time_exit"
    if now < settings.CONVEX_SESSION_END:
        return "market_exit"
    return "done"


def owned_positions(account: AccountState, today: date) -> list[OpenOption]:
    """Long single legs on the convex symbols inside the expiry window: what this mode manages."""
    out = []
    for leg in account.legs:
        days = (leg.expiration - today).days
        if leg.qty > 0 and leg.underlying in settings.CONVEX_SYMBOLS and 0 <= days <= settings.CONVEX_MAX_EXPIRY_DAYS:
            out.append(OpenOption(symbol=leg.symbol, underlying=leg.underlying, expiration=leg.expiration,
                                  option_type=leg.option_type, strike=leg.strike, qty=leg.qty,
                                  avg_entry_price=leg.avg_entry_price))
    return out


def foreign_holdings(account: AccountState, open_client_ids: dict[str, str], today: date) -> list[str]:
    """Reasons the convex mode must NOT trade on this account right now (empty = clean).

    Any paired spread, any short leg, any long leg it would not manage, any
    unparsed position, and any open order without the `cx-` client id prefix.
    """
    reasons: list[str] = []
    spreads, _ = pos_and_risk.pair_spreads(list(account.legs))
    spread_legs = {s for spread in spreads for s in (spread.long_symbol, spread.short_symbol)}
    for spread in spreads:
        reasons.append(f"spread {spread.underlying} {spread.expiration} {spread.option_type} x{spread.qty}")
    owned = {o.symbol for o in owned_positions(account, today)}
    for leg in account.legs:
        if leg.symbol in spread_legs:
            continue
        if leg.qty < 0:
            reasons.append(f"short leg {leg.symbol}")
        elif leg.symbol not in owned:
            reasons.append(f"foreign long {leg.symbol}")
    reasons.extend(f"unparsed position {symbol}" for symbol in account.unparsed_positions)
    for order_id, client_id in open_client_ids.items():
        if not str(client_id or "").startswith(PREFIX):
            reasons.append(f"foreign order {client_id or order_id}")
    return reasons


def size_all_in(cash: float | None, options_bp: float | None, ask: float | None) -> tuple[int, str | None]:
    """Contracts bought with CASH_FRACTION of min(cash, options buying power), capped at MAX_CONTRACTS."""
    budgets = [value for value in (cash, options_bp) if value is not None]
    if not budgets:
        return 0, "unknown_cash"
    if ask is None or ask <= 0:
        return 0, "bad_ask"
    qty = min(math.floor(min(budgets) * settings.CONVEX_CASH_FRACTION / (ask * 100.0)), settings.CONVEX_MAX_CONTRACTS)
    return (qty, None) if qty >= 1 else (0, "insufficient_cash")


def exit_reason(position: OpenOption, mark: float | None, phase: str) -> str | None:
    """time in the exit phases (needs no mark); else take_profit / stop from the mark; None = hold."""
    if phase in ("time_exit", "market_exit", "done"):
        return "time"
    if position.avg_entry_price is None or mark is None:
        return None
    if mark >= settings.CONVEX_TAKE_PROFIT_MULT * position.avg_entry_price:
        return "take_profit"
    if mark <= settings.CONVEX_STOP_FRACTION * position.avg_entry_price:
        return "stop"
    return None


def check_contract(quote: LegQuote, server_time: datetime) -> str | None:
    """options_screener.check_leg with the convex thresholds and no IV requirement."""
    if quote.open_interest is not None and quote.open_interest < settings.CONVEX_MIN_OPEN_INTEREST:
        return "low_open_interest"
    if quote.bid is None or quote.ask is None or quote.quote_time is None:
        return "no_quote"
    if quote.bid <= 0 or quote.ask <= 0 or quote.bid > quote.ask:
        return "crossed_quote"
    age = (server_time - quote.quote_time).total_seconds()
    if age < -settings.CONVEX_MAX_QUOTE_AGE_SECONDS:
        return "future_quote"
    if age > settings.CONVEX_MAX_QUOTE_AGE_SECONDS:
        return "stale_quote"
    if options_screener.quote_spread_bps(quote) > settings.CONVEX_MAX_SPREAD_BPS:
        return "wide_spread"
    return None


def pick_strikes(rows: list[dict], spot: float) -> list[dict]:
    """Per (expiration, type): the STRIKES_EACH_SIDE nearest strikes at/below spot and above it."""
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        groups.setdefault((row["expiration"], row["type"]), []).append(row)
    n = settings.CONVEX_STRIKES_EACH_SIDE
    out: list[dict] = []
    for group in groups.values():
        below = sorted((r for r in group if r["strike"] <= spot), key=lambda r: -r["strike"])[:n]
        above = sorted((r for r in group if r["strike"] > spot), key=lambda r: r["strike"])[:n]
        out.extend(below + above)
    return sorted(out, key=lambda r: (r["expiration"], r["type"], r["strike"]))


def session_facts(frame: pd.DataFrame | None, now_et: datetime) -> dict:
    """Prev close, today's open/high/low/last, gap and range %, minutes since the open, first-bar direction."""
    if frame is None or frame.empty:
        return {}
    et_index = frame.index.tz_convert(ET)
    is_today = et_index.date == now_et.date()
    before, today = frame[~is_today], frame[is_today]
    facts: dict = {"prev_close": float(before["close"].iloc[-1]) if len(before) else None}
    minutes = et_index[is_today].hour * 60 + et_index[is_today].minute
    regular = today[minutes >= 9 * 60 + 30]
    if regular.empty:
        return facts
    first = regular.iloc[0]
    open_et = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
    facts.update({
        "open": float(first["open"]), "high": float(regular["high"].max()), "low": float(regular["low"].min()),
        "last": float(regular["close"].iloc[-1]),
        "first_bar_direction": "up" if first["close"] > first["open"] else ("down" if first["close"] < first["open"] else "flat"),
        "first_bar_high": float(first["high"]), "first_bar_low": float(first["low"]),
        "minutes_since_open": int((now_et - open_et).total_seconds() // 60),
    })
    if facts["prev_close"]:
        base = facts["prev_close"]
        facts["gap_pct"] = round(100 * (facts["open"] - base) / base, 2)
        facts["change_pct"] = round(100 * (facts["last"] - base) / base, 2)
        facts["range_pct"] = round(100 * (facts["high"] - facts["low"]) / base, 2)
    return facts


def parse_convex_choice(text: str, eligible: dict[tuple, str], model: str) -> ConvexChoice | None:
    """Strict validation: an entry must name an eligible (symbol, expiration, type, strike); garbage = None."""
    data = decision_layer._extract_json(text)
    if data is None:
        return None
    thesis = data.get("thesis") if isinstance(data.get("thesis"), str) else ""
    action = data.get("action")
    if action == "pass":
        return ConvexChoice(action="pass", symbol=None, direction=None, expiration=None, strike=None,
                            contract_symbol=None, thesis=thesis, model=model)
    if action != "enter":
        return None
    symbol = str(data.get("symbol") or "").upper()
    direction = data.get("direction")
    if symbol not in settings.CONVEX_SYMBOLS or direction not in ("CALL", "PUT"):
        return None
    try:
        expiration = date.fromisoformat(str(data.get("expiration")))
        strike = float(data.get("strike"))
    except (TypeError, ValueError):
        return None
    contract = eligible.get((symbol, expiration, "C" if direction == "CALL" else "P", round(strike, 3)))
    if contract is None:
        return None
    return ConvexChoice(action="enter", symbol=symbol, direction=direction, expiration=expiration, strike=strike,
                        contract_symbol=contract, thesis=thesis, model=model)


def build_entry_plan(contract_symbol: str, underlying: str, qty: int, ask: float, cycle_id: str) -> SingleLegPlan:
    return SingleLegPlan(kind="enter", symbol=contract_symbol, underlying=underlying, qty=qty, side="buy",
                         intent="buy_to_open", limit_price=round(ask, 2),
                         client_order_id=f"{PREFIX}{cycle_id}-enter-{underlying}")


def build_exit_plan(position: OpenOption, quote: LegQuote | None, cycle_id: str, reason: str,
                    *, market: bool = False) -> SingleLegPlan | None:
    """Sell-to-close at the bid, or a market order for the last-resort exit; None without a usable bid."""
    if market:
        limit = None
    else:
        if quote is None or quote.bid is None or quote.bid <= 0:
            return None
        limit = round(quote.bid, 2)
    return SingleLegPlan(kind="exit", symbol=position.symbol, underlying=position.underlying, qty=position.qty,
                         side="sell", intent="sell_to_close", limit_price=limit,
                         client_order_id=f"{PREFIX}{cycle_id}-exit-{position.underlying}-{reason}")


def eligible_index(chain: list[dict]) -> dict[tuple, str]:
    return {
        (row["underlying"], date.fromisoformat(row["expiration"]), "C" if row["type"] == "CALL" else "P", round(row["strike"], 3)): row["contract"]
        for row in chain
        if row["eligible"]
    }


# --- I/O ---------------------------------------------------------------------


def append_journal(record: dict) -> None:
    JOURNAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with JOURNAL_PATH.open("a", encoding="utf-8") as handle:
        handle.write(to_json_line(record) + "\n")


def _settle(trading: object, plan: SingleLegPlan, execute: bool) -> dict:
    if not execute:
        return {"submitted": False, "dry_run": True,
                "plan": {"kind": plan.kind, "symbol": plan.symbol, "qty": plan.qty, "side": plan.side,
                         "limit_price": plan.limit_price, "client_order_id": plan.client_order_id}}
    receipt = broker.submit_single_leg_order(trading, plan)
    logger.info("order {}: submitted={} status={} error={}", plan.client_order_id, receipt.submitted,
                receipt.status, receipt.error)
    if receipt.submitted:
        sounds.play_order_sound()
    return {"submitted": receipt.submitted, "order_id": receipt.order_id, "status": receipt.status,
            "error": receipt.error, "client_order_id": receipt.client_order_id}


def _await_fill(trading: object, order_id: str, timeout: float) -> str:
    """Poll until filled / dead / timeout; an unfilled entry is canceled so no bid rests at the open."""
    deadline = clock_time.monotonic() + timeout
    while True:
        status = broker.fetch_order_status(trading, order_id)
        if status in cli._FILLED:
            sounds.play_fill_sound()
            return "filled"
        if status in cli._DEAD:
            return status
        if clock_time.monotonic() >= deadline:
            try:
                broker.cancel_order(trading, order_id)
            except broker.BrokerError as error:
                logger.warning("cancel of unfilled entry failed: {}", error)
                return "unfilled_cancel_failed"
            return "canceled_unfilled"
        clock_time.sleep(FILL_POLL_SECONDS)


def chain_table(trading, option_data, symbol: str, spot: float, today: date, server_time: datetime,
                budget: float | None) -> list[dict]:
    """The near-the-money chain with live quotes and the code's eligibility verdict per row."""
    rows = broker.fetch_contracts_window(trading, symbol, spot, today, settings.CONVEX_MAX_EXPIRY_DAYS,
                                         settings.CONVEX_STRIKE_BAND_PCT)
    rows = pick_strikes(rows, spot)
    snapshots = broker.fetch_option_snapshots(option_data, [r["symbol"] for r in rows]) if rows else {}
    table = []
    for row in rows:
        quote = broker.leg_quote_from_snapshot(row["symbol"], row["strike"], snapshots.get(row["symbol"]), row["open_interest"])
        reject = check_contract(quote, server_time)
        qty, size_reason = size_all_in(budget, budget, quote.ask)
        two_sided = quote.bid is not None and quote.ask is not None and quote.bid > 0 and quote.ask > 0
        table.append({
            "contract": row["symbol"], "underlying": symbol, "expiration": row["expiration"].isoformat(),
            "dte": (row["expiration"] - today).days, "type": "CALL" if row["type"] == "C" else "PUT",
            "strike": row["strike"], "bid": quote.bid, "ask": quote.ask, "mid": quote.mid if two_sided else None,
            "spread_bps": round(options_screener.quote_spread_bps(quote)) if two_sided else None,
            "iv": quote.implied_vol, "open_interest": row["open_interest"],
            "quote_age_s": round((server_time - quote.quote_time).total_seconds()) if quote.quote_time else None,
            "affordable_qty": qty, "eligible": reject is None and qty >= 1, "reject": reject or size_reason,
        })
    return table


def build_briefing(config: Config, trading, stock_data, option_data, clock, account: AccountState) -> dict:
    now_et = clock.server_time.astimezone(ET)
    today = now_et.date()
    budget = min(v for v in (account.cash, account.options_buying_power) if v is not None) \
        if account.cash is not None or account.options_buying_power is not None else None
    symbols = list(settings.CONVEX_SYMBOLS)
    quotes = broker.fetch_spot_quotes(stock_data, tuple(symbols))
    try:
        trades = broker.fetch_recent_trades(stock_data, tuple(symbols), TAPE_MINUTES, clock.server_time)
    except broker.BrokerError as error:
        logger.warning("trades read failed, tape unknown: {}", error)
        trades = {}
    briefing: dict = {
        "now_et": now_et.strftime("%Y-%m-%d %H:%M"), "entry_window_end": settings.CONVEX_ENTRY_END.strftime("%H:%M"),
        "time_exit": settings.CONVEX_TIME_EXIT.strftime("%H:%M"), "cash": account.cash,
        "options_buying_power": account.options_buying_power,
        "sizing": {"cash_fraction": settings.CONVEX_CASH_FRACTION, "max_contracts": settings.CONVEX_MAX_CONTRACTS},
        "exits": {"take_profit_mult": settings.CONVEX_TAKE_PROFIT_MULT, "stop_fraction": settings.CONVEX_STOP_FRACTION},
        "symbols": {},
    }
    for symbol in symbols:
        quote = quotes.get(symbol)
        spot = quote.mid if quote is not None else None
        entry: dict = {"spot": spot}
        try:
            frame = signals.add_indicators(market_data.fetch_ohlcv(stock_data, symbol, "5m", clock.server_time))
            flow = tape.tick_rule(trades.get(symbol, []), TAPE_MIN_TRADES)
            l1 = tape.l1_imbalance(quote.bid_size, quote.ask_size) if quote is not None else None
            features = signals.build_signal(symbol, frame, spot, clock.server_time, BAR_SECONDS, flow=flow, l1_imbalance=l1)
            entry["session"] = session_facts(frame, now_et)
            entry["indicators"] = {"rsi": features.rsi, "atr": features.atr, "macd_hist": features.macd_hist,
                                   "ema_fast_dist": features.ema_fast_dist, "ema_slow_dist": features.ema_slow_dist,
                                   "events": [f"{e.kind}:{e.direction}" for e in features.events]}
            entry["tape"] = {"flow_imbalance": features.flow_imbalance, "flow_trades": features.flow_trades,
                             "l1_imbalance": features.l1_imbalance}
        except Exception as error:  # noqa: BLE001 - a data gap is reported to the model, never invented
            logger.warning("{}: bars/tape unavailable ({})", symbol, type(error).__name__)
            entry["indicators"] = None
        if spot is not None:
            try:
                entry["chain"] = chain_table(trading, option_data, symbol, spot, today, clock.server_time, budget)
            except broker.BrokerError as error:
                logger.warning("{}: chain unavailable ({})", symbol, error)
                entry["chain"] = []
        else:
            entry["chain"] = []
        briefing["symbols"][symbol] = entry
    return briefing


def ask_model(briefing: dict, api_key: str, transport=None) -> tuple[ConvexChoice | None, str, str]:
    """(choice, raw content, model). LlmError propagates."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": json.dumps(briefing)}]
    content, model = decision_layer.call_llm(messages, api_key, transport=transport, max_tokens=LLM_MAX_TOKENS)
    eligible = eligible_index([row for entry in briefing["symbols"].values() for row in entry.get("chain", [])])
    return parse_convex_choice(content, eligible, model), content, model


def run_cycle(config: Config, trading, stock_data, option_data, *, execute: bool, state: ConvexState,
              llm_transport=None, flatten_now: bool = False) -> dict:
    started = datetime.now(timezone.utc)
    cycle_id = started.strftime("%Y%m%d-%H%M%S")
    record: dict = {"cycle_id": cycle_id, "started_at": started, "dry_run": not execute, "mode": "convex"}
    try:
        clock = broker.fetch_clock(trading)
        account = broker.fetch_account_state(trading, tuple(settings.CONVEX_SYMBOLS))
        open_ids = broker.fetch_open_client_ids(trading)
    except broker.BrokerError as error:
        logger.error("cycle aborted: {}", error)
        record.update(outcome="error", error=str(error))
        append_journal(record)
        return record
    today = clock.server_time.astimezone(ET).date()
    foreign = foreign_holdings(account, open_ids, today)
    record.update(equity=account.equity, cash=account.cash, options_buying_power=account.options_buying_power,
                  market_open=clock.is_open)
    if foreign:
        for reason in foreign:
            logger.error("refusing to trade next to: {}", reason)
        record.update(outcome="foreign_positions", foreign=foreign)
        append_journal(record)
        return record
    if not clock.is_open:
        logger.info("market closed; nothing to do")
        record["outcome"] = "market_closed"
        append_journal(record)
        return record
    now_et = clock.server_time.astimezone(ET)
    phase = "time_exit" if flatten_now else phase_at(now_et.time())
    record["phase"] = phase
    held = owned_positions(account, today)
    record["held"] = [{"symbol": p.symbol, "qty": p.qty, "entry": p.avg_entry_price} for p in held]

    # --- manage what we hold: exits always run before any entry ---
    exits: list[dict] = []
    record["exits"] = exits
    if held:
        snapshots = broker.fetch_option_snapshots(option_data, [p.symbol for p in held])
        for position in held:
            quote = broker.leg_quote_from_snapshot(position.symbol, position.strike, snapshots.get(position.symbol), None)
            two_sided = quote.bid is not None and quote.ask is not None and quote.bid > 0 and quote.ask > 0
            mark = quote.mid if two_sided else None
            reason = exit_reason(position, mark, phase)
            pnl_pct = None if mark is None or not position.avg_entry_price else round(100 * (mark / position.avg_entry_price - 1), 1)
            entry: dict = {"symbol": position.symbol, "qty": position.qty, "entry": position.avg_entry_price,
                           "mark": mark, "bid": quote.bid, "pnl_pct": pnl_pct, "reason": reason}
            if reason is None:
                logger.info("{} x{} mark {} ({}{}%) hold", position.symbol, position.qty, mark,
                            "+" if (pnl_pct or 0) >= 0 else "", pnl_pct)
                exits.append(entry)
                continue
            if position.symbol in account.open_order_symbols:
                entry["skipped"] = "pending_order"
                if phase == "market_exit":
                    for order_id, client_id in open_ids.items():
                        if order_id not in state.cancel_requested:
                            try:
                                broker.cancel_order(trading, order_id)
                                state.cancel_requested.add(order_id)
                                entry["canceled"] = client_id
                            except broker.BrokerError as error:
                                logger.warning("cancel failed: {}", error)
            else:
                plan = build_exit_plan(position, quote, cycle_id, reason, market=(phase == "market_exit"))
                if plan is None:
                    entry["skipped"] = "no_quote"
                else:
                    entry["receipt"] = _settle(trading, plan, execute)
                    if entry["receipt"].get("submitted") or entry["receipt"].get("dry_run"):
                        state.last_exit_at = clock.server_time
            logger.info("exit {} {}: {}", position.symbol, reason, entry.get("receipt", entry.get("skipped")))
            exits.append(entry)

    # --- entry: only flat, no open orders, inside the window, past the cooldown, under the daily cap ---
    gate = None
    if held:
        gate = "holding"
    elif account.open_order_symbols:
        gate = "pending_order"
    elif phase != "entry":
        gate = "outside_window" if phase != "done" else "done"
    elif state.last_exit_at is not None and (clock.server_time - state.last_exit_at).total_seconds() < settings.CONVEX_COOLDOWN_SECONDS:
        gate = "cooldown"
    elif state.entries_today >= settings.CONVEX_MAX_ENTRIES_PER_DAY:
        gate = "entry_cap"
    if gate is not None:
        record["outcome"] = gate if not exits or gate != "holding" else ("submitted" if any((e.get("receipt") or {}).get("submitted") for e in exits) else ("planned" if any((e.get("receipt") or {}).get("dry_run") for e in exits) else "holding"))
        append_journal(record)
        return record

    briefing = build_briefing(config, trading, stock_data, option_data, clock, account)
    record["briefing"] = {sym: {k: v for k, v in entry.items() if k != "chain"} for sym, entry in briefing["symbols"].items()}
    record["chain_eligible"] = [row["contract"] for entry in briefing["symbols"].values() for row in entry.get("chain", []) if row["eligible"]]
    if not record["chain_eligible"]:
        logger.info("no eligible contract this cycle")
        record["outcome"] = "no_eligible_contract"
        append_journal(record)
        return record
    if not config.llm_api_key:
        record["outcome"] = "no_llm_key"
        append_journal(record)
        return record
    try:
        choice, raw, model = ask_model(briefing, config.llm_api_key, transport=llm_transport)
    except decision_layer.LlmError as error:
        logger.error("LLM decision failed, no bet: {}", error)
        record["outcome"] = "llm_error"
        append_journal(record)
        return record
    record["llm"] = {"model": model, "raw": raw[:1500], "choice": None if choice is None else choice.__dict__}
    if choice is None or choice.action != "enter":
        logger.info("decider {}: {}", "passed" if choice else "answered garbage", choice.thesis if choice else raw[:200])
        record["outcome"] = "pass"
        append_journal(record)
        return record
    logger.info("bet: {} {} {} {} ({}): {}", choice.symbol, choice.direction, choice.expiration, choice.strike, model, choice.thesis)

    # --- pre-submit recheck on fresh data, then size on cash and submit at the ask ---
    entry: dict = {"symbol": choice.symbol, "direction": choice.direction, "contract": choice.contract_symbol,
                   "expiration": choice.expiration, "strike": choice.strike, "thesis": choice.thesis, "model": model}
    record["entry"] = entry
    briefed = next(row for e in briefing["symbols"].values() for row in e.get("chain", []) if row["contract"] == choice.contract_symbol)
    try:
        fresh_clock = broker.fetch_clock(trading)
        fresh_account = broker.fetch_account_state(trading, tuple(settings.CONVEX_SYMBOLS))
        fresh_ids = broker.fetch_open_client_ids(trading)
        snapshot = broker.fetch_option_snapshots(option_data, [choice.contract_symbol]).get(choice.contract_symbol)
    except broker.BrokerError as error:
        entry["rejected"] = f"recheck: {error}"
        record["outcome"] = "hold"
        append_journal(record)
        return record
    quote = broker.leg_quote_from_snapshot(choice.contract_symbol, choice.strike, snapshot, briefed["open_interest"])
    failure = check_contract(quote, fresh_clock.server_time)
    if failure is None and phase_at(fresh_clock.server_time.astimezone(ET).time()) != "entry":
        failure = "window_closed"
    if failure is None and (owned_positions(fresh_account, today) or fresh_account.open_order_symbols or foreign_holdings(fresh_account, fresh_ids, today)):
        failure = "account_changed"
    if failure is None and briefed["ask"] and quote.ask > briefed["ask"] * (1 + MAX_ASK_DRIFT):
        failure = "ask_ran_away"
    if failure is None and execute and (fresh_account.options_level or 0) < MIN_OPTIONS_LEVEL:
        failure = "options_level_too_low"
    if failure is not None:
        entry["rejected"] = f"recheck: {failure}"
        logger.info("entry refused at recheck: {}", failure)
        record["outcome"] = "hold"
        append_journal(record)
        return record
    qty, size_reason = size_all_in(fresh_account.cash, fresh_account.options_buying_power, quote.ask)
    if qty < 1:
        entry["rejected"] = f"size: {size_reason}"
        record["outcome"] = "hold"
        append_journal(record)
        return record
    plan = build_entry_plan(choice.contract_symbol, choice.symbol, qty, quote.ask, cycle_id)
    entry.update(qty=qty, limit_price=plan.limit_price, premium=round(plan.limit_price * qty * 100, 2))
    entry["receipt"] = _settle(trading, plan, execute)
    if entry["receipt"].get("submitted"):
        state.entries_today += 1
        order_id = entry["receipt"].get("order_id")
        if order_id:
            entry["fill"] = _await_fill(trading, order_id, ENTRY_FILL_TIMEOUT_SECONDS)
            entry["fill_status"], entry["fill_price"] = broker.fetch_order_fill(trading, order_id)
            logger.info("entry {}: {} at {}", choice.contract_symbol, entry["fill"], entry["fill_price"])
        record["outcome"] = "submitted"
    else:
        record["outcome"] = "planned" if entry["receipt"].get("dry_run") else "hold"
    append_journal(record)
    return record


def _safe_cycle(*args, **kwargs) -> dict:
    try:
        return run_cycle(*args, **kwargs)
    except KeyboardInterrupt:
        raise
    except Exception as error:  # noqa: BLE001 - the loop must survive anything
        tb = error.__traceback__
        while tb is not None and tb.tb_next is not None:
            tb = tb.tb_next
        where = f"{Path(tb.tb_frame.f_code.co_filename).name}:{tb.tb_lineno}" if tb else "?"
        logger.error("cycle crashed: {} at {}", type(error).__name__, where)
        started = datetime.now(timezone.utc)
        record = {"cycle_id": started.strftime("%Y%m%d-%H%M%S"), "started_at": started, "mode": "convex",
                  "outcome": "error", "error": type(error).__name__, "where": where}
        append_journal(record)
        return record


def _new_orders(record: dict) -> dict[str, str]:
    pending = {}
    for item in [record.get("entry")] + list(record.get("exits") or []):
        receipt = (item or {}).get("receipt") or {}
        if receipt.get("submitted") and receipt.get("order_id"):
            pending[receipt["order_id"]] = f"{'exit' if 'reason' in item else 'entry'} {item.get('symbol')}"
    return pending


@app.command()
def brief(ask: bool = typer.Option(False, "--ask", help="Also ask the model and print its raw reply + parsed choice.")) -> None:
    """Print the briefing the model would see right now. Never trades."""
    cli.setup_logging()
    config, trading, stock_data, option_data = cli._bootstrap()
    clock = broker.fetch_clock(trading)
    account = broker.fetch_account_state(trading, tuple(settings.CONVEX_SYMBOLS))
    briefing = build_briefing(config, trading, stock_data, option_data, clock, account)
    typer.echo(json.dumps(briefing, indent=1, default=str))
    if ask:
        if not config.llm_api_key:
            typer.echo("FEATHERLESS_API_KEY missing")
            raise typer.Exit(1)
        choice, raw, model = ask_model(briefing, config.llm_api_key)
        typer.echo(f"\n--- {model} raw reply ---\n{raw}\n--- parsed ---\n{choice}")


@app.command()
def run(
    execute: bool = typer.Option(False, help="Actually submit paper orders (dry run otherwise)."),
    loop: bool = typer.Option(False, help="Run until session_end."),
    interval: int = typer.Option(settings.CONVEX_LOOP_INTERVAL_SECONDS, help="Seconds between cycles with --loop."),
    flatten_now: bool = typer.Option(False, "--flatten-now", help="Force the time exit this cycle (manual bail-out)."),
) -> None:
    """The all-in single-option loop. Paper only; dry run unless --execute."""
    cli.setup_logging(file_sink=True)
    config, trading, stock_data, option_data = cli._bootstrap()
    if not config.llm_api_key:
        typer.echo("FEATHERLESS_API_KEY missing: the convex mode has no manual decider")
        raise typer.Exit(1)
    try:
        model, seconds = decision_layer.ping(config.llm_api_key)
    except decision_layer.LlmError as error:
        typer.echo(f"LLM check failed: {error}")
        raise typer.Exit(1)
    logger.info("LLM ready: {} answered in {:.1f}s", model, seconds)
    try:
        account = broker.fetch_account_state(trading, tuple(settings.CONVEX_SYMBOLS))
        clock = broker.fetch_clock(trading)
        foreign = foreign_holdings(account, broker.fetch_open_client_ids(trading), clock.server_time.astimezone(ET).date())
    except broker.BrokerError as error:
        typer.echo(f"startup check failed: {error}")
        raise typer.Exit(1)
    if foreign:
        typer.echo("refusing to start next to positions/orders this mode did not create:\n  " + "\n  ".join(foreign))
        raise typer.Exit(1)
    logger.info("foreign check: clean (cash {} options_bp {})", account.cash, account.options_buying_power)
    if execute:
        logger.warning("ARMED: paper order submission is enabled (all-in convex mode)")
    state = ConvexState()
    try:
        while True:
            cycle_started = clock_time.monotonic()
            record = _safe_cycle(config, trading, stock_data, option_data, execute=execute, state=state, flatten_now=flatten_now)
            logger.info("cycle {} outcome: {}", record["cycle_id"], record.get("outcome"))
            state.pending.update(_new_orders(record))
            cli._check_fills(trading, state.pending)
            if not loop or record.get("outcome") == "done":
                if state.pending:
                    logger.info("open orders still working: {}", ", ".join(state.pending.values()))
                break
            clock_time.sleep(max(0.0, interval - (clock_time.monotonic() - cycle_started)))
    except KeyboardInterrupt:
        logger.warning("stopped by user; an open position is NOT closed by this - run with --flatten-now to close it")


if __name__ == "__main__":
    app()
