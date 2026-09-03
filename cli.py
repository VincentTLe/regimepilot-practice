"""Typer CLI + the cycle engine that wires the diagram together.

Entry signal + option screener -> risk manager -> execution -> account state
-> position manager. Dry run is the default; --execute is the only way an
order reaches Alpaca (paper endpoint, enforced in broker.py).

Run as: uv run --env-file .env cli.py <command>
"""

from __future__ import annotations

import math
import os
import sys
import time
from collections import deque
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

import typer
from loguru import logger

import broker
import dashboard
import decision_layer
import market_data
import options_screener
import pos_and_risk
import settings
import signals
import sounds
import tape
from data_models import (
    Config,
    EntryChoice,
    OpenSpread,
    OrderPlan,
    SpreadQuote,
    SymbolFeatures,
    journal_entries,
    to_json_line,
)

JOURNAL_PATH = Path("logs") / "cycles.jsonl"
MIN_OPTIONS_LEVEL = 3  # spreads need Alpaca options trading level 3
MAX_DECISIONS_PER_CYCLE = 3  # LLM calls per cycle: keeps the worst-case wall-clock under the interval

app = typer.Typer(add_completion=False, no_args_is_help=True)


def setup_logging(file_sink: bool = False) -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <7}</level> | <level>{message}</level>",
    )
    if file_sink:  # the continuous loop keeps a daily file (logs/ is git-ignored)
        logger.add(
            Path("logs") / "paca_{time:YYYYMMDD}.log",
            level="INFO",
            rotation="1 day",
            retention="14 days",
            encoding="utf-8",
        )


def append_journal(record: dict) -> None:
    JOURNAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with JOURNAL_PATH.open("a", encoding="utf-8") as handle:
        handle.write(to_json_line(record) + "\n")


def _bootstrap() -> tuple[Config, object, object, object]:
    config = broker.load_config()
    trading, stock_data, option_data = broker.build_clients(config)
    return config, trading, stock_data, option_data


def _screen_spread(
    trading: object,
    option_data: object,
    underlying: str,
    direction: str,
    spot: float,
    clock_time: datetime,
    today,
    exclude_symbols: frozenset[str] = frozenset(),
) -> tuple[SpreadQuote | None, dict]:
    """Fetch chains + snapshots for one underlying and pick the best spread
    across the nearest settings.EXPIRIES_TO_SCREEN eligible expiries.

    `exclude_symbols` are legs we already hold on this underlying. An add must
    never touch them: Alpaca nets positions per contract, so buying a strike we
    are short would shrink the held leg, leave the held spread unpaired, and
    put it beyond the stop/take-profit manager.
    """
    by_expiry = broker.fetch_contracts(trading, underlying, direction, spot, today)
    if exclude_symbols:
        by_expiry = {
            exp: {k: info for k, info in chain.items() if info["symbol"] not in exclude_symbols}
            for exp, chain in by_expiry.items()
        }
    expirations = options_screener.pick_expirations(
        options_screener.liquid_expirations(by_expiry, spot), today
    )
    if not expirations:
        return None, {"no_expiration": 1}
    symbols = [
        info["symbol"] for exp in expirations for info in by_expiry[exp].values()
    ]
    snapshots = broker.fetch_option_snapshots(option_data, symbols)
    chains = {
        exp: {
            strike: broker.leg_quote_from_snapshot(
                info["symbol"],
                strike,
                snapshots.get(info["symbol"]),
                info["open_interest"],
            )
            for strike, info in by_expiry[exp].items()
        }
        for exp in expirations
    }
    return options_screener.select_spread(
        chains, direction, spot, underlying, clock_time
    )


def run_cycle(
    config: Config,
    trading: object,
    stock_data: object,
    option_data: object,
    *,
    execute: bool,
    manual_mode: bool,
    llm_transport: object | None = None,
    flow_state: dict[str, tape.TapeState] | None = None,
) -> dict:
    """One full cycle. Returns the journal record (also appended to the journal).

    `flow_state` keeps, per underlying, the last FLOW_EXIT_BARS tape readings and
    a pending-reversal counter across cycles (the loop passes one dict for its
    lifetime). A reversal exit needs the opposing event AND the tape against the
    spread for FLOW_EXIT_BARS consecutive cycles; the event is a one-bar pulse,
    so a held-back reversal stays armed for FLOW_EXIT_BARS more cycles. Without
    state (single-shot runs) the event-only rule applies.
    """
    started = datetime.now(timezone.utc)
    cycle_id = started.strftime("%Y%m%d-%H%M%S")
    record: dict = {"cycle_id": cycle_id, "started_at": started, "dry_run": not execute}

    try:
        clock = broker.fetch_clock(trading)
        account = broker.fetch_account_state(trading, config.symbols)
    except broker.BrokerError as error:
        logger.error("cycle aborted: {}", error)
        record["outcome"] = "error"
        record["error"] = str(error)
        append_journal(record)
        return record

    spreads, warnings = pos_and_risk.pair_spreads(account.legs)
    open_risk = pos_and_risk.open_premium_at_risk(spreads)
    record.update(
        {
            "market_open": clock.is_open,
            "equity": account.equity,
            "options_level": account.options_level,
            "open_spreads": [
                f"{s.underlying} {s.expiration} {s.option_type} x{s.qty}"
                for s in spreads
            ],
            "open_risk": open_risk,
            "warnings": warnings
            + [f"unparsed position: {p}" for p in account.unparsed_positions]
            + pos_and_risk.over_cap_warnings(spreads, account.equity),
        }
    )
    for warning in record["warnings"]:
        logger.warning(warning)

    if not clock.is_open:
        logger.info("market closed; nothing to do")
        record["outcome"] = "market_closed"
        append_journal(record)
        return record

    # --- Trading signals: needed by the reversal exit AND the entry side, so they
    # cover the whitelist plus every held underlying (even one removed from the list)
    watch_symbols = tuple(
        dict.fromkeys(config.symbols + tuple(s.underlying for s in spreads))
    )
    try:
        quotes = broker.fetch_spot_quotes(stock_data, watch_symbols)
        mids = {symbol: quote.mid for symbol, quote in quotes.items()}
    except broker.BrokerError as error:
        # Exits must still run on a quote outage; entries will gate out naturally.
        logger.error("quote read failed, exits still run, entries blocked: {}", error)
        quotes = {}
        mids = {symbol: None for symbol in watch_symbols}
    try:
        trades = broker.fetch_recent_trades(
            stock_data, watch_symbols, settings.FLOW_LOOKBACK_MINUTES, clock.server_time
        )
    except broker.BrokerError as error:
        # The tape is a confirmation, never a substitute: unknown flow gates entries
        # (flow_unknown) and leaves the reversal exit on the event-only rule.
        logger.warning("trades read failed, tape unknown this cycle: {}", error)
        trades = {}
    features = _build_trading_signals(
        watch_symbols, config, stock_data, mids, clock.server_time, trades=trades, quotes=quotes
    )

    # --- Position manager: mechanical exits run before entries and are never gated ---
    exits: list[dict] = []
    exiting: set[str] = set()  # underlyings with an exit this cycle: never add to those
    flow_holds: list[str] = []  # reversals held back because the tape lacks conviction
    tape_updated: set[str] = set()  # one tape reading per underlying per cycle
    if spreads:
        leg_symbols = [
            s for spread in spreads for s in (spread.long_symbol, spread.short_symbol)
        ]
        try:
            snapshots = broker.fetch_option_snapshots(option_data, leg_symbols)
        except broker.BrokerError as error:
            logger.error("exit snapshot read failed: {}", error)
            snapshots = {}
        for spread in spreads:
            long_q = broker.leg_quote_from_snapshot(
                spread.long_symbol, 0.0, snapshots.get(spread.long_symbol), None
            )
            short_q = broker.leg_quote_from_snapshot(
                spread.short_symbol, 0.0, snapshots.get(spread.short_symbol), None
            )
            symbol_features = features.get(spread.underlying)
            opposing_event = symbol_features is not None and pos_and_risk.opposing_event_fired(
                spread, symbol_features.events
            )
            flow_now = symbol_features.flow_imbalance if symbol_features is not None else None
            against = None
            streak = 0
            armed = opposing_event
            state = None
            if flow_state is not None:
                state = flow_state.get(spread.underlying)
                if state is None:
                    state = flow_state[spread.underlying] = tape.TapeState(
                        readings=deque(maxlen=settings.FLOW_EXIT_BARS)
                    )
                if spread.underlying not in tape_updated:  # once per underlying per cycle
                    state.readings.append(flow_now)
                    tape_updated.add(spread.underlying)
                    if opposing_event:
                        state.pending_reversal = settings.FLOW_EXIT_BARS  # arm for a few cycles
                    elif state.pending_reversal > 0:
                        state.pending_reversal -= 1  # another cycle spent waiting for the tape
                armed = opposing_event or state.pending_reversal > 0
                against = tape.flow_against(
                    spread.option_type, list(state.readings), settings.FLOW_EXIT_BARS, settings.FLOW_MIN_IMBALANCE
                )
                streak = tape.opposing_streak(spread.option_type, list(state.readings), settings.FLOW_MIN_IMBALANCE)
            # Reversal needs the event (or a still-armed one) AND, when required, the
            # tape against the spread for FLOW_EXIT_BARS cycles. Unknown tape (None)
            # or no state at all falls back to the event-only rule: never hold a
            # position on missing data.
            opposing = armed and (
                not settings.REVERSAL_NEEDS_FLOW or flow_state is None or against is None or against
            )
            # Trailing exit memory: the highest net mark seen while the loop watched
            # this spread (per process; a restart starts the peak afresh).
            peak_mark = None
            if state is not None:
                spread_key = f"{spread.long_symbol}/{spread.short_symbol}"
                mark_now = pos_and_risk.net_mark(long_q, short_q)
                if mark_now is not None:
                    previous = state.peak_marks.get(spread_key)
                    state.peak_marks[spread_key] = mark_now if previous is None else max(previous, mark_now)
                peak_mark = state.peak_marks.get(spread_key)
            if opposing and state is not None:
                state.pending_reversal = 0
            if armed and not opposing:
                note = (
                    f"{spread.underlying} {spread.expiration} {spread.option_type}: "
                    f"opposing event, tape streak {streak}/{settings.FLOW_EXIT_BARS}"
                )
                flow_holds.append(note)
                logger.info("reversal held back: {}", note)
            decision = pos_and_risk.exit_decision(
                spread,
                long_q,
                short_q,
                clock.server_time.date(),
                opposing_event=opposing,
                peak_mark=peak_mark,
            )
            if decision is None:
                if spread.net_entry_debit is None:
                    logger.warning(
                        "cannot compute stop/TP for {} (unknown entry debit)",
                        spread.underlying,
                    )
                continue
            entry: dict = {
                "spread": f"{spread.underlying} {spread.expiration} {spread.option_type}",
                "reason": decision.reason,
                "net_mark": decision.net_mark,
                "flow_imbalance": flow_now,
                "flow_against": against,
                "peak_mark": peak_mark,
            }
            if {spread.long_symbol, spread.short_symbol} & account.open_order_symbols:
                entry["skipped"] = "pending_order"
            else:
                plan = options_screener.build_exit_plan(
                    spread, long_q, short_q, cycle_id
                )
                if plan is None:
                    entry["skipped"] = "no_quote"
                else:
                    entry["receipt"] = _settle(trading, plan, execute)
            exits.append(entry)
            exiting.add(spread.underlying)
            logger.info(
                "exit {}: {}",
                entry["spread"],
                entry.get("receipt", entry.get("skipped")),
            )
    record["exits"] = exits
    record["flow_holds"] = flow_holds

    # --- Entry candidates: whitelist symbols only ---
    whitelist_features = {symbol: features[symbol] for symbol in config.symbols}
    pending = {
        pos_and_risk.parse_occ(sym)[0]
        for sym in account.open_order_symbols
        if pos_and_risk.parse_occ(sym) is not None
    }
    held_by_underlying: dict[str, list[OpenSpread]] = {}
    for spread in spreads:
        held_by_underlying.setdefault(spread.underlying, []).append(spread)
    candidates = []
    for c in signals.build_candidates(
        whitelist_features, clock.is_open, config.bar_seconds
    ):
        if c.gate_block is None:
            c = _gate_held(
                c,
                held_by_underlying.get(c.symbol, []),
                pending=c.symbol in pending,
                exiting=c.symbol in exiting,
            )
        candidates.append(c)
    record["candidates"] = [
        {
            "symbol": c.symbol,
            "mid": c.mid,  # journaled so the post-close review can grade decisions against later prices
            "events": [e.kind for e in c.events],
            "rsi": c.rsi,
            "atr": c.atr,
            "macd_hist": c.macd_hist,
            "ema_fast_dist": c.ema_fast_dist,
            "ema_slow_dist": c.ema_slow_dist,
            "flow_imbalance": c.flow_imbalance,
            "flow_trades": c.flow_trades,
            "l1_imbalance": c.l1_imbalance,
            "held": c.held,
            "gate_block": c.gate_block,
        }
        for c in candidates
    ]
    tradeable = [c for c in candidates if c.gate_block is None]
    logger.info("candidates passing gates: {}", [c.symbol for c in tradeable] or "none")

    # --- Decision + screener + risk + execution ---
    # One decision at a time; ask again with the remaining candidates until the
    # cycle has placed floor(per_cycle / per_entry) entries (2 with the shipped
    # settings), the decider passes, or candidates run out. Rejected attempts
    # (no spread, risk caps, recheck) consume their symbol but not a slot.
    max_entries = max(
        1, math.floor(settings.PER_CYCLE_FRACTION / settings.PER_ENTRY_FRACTION)
    )
    entries: list[dict] = []
    record["entries"] = entries
    cycle_spent = 0.0  # premium committed by earlier entries in this cycle
    planned = 0
    decisions = 0
    remaining = list(tradeable)
    while remaining and planned < max_entries and decisions < MAX_DECISIONS_PER_CYCLE:
        choice = _decide(remaining, config, manual_mode, llm_transport)
        decisions += 1
        if choice is None:
            break
        remaining = [c for c in remaining if c.symbol != choice.symbol]
        held = held_by_underlying.get(choice.symbol, [])
        if held and choice.direction != pos_and_risk.held_direction(held):
            # Deterministic guard: an add must follow the held spread's direction,
            # whatever the decider (LLM or human) replied.
            entry = {
                "symbol": choice.symbol,
                "direction": choice.direction,
                "thesis": choice.thesis,
                "model": choice.model,
                "rejected": "opposes_held_spread",
            }
            logger.info(
                "entry refused: {} {} opposes the held spread", choice.symbol, choice.direction
            )
        else:
            entry = _attempt_entry(
                choice,
                features[choice.symbol].mid,
                config,
                trading,
                option_data,
                account.equity,
                open_risk,
                pos_and_risk.open_premium_at_risk(held),
                account.open_order_symbols,
                cycle_id,
                execute,
                cycle_spent=cycle_spent,
                exclude_symbols=frozenset(
                    leg for s in held for leg in (s.long_symbol, s.short_symbol)
                ),
            )
        entries.append(entry)
        receipt = entry.get("receipt") or {}
        if receipt.get("submitted") or receipt.get("dry_run"):
            cycle_spent += entry["premium"]
            planned += 1

    submitted = any(
        (e.get("receipt") or {}).get("submitted") for e in exits + entries
    )
    record["outcome"] = (
        "submitted"
        if submitted
        else ("planned" if not execute and (exits or entries) else "hold")
    )
    append_journal(record)
    return record


def _gate_held(
    c: SymbolFeatures, held: list[OpenSpread], *, pending: bool, exiting: bool
) -> SymbolFeatures:
    """Entry gates for an underlying we already hold or have an open order on.

    allow_stacking off: any held or pending underlying is out (already_held).
    allow_stacking on: a further entry is allowed only as an ADD in the held
    spread's direction — a pending order or a same-cycle exit still blocks,
    events against the held direction are dropped, and the candidate carries
    the held direction so the decider knows it is adding.
    """
    if not held and not pending:
        return c
    if not settings.ALLOW_STACKING:
        return replace(c, gate_block="already_held")
    if pending:
        return replace(c, gate_block="pending_order")
    if exiting:
        return replace(c, gate_block="exiting")
    direction = pos_and_risk.held_direction(held)  # None = mixed book, no add
    aligned = tuple(e for e in c.events if e.direction == direction)
    if not aligned:
        return replace(c, gate_block="opposing_held", held=direction)
    return replace(c, events=aligned, held=direction)


def _build_trading_signals(
    symbols: tuple[str, ...],
    config: Config,
    stock_data: object,
    mids: dict,
    now: datetime,
    trades: dict | None = None,
    quotes: dict | None = None,
) -> dict[str, SymbolFeatures]:
    """Create the trading signals: OHLCV -> RSI/ATR/MACD -> events, per symbol,
    plus the tape reading (tick-rule imbalance over `trades`, L1 skew from `quotes`).

    A failed symbol is marked data_error and skipped, never invented — one bad
    symbol must not kill the cycle.
    """
    features: dict[str, SymbolFeatures] = {}
    for symbol in symbols:
        flow = (
            tape.tick_rule(trades.get(symbol, []), settings.FLOW_MIN_TRADES)
            if trades is not None
            else None
        )
        quote = (quotes or {}).get(symbol)
        l1 = tape.l1_imbalance(quote.bid_size, quote.ask_size) if quote is not None else None
        try:
            frame = market_data.fetch_ohlcv(
                stock_data, symbol, config.bar_timeframe, now
            )
            frame = signals.add_indicators(frame)
            features[symbol] = signals.build_signal(
                symbol, frame, mids.get(symbol), now, config.bar_seconds, flow=flow, l1_imbalance=l1
            )
        except market_data.MarketDataError as error:
            logger.warning("{}", error)
            features[symbol] = SymbolFeatures(
                symbol=symbol,
                mid=mids.get(symbol),
                rsi=None,
                atr=None,
                macd_hist=None,
                events=(),
                bar_age_seconds=None,
                gate_block="data_error",
            )
    return features


def _decide(
    tradeable, config: Config, manual_mode: bool, llm_transport
) -> EntryChoice | None:
    if manual_mode:
        choice = decision_layer.manual_decide(tradeable)
    else:
        if not config.llm_api_key:
            logger.error("FEATHERLESS_API_KEY missing; use --manual-mode or set the key")
            return None
        try:
            choice = decision_layer.decide_entry(
                tradeable, config.llm_api_key, transport=llm_transport
            )
        except decision_layer.LlmError as error:
            logger.error("LLM decision failed, holding: {}", error)
            return None
    if choice:
        logger.info(
            "entry choice: {} {} ({})", choice.symbol, choice.direction, choice.model
        )
    else:
        logger.info("no entry this cycle")
    return choice


def _attempt_entry(
    choice: EntryChoice,
    spot: float | None,
    config: Config,
    trading: object,
    option_data: object,
    equity: float | None,
    open_risk: float | None,
    underlying_risk: float | None,
    pending_symbols: frozenset[str],
    cycle_id: str,
    execute: bool,
    *,
    cycle_spent: float,
    exclude_symbols: frozenset[str] = frozenset(),
) -> dict:
    entry: dict = {
        "symbol": choice.symbol,
        "direction": choice.direction,
        "thesis": choice.thesis,
        "model": choice.model,
    }
    if spot is None:
        entry["rejected"] = "missing_quote"
        return entry
    try:
        # Fresh clock: the cycle-start clock is stale by now (manual mode can sit at
        # the prompt for minutes), and quotes newer than it fail check_leg's
        # future_quote sanity check.
        screen_clock = broker.fetch_clock(trading)
        spread, rejections = _screen_spread(
            trading,
            option_data,
            choice.symbol,
            choice.direction,
            spot,
            screen_clock.server_time,
            screen_clock.server_time.date(),
            exclude_symbols,
        )
    except broker.BrokerError as error:
        entry["rejected"] = str(error)
        return entry
    entry["screen_rejections"] = rejections
    if spread is None:
        entry["rejected"] = "no_spread"
        logger.info(
            "no acceptable spread for {} {}: {}",
            choice.symbol,
            choice.direction,
            rejections,
        )
        return entry
    entry["spread"] = {
        "long": spread.long.symbol,
        "short": spread.short.symbol,
        "expiration": spread.expiration,
        "width": spread.width,
        "net_debit": spread.net_debit,
        "skew": round(spread.skew, 4),
    }
    qty, reason = pos_and_risk.size_entry(
        spread.net_debit, equity, open_risk, underlying_risk, cycle_spent=cycle_spent
    )
    if reason is not None:
        entry["rejected"] = reason
        logger.info("entry refused by risk manager: {}", reason)
        return entry
    entry["qty"] = qty

    # Fresh pre-submit re-check: account conflicts + re-quoted legs against a fresh clock.
    try:
        fresh_clock = broker.fetch_clock(trading)
        fresh_account = broker.fetch_account_state(trading, config.symbols)
        fresh_snaps = broker.fetch_option_snapshots(
            option_data, [spread.long.symbol, spread.short.symbol]
        )
    except broker.BrokerError as error:
        entry["rejected"] = f"recheck: {error}"
        return entry
    if {spread.long.symbol, spread.short.symbol} & fresh_account.open_order_symbols:
        entry["rejected"] = "pending_order_conflict"
        return entry
    long_q = broker.leg_quote_from_snapshot(
        spread.long.symbol,
        spread.long.strike,
        fresh_snaps.get(spread.long.symbol),
        spread.long.open_interest,
    )
    short_q = broker.leg_quote_from_snapshot(
        spread.short.symbol,
        spread.short.strike,
        fresh_snaps.get(spread.short.symbol),
        spread.short.open_interest,
    )
    for leg in (long_q, short_q):
        failure = options_screener.check_leg(leg, fresh_clock.server_time)
        if failure is not None:
            entry["rejected"] = f"recheck: {failure}"
            return entry
    fresh_debit = round(long_q.ask - short_q.bid, 2)  # type: ignore[operator]
    if not (settings.MIN_NET_DEBIT <= fresh_debit < spread.width):
        entry["rejected"] = "recheck: bad_debit"
        return entry
    if not (settings.MIN_DEBIT_FRAC * spread.width <= fresh_debit <= settings.MAX_DEBIT_FRAC * spread.width):
        entry["rejected"] = "recheck: debit_out_of_band"
        return entry
    qty, reason = pos_and_risk.size_entry(
        fresh_debit, fresh_account.equity, open_risk, underlying_risk, cycle_spent=cycle_spent
    )
    if reason is not None:
        entry["rejected"] = f"recheck: {reason}"
        return entry
    if execute and (fresh_account.options_level or 0) < MIN_OPTIONS_LEVEL:
        entry["rejected"] = "options_level_too_low"
        return entry

    fresh_spread = SpreadQuote(
        underlying=spread.underlying,
        direction=spread.direction,
        expiration=spread.expiration,
        long=long_q,
        short=short_q,
        width=spread.width,
        net_debit=fresh_debit,
        skew=spread.skew,
    )
    entry["premium"] = round(fresh_debit * qty * 100.0, 2)  # dollars this entry commits
    plan = options_screener.build_entry_plan(fresh_spread, qty, cycle_id)
    entry["receipt"] = _settle(trading, plan, execute)
    return entry


def _settle(trading: object, plan: OrderPlan, execute: bool) -> dict:
    if not execute:
        return {
            "submitted": False,
            "dry_run": True,
            "plan": {
                "kind": plan.kind,
                "qty": plan.qty,
                "limit_price": plan.limit_price,
                "legs": [f"{l.side} {l.symbol}" for l in plan.legs],
                "client_order_id": plan.client_order_id,
            },
        }
    receipt = broker.submit_paper_order(trading, plan)
    logger.info(
        "order {}: submitted={} status={} error={}",
        plan.client_order_id,
        receipt.submitted,
        receipt.status,
        receipt.error,
    )
    if receipt.submitted:
        sounds.play_order_sound()
    return {
        "submitted": receipt.submitted,
        "order_id": receipt.order_id,
        "status": receipt.status,
        "error": receipt.error,
        "client_order_id": receipt.client_order_id,
    }


# --- Fill tracking: sound + log when a submitted order actually fills ---
FILL_POLL_TIMEOUT_SECONDS = 30  # short poll right after a cycle submits an order
FILL_POLL_INTERVAL_SECONDS = 2
_FILLED = {"filled"}
_DEAD = {"canceled", "cancelled", "expired", "rejected", "done_for_day"}
# anything else (new, accepted, partially_filled, ...) stays pending


def _new_orders(record: dict) -> dict[str, str]:
    """order_id -> readable label for every order the cycle actually submitted."""
    orders: dict[str, str] = {}
    for exit_entry in record.get("exits") or []:
        receipt = exit_entry.get("receipt") or {}
        if receipt.get("submitted") and receipt.get("order_id"):
            orders[receipt["order_id"]] = f"exit {exit_entry.get('spread')}"
    for entry in journal_entries(record):
        receipt = entry.get("receipt") or {}
        if receipt.get("submitted") and receipt.get("order_id"):
            orders[receipt["order_id"]] = f"entry {entry.get('symbol')}"
    return orders


def _check_fills(trading: object, pending: dict[str, str]) -> None:
    """Resolve pending orders in place; sound + log on a fill. Notification only, never raises."""
    for order_id, label in list(pending.items()):
        status = broker.fetch_order_status(trading, order_id)
        if status in _FILLED:
            logger.info("FILLED: {} (order {})", label, order_id)
            sounds.play_fill_sound()
            del pending[order_id]
        elif status in _DEAD:
            logger.info("order not filled ({}): {} (order {})", status, label, order_id)
            del pending[order_id]
        # None (lookup failed) or still open: keep waiting
    if pending:
        logger.info("awaiting fill: {}", ", ".join(pending.values()))


def _safe_cycle(
    config: Config,
    trading: object,
    stock_data: object,
    option_data: object,
    *,
    execute: bool,
    manual_mode: bool,
    **kwargs,
) -> dict:
    """run_cycle that never raises: a crash is journaled as outcome=error so
    --loop keeps going. The traceback is reduced to type + location (loguru's
    diagnose output would print frame variables, credentials included)."""
    try:
        return run_cycle(
            config, trading, stock_data, option_data,
            execute=execute, manual_mode=manual_mode, **kwargs,
        )
    except KeyboardInterrupt:
        raise
    except Exception as error:  # noqa: BLE001 - the loop must survive anything
        tb = error.__traceback__
        while tb is not None and tb.tb_next is not None:
            tb = tb.tb_next
        where = f"{Path(tb.tb_frame.f_code.co_filename).name}:{tb.tb_lineno}" if tb else "?"
        logger.error("cycle crashed: {} at {}", type(error).__name__, where)
        started = datetime.now(timezone.utc)
        record = {
            "cycle_id": started.strftime("%Y%m%d-%H%M%S"),
            "started_at": started,
            "dry_run": not execute,
            "outcome": "error",
            "error": type(error).__name__,
        }
        append_journal(record)
        return record


@app.command()
def run(
    execute: bool = typer.Option(
        False, help="Actually submit paper orders (dry run otherwise)."
    ),
    manual_mode: bool = typer.Option(
        False, help="Pick the entry candidate yourself instead of asking the LLM."
    ),
    loop: bool = typer.Option(False, help="Run forever on an interval."),
    interval: int = typer.Option(
        settings.LOOP_INTERVAL_SECONDS, help="Seconds between cycles with --loop."
    ),
    with_dashboard: bool = typer.Option(
        False, "--dashboard",
        help="Refresh the dashboard data after every cycle; with --loop also serve it locally.",
    ),
    serve_port: int = typer.Option(8080, help="Local port for the dashboard pages (--loop --dashboard)."),
) -> None:
    """Run one trading cycle (or loop). Paper only; dry run unless --execute."""
    setup_logging(file_sink=True)
    config, trading, stock_data, option_data = _bootstrap()
    if not manual_mode and not config.llm_api_key:
        # Without a key every cycle would silently hold: refuse to start instead.
        typer.echo("FEATHERLESS_API_KEY missing: set it in .env or run with --manual-mode")
        raise typer.Exit(1)
    if not manual_mode:
        # A present-but-rejected key or a wrong model id would also mean a silent
        # all-hold day: prove the LLM answers before the first cycle.
        try:
            model, seconds = decision_layer.ping(config.llm_api_key)
        except decision_layer.LlmError as error:
            typer.echo(f"LLM check failed ({settings.LLM_PROVIDER}): {error}")
            raise typer.Exit(1)
        logger.info("LLM ready: {} answered in {:.1f}s", model, seconds)
    if execute:
        logger.warning("ARMED: paper order submission is enabled")
    # Seed fill tracking from orders already open at the broker, so a restart
    # resumes watching what a previous run submitted.
    pending: dict[str, str] = {}  # order_id -> label
    try:
        pending = broker.fetch_open_orders(trading)
    except broker.BrokerError as error:
        logger.warning("could not list open orders at startup: {}", error)
    if pending:
        logger.info("watching open orders for fills: {}", ", ".join(pending.values()))
    if with_dashboard and loop:
        dashboard.serve(serve_port)
        logger.info(
            "dashboard: http://localhost:{}/paca-cycles/  http://localhost:{}/paca-candles/",
            serve_port, serve_port,
        )
    # Tape memory lives in the loop process; a single-shot run has no previous
    # cycle to confirm against, so it keeps the event-only reversal rule.
    flow_state: dict[str, tape.TapeState] | None = {} if loop else None
    try:
        while True:
            cycle_started = time.monotonic()
            record = _safe_cycle(
                config,
                trading,
                stock_data,
                option_data,
                execute=execute,
                manual_mode=manual_mode,
                flow_state=flow_state,
            )
            logger.info("cycle {} outcome: {}", record["cycle_id"], record.get("outcome"))
            new = _new_orders(record)
            pending.update(new)
            if new:
                # Short poll for instant fill feedback, timeboxed so an old
                # straggler can never stall the loop.
                deadline = time.monotonic() + FILL_POLL_TIMEOUT_SECONDS
                while True:
                    _check_fills(trading, pending)
                    if not (new.keys() & pending.keys()) or time.monotonic() >= deadline:
                        break
                    time.sleep(FILL_POLL_INTERVAL_SECONDS)
            if with_dashboard:
                try:
                    logger.info("dashboard export: {}", dashboard.export_all())
                except Exception as error:  # noqa: BLE001 - never let the dashboard stop trading
                    logger.warning("dashboard export skipped: {}", type(error).__name__)
            if not loop:
                _check_fills(trading, pending)  # final status check before exit
                if pending:
                    logger.info(
                        "exiting with open orders; a later `run` resumes watching them"
                    )
                break
            # Keep the cadence: sleep for what is left of the interval after the
            # cycle + exports, so 5-minute bars are read once each.
            time.sleep(max(0.0, interval - (time.monotonic() - cycle_started)))
            _check_fills(trading, pending)  # catch slow fills from earlier cycles
    except KeyboardInterrupt:
        logger.warning("stopped by user; open orders (if any) keep working at the broker")


@app.command()
def preflight() -> None:
    """Pre-flight smoke test: settings.yaml, credentials + paper guards, connectivity."""
    setup_logging()
    try:
        values = settings.load_settings()
    except settings.SettingsError as error:
        typer.echo(f"FAIL settings.yaml: {error}")
        raise typer.Exit(1)
    typer.echo(
        f"OK   settings.yaml — all {len(values)} required values present and sane:"
    )
    for name in sorted(values):
        typer.echo(f"       {name} = {values[name]}")

    try:
        config = broker.load_config()
    except broker.ConfigError as error:
        typer.echo(f"FAIL credentials: {error}")
        raise typer.Exit(1)
    typer.echo("OK   credentials + paper-only guards (.env)")

    try:
        trading, _, _ = broker.build_clients(config)
        clock = broker.fetch_clock(trading)
    except broker.BrokerError as error:
        typer.echo(f"FAIL Alpaca connectivity: {error}")
        raise typer.Exit(1)
    state = "open" if clock.is_open else "closed"
    typer.echo(
        f"OK   Alpaca connectivity — market {state}, server time {clock.server_time}"
    )

    try:
        sdk_number = str(getattr(broker.guarded("account read", trading.get_account), "account_number", "") or "")
    except broker.BrokerError:
        sdk_number = ""
    profile = os.environ.get("ALPACA_CLI_PROFILE", "").strip() or None
    snapshot = dashboard.cli_snapshot(profile, expected_account_number=sdk_number or None)
    if snapshot["source"] == "alpaca-cli":
        typer.echo(
            f"OK   Alpaca CLI — profile {profile or 'default'} reads account "
            f"{snapshot['account'].get('account_number')} (same as the .env keys)"
        )
    else:
        typer.echo(
            f"WARN Alpaca CLI snapshot unavailable ({snapshot.get('cli_error')}); "
            "the dashboard shows SDK data only"
        )

    if not config.llm_api_key:
        typer.echo("WARN LLM key missing (FEATHERLESS_API_KEY): only --manual-mode can decide entries")
    else:
        try:
            model, seconds = decision_layer.ping(config.llm_api_key)
        except decision_layer.LlmError as error:
            typer.echo(f"FAIL LLM ({settings.LLM_PROVIDER}): {error}")
            raise typer.Exit(1)
        typer.echo(f"OK   LLM {settings.LLM_PROVIDER} — {model} answered in {seconds:.1f}s")
    typer.echo("preflight passed")


@app.command()
def account(
    export: bool = typer.Option(
        False, "--export", help="Also write the snapshot to logs/account.json (dashboard data)."
    ),
) -> None:
    """Show equity, options level, paired spreads and warnings (read-only)."""
    setup_logging()
    config, trading, _, _ = _bootstrap()
    state = broker.fetch_account_state(trading, config.symbols)
    for leg in state.legs:
        logger.info("position: {} qty={} avg_entry={}", leg.symbol, leg.qty, leg.avg_entry_price)
    spreads, warnings = pos_and_risk.pair_spreads(state.legs)
    open_risk = pos_and_risk.open_premium_at_risk(spreads)
    typer.echo(f"equity: {state.equity}  options_level: {state.options_level}")
    typer.echo(f"open premium at risk: {open_risk}")
    for spread in spreads:
        typer.echo(
            f"  {spread.underlying} {spread.expiration} {spread.option_type} x{spread.qty} "
            f"long={spread.long_symbol} short={spread.short_symbol} entry_debit={spread.net_entry_debit}"
        )
    for warning in warnings:
        typer.echo(f"  WARNING {warning}")
    for symbol in state.unparsed_positions:
        typer.echo(f"  WARNING unparsed position: {symbol}")
    if export:
        snapshot = {
            "generated_at": datetime.now(timezone.utc),
            "account_number": state.account_number,
            "equity": state.equity,
            "options_level": state.options_level,
            "open_risk": open_risk,
            "spreads": [asdict(s) for s in spreads],
            "warnings": warnings,
            "unparsed_positions": list(state.unparsed_positions),
        }
        path = Path("logs") / "account.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(to_json_line(snapshot) + "\n", encoding="utf-8")
        typer.echo(f"exported {path}")


@app.command()
def cancel(
    order_id: str = typer.Argument(
        None, help="Order id to cancel; omit to cancel ALL open orders."
    ),
    yes: bool = typer.Option(False, "--yes", help="Skip the confirmation prompt."),
) -> None:
    """Cancel open orders on the paper account (all of them, or one by id)."""
    setup_logging()
    _, trading, _, _ = _bootstrap()
    open_orders = broker.fetch_open_orders(trading)
    if not open_orders:
        typer.echo("no open orders")
        return
    if order_id is not None:
        if order_id not in open_orders:
            typer.echo(f"order {order_id} is not open (open: {list(open_orders)})")
            raise typer.Exit(1)
        targets = {order_id: open_orders[order_id]}
    else:
        targets = open_orders
    for oid, label in targets.items():
        typer.echo(f"  {oid}  {label}")
    if not yes:
        typer.confirm(f"cancel {len(targets)} open order(s)?", abort=True)
    failed = False
    for oid, label in targets.items():
        try:
            broker.cancel_order(trading, oid)
            typer.echo(f"cancel requested: {oid}  {label}")
        except broker.BrokerError as error:
            failed = True
            typer.echo(f"FAIL {oid}  {label}: {error}")
    if failed:
        raise typer.Exit(1)


@app.command()
def flow(
    minutes: int = typer.Option(
        settings.FLOW_LOOKBACK_MINUTES, help="Minutes of IEX prints behind the reading."
    ),
    at: str = typer.Option(
        None, help="Read the window ending at this UTC time 'YYYY-MM-DD HH:MM' instead of now (after-hours checks)."
    ),
) -> None:
    """Tape sensor read-out per whitelisted symbol (read-only): prints, buy/sell volume, flow, L1 skew."""
    setup_logging()
    config, trading, stock_data, _ = _bootstrap()
    if at:
        end = datetime.strptime(at, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
    else:
        end = broker.fetch_clock(trading).server_time
    trades = broker.fetch_recent_trades(stock_data, config.symbols, minutes, end)
    quotes = broker.fetch_spot_quotes(stock_data, config.symbols)
    typer.echo(
        f"window: {minutes} min ending {end.isoformat(timespec='seconds')}  "
        f"(min prints {settings.FLOW_MIN_TRADES}, entry gate |flow| >= {settings.FLOW_MIN_IMBALANCE})"
    )
    for symbol in config.symbols:
        stats = tape.tick_rule(trades.get(symbol, []), settings.FLOW_MIN_TRADES)
        quote = quotes.get(symbol)
        l1 = tape.l1_imbalance(quote.bid_size, quote.ask_size) if quote is not None else None
        flow_text = "unknown" if stats.imbalance is None else f"{stats.imbalance:+.2f}"
        l1_text = "n/a" if l1 is None else f"{l1:+.2f}"
        typer.echo(
            f"  {symbol:<5} prints={stats.trades:<5} buy={stats.buy_volume:>9.0f} "
            f"sell={stats.sell_volume:>9.0f} flow={flow_text:>8} l1={l1_text}"
        )


@app.command()
def candidates() -> None:
    """Show indicators, events and gate results for every whitelisted symbol (read-only)."""
    setup_logging()
    config, trading, stock_data, _ = _bootstrap()
    clock = broker.fetch_clock(trading)
    quotes = broker.fetch_spot_quotes(stock_data, config.symbols)
    mids = {symbol: quote.mid for symbol, quote in quotes.items()}
    try:
        trades = broker.fetch_recent_trades(
            stock_data, config.symbols, settings.FLOW_LOOKBACK_MINUTES, clock.server_time
        )
    except broker.BrokerError as error:
        logger.warning("trades read failed, tape unknown: {}", error)
        trades = {}
    features = _build_trading_signals(
        config.symbols, config, stock_data, mids, clock.server_time, trades=trades, quotes=quotes
    )
    for c in signals.build_candidates(features, clock.is_open, config.bar_seconds):
        events = ",".join(e.kind for e in c.events) or "-"
        flow = f"{c.flow_imbalance:+.2f}/{c.flow_trades}" if c.flow_imbalance is not None else f"?/{c.flow_trades}"
        rsi = f"{c.rsi:.1f}" if c.rsi is not None else "-"
        atr = f"{c.atr:.3f}" if c.atr is not None else "-"
        hist = f"{c.macd_hist:+.4f}" if c.macd_hist is not None else "-"
        ema_fast = f"{c.ema_fast_dist:+.2f}" if c.ema_fast_dist is not None else "-"
        ema_slow = f"{c.ema_slow_dist:+.2f}" if c.ema_slow_dist is not None else "-"
        typer.echo(
            f"{c.symbol:<6} mid={c.mid} rsi={rsi} atr={atr} macd_hist={hist} "
            f"ema{settings.TREND_EMA_FAST}={ema_fast} ema{settings.TREND_EMA_SLOW}={ema_slow} "
            f"flow={flow} events={events} gate={c.gate_block or 'PASS'}"
        )


@app.command()
def screen(
    symbol: str = typer.Argument(..., help="Underlying symbol, e.g. SPY"),
    direction: str = typer.Option(..., "--direction", help="CALL or PUT"),
) -> None:
    """Show the exact spread the screener would pick (read-only, no LLM, no order)."""
    setup_logging()
    direction = direction.upper()
    if direction not in ("CALL", "PUT"):
        raise typer.BadParameter("--direction must be CALL or PUT")
    config, trading, stock_data, option_data = _bootstrap()
    clock = broker.fetch_clock(trading)
    symbol = symbol.upper()
    spot = broker.fetch_spot_mids(stock_data, (symbol,))[symbol]
    if spot is None:
        typer.echo("no usable underlying quote")
        raise typer.Exit(1)
    spread, rejections = _screen_spread(
        trading,
        option_data,
        symbol,
        direction,
        spot,
        clock.server_time,
        clock.server_time.date(),
    )
    typer.echo(f"spot: {spot}  rejections: {rejections}")
    if spread is None:
        typer.echo("no acceptable spread")
        raise typer.Exit(1)
    typer.echo(
        f"{spread.direction} {spread.underlying} {spread.expiration}: "
        f"long {spread.long.symbol} @ {spread.long.strike} / short {spread.short.symbol} @ {spread.short.strike}\n"
        f"width={spread.width} net_debit={spread.net_debit} skew={spread.skew:.4f} "
        f"OI long/short={spread.long.open_interest}/{spread.short.open_interest}"
    )


if __name__ == "__main__":
    app()
