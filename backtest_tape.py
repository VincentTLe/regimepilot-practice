"""Signal-quality backtest for the tape sensor. Read-only research: no orders.

    uv run --env-file .env backtest_tape.py [--days 5] [--symbols SPY,NVDA] [--out logs/backtest_tape.csv]

For each completed session and whitelisted symbol it pulls the same data the
live engine reads (5m IEX bars, IEX trade prints), replays the live entry gates
on every regular-hours bar — events on the completed bar, the RSI exhaustion
filter, the tape agreement over the trailing FLOW_LOOKBACK_MINUTES — and then
measures what the underlying did next: the move in ATR units and percent 6 and
12 bars later, and a reversal-exit simulation with and without the tape
confirmation. The LLM is not replayed: every gate-passing event is taken.

Options P&L is NOT modelled (no historical option data on this account). The
`$ proxy` is a coarse translation: a near-ATM debit vertical carries delta
~0.4, so P&L per contract ~ 0.4 x underlying move x 100 minus ~$12 friction.
"""

from __future__ import annotations

import bisect
import math
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import typer
from alpaca.data.requests import StockBarsRequest, StockTradesRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.trading.requests import GetCalendarRequest
from loguru import logger

import broker
import market_data
import settings
import signals
import tape
from cli import setup_logging
from export_candles import bar_events

HORIZONS = (6, 12)  # bars ahead: 30 and 60 minutes on 5m bars
REVERSAL_HORIZON = 12
DELTA_PROXY = 0.4
FRICTION_PER_CONTRACT = 12.0
REQUEST_PAUSE = 0.25  # seconds between data requests (rate-limit courtesy)
DIRECTION_OF = {"gap_up": "CALL", "breakout_up": "CALL", "macd_cross_up": "CALL",
                "gap_down": "PUT", "breakout_down": "PUT", "macd_cross_down": "PUT"}

app = typer.Typer(add_completion=False)


# --- pure helpers -----------------------------------------------------------

def flow_at(prints: list[tuple[float, float, float]], end_ts: float, minutes: int,
            min_trades: int) -> tape.FlowStats:
    """Tick-rule flow over the prints in (end_ts - minutes, end_ts]; prints are (ts, price, size) sorted."""
    stamps = [p[0] for p in prints]
    lo = bisect.bisect_right(stamps, end_ts - minutes * 60)
    hi = bisect.bisect_right(stamps, end_ts)
    return tape.tick_rule([(p[1], p[2]) for p in prints[lo:hi]], min_trades)


def rsi_filtered(events: list[str], rsi: float | None) -> list[str]:
    """signals.entry_events rule on event kinds: CALL dropped at RSI >= overbought, PUT at <= oversold."""
    if rsi is None or math.isnan(rsi):
        return list(events)
    out = []
    for kind in events:
        direction = DIRECTION_OF[kind]
        if direction == "CALL" and rsi >= settings.RSI_OVERBOUGHT:
            continue
        if direction == "PUT" and rsi <= settings.RSI_OVERSOLD:
            continue
        out.append(kind)
    return out


def evaluate_day(
    df: pd.DataFrame,
    prints: list[tuple[float, float, float]],
    session_open: datetime,
    session_close: datetime,
    *,
    bar_seconds: int = settings.BAR_SECONDS,
) -> list[dict]:
    """Replay the live entry gates on every regular-hours bar of one session.

    `df` is an add_indicators() frame (UTC bar-start index) that may include
    warm-up bars from earlier sessions. One row per (bar, direction) whose
    events survive the RSI filter, with the tape status and forward outcomes.
    """
    if df.empty:
        return []
    events_by_bar = bar_events(df)
    stamps = [ts.timestamp() for ts in df.index]
    open_ts, close_ts = session_open.timestamp(), session_close.timestamp()
    session_last = max((i for i, ts in enumerate(stamps) if ts + bar_seconds <= close_ts + 1), default=-1)
    closes = df["close"].to_list()
    atrs = df["atr"].to_list()
    rsis = df["rsi"].to_list()
    flow_cache: dict[int, tape.FlowStats] = {}

    def flow(i: int) -> tape.FlowStats:
        if i not in flow_cache:
            flow_cache[i] = flow_at(prints, stamps[i] + bar_seconds, settings.FLOW_LOOKBACK_MINUTES,
                                    settings.FLOW_MIN_TRADES)
        return flow_cache[i]

    rows: list[dict] = []
    for i in range(len(df)):
        if stamps[i] < open_ts or stamps[i] + bar_seconds > close_ts + 1:
            continue  # only bars completed inside the session
        if i < settings.MIN_BARS or any(pd.isna(v) for v in (atrs[i], rsis[i])) or atrs[i] <= 0:
            continue
        kinds = rsi_filtered(events_by_bar[i], rsis[i])
        if not kinds:
            continue
        stats = flow(i)
        for direction in sorted({DIRECTION_OF[k] for k in kinds}):
            if stats.imbalance is None:
                status = "unknown"
            elif tape.flow_agrees(direction, stats.imbalance, settings.FLOW_MIN_IMBALANCE):
                status = "agree"
            else:
                status = "disagree"
            sign = 1.0 if direction == "CALL" else -1.0
            row = {
                "date": session_open.date().isoformat(),
                "bar_time": df.index[i].isoformat(),
                "symbol": None,  # filled by the caller
                "direction": direction,
                "events": "+".join(k for k in kinds if DIRECTION_OF[k] == direction),
                "close": closes[i],
                "rsi": round(rsis[i], 1),
                "atr": round(atrs[i], 4),
                "flow": None if stats.imbalance is None else round(stats.imbalance, 3),
                "flow_trades": stats.trades,
                "tape": status,
            }
            for horizon in HORIZONS:
                j = min(i + horizon, session_last)
                move = (closes[j] - closes[i]) * sign if j > i else 0.0
                row[f"atr_move_{horizon}"] = round(move / atrs[i], 3)
                row[f"pct_move_{horizon}"] = round(100 * move / closes[i], 3)
            # reversal-exit simulation: event-only rule (dev/paca) vs tape-confirmed rule
            opposite = "PUT" if direction == "CALL" else "CALL"
            option_type = "C" if direction == "CALL" else "P"
            end = min(i + REVERSAL_HORIZON, session_last)
            exit_a = exit_b = end
            pending = 0
            for j in range(i + 1, end + 1):
                opposing_event = any(DIRECTION_OF[k] == opposite for k in events_by_bar[j])
                if opposing_event and exit_a == end:
                    exit_a = j
                if opposing_event:
                    pending = settings.FLOW_EXIT_BARS
                elif pending > 0:
                    pending -= 1
                if exit_b == end and (opposing_event or pending > 0):
                    readings = [flow(k).imbalance for k in range(max(i, j - settings.FLOW_EXIT_BARS + 1), j + 1)]
                    against = tape.flow_against(option_type, readings, settings.FLOW_EXIT_BARS,
                                                settings.FLOW_MIN_IMBALANCE)
                    if against is None or against:
                        exit_b = j
                        pending = 0
                if exit_a != end and exit_b != end:
                    break
            for label, j in (("event_only", exit_a), ("tape_rule", exit_b)):
                move = (closes[j] - closes[i]) * sign if j > i else 0.0
                row[f"exit_{label}_bars"] = j - i
                row[f"exit_{label}_atr"] = round(move / atrs[i], 3)
            rows.append(row)
    return rows


def dollar_proxy(pct_move: float, close: float) -> float:
    return DELTA_PROXY * (pct_move / 100.0) * close * 100 - FRICTION_PER_CONTRACT


# --- data ---------------------------------------------------------------------

def completed_sessions(trading, days: int, now: datetime) -> list:
    request = GetCalendarRequest(start=(now - timedelta(days=days * 3 + 7)).date(), end=now.date())
    calendar = broker.guarded("calendar read", lambda: trading.get_calendar(request))
    done = [c for c in calendar if c.close.astimezone(timezone.utc) <= now]
    return done[-days:]


def fetch_bars(stock, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
    request = StockBarsRequest(symbol_or_symbols=symbol, timeframe=TimeFrame(5, TimeFrameUnit.Minute),
                               start=start, end=end, feed=market_data.STOCK_FEED)
    raw = broker.guarded(f"bars read {symbol}", lambda: stock.get_stock_bars(request))
    rows = []
    for bar in raw.data.get(symbol, []):
        rows.append({"timestamp": bar.timestamp, "open": float(bar.open), "high": float(bar.high),
                     "low": float(bar.low), "close": float(bar.close), "volume": float(bar.volume)})
    frame = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    frame = frame.drop_duplicates(subset="timestamp", keep="last").set_index("timestamp").sort_index()
    return signals.add_indicators(frame) if len(frame) else frame


def fetch_prints(stock, symbol: str, start: datetime, end: datetime) -> list[tuple[float, float, float]]:
    request = StockTradesRequest(symbol_or_symbols=symbol, start=start, end=end,
                                 feed=market_data.STOCK_FEED, limit=400_000)
    raw = broker.guarded(f"trades read {symbol}", lambda: stock.get_stock_trades(request))
    data = getattr(raw, "data", raw)
    prints = []
    for row in data.get(symbol, []) or []:
        price, size = broker.as_float(getattr(row, "price", None)), broker.as_float(getattr(row, "size", None))
        if price and size and size > 0:
            prints.append((row.timestamp.timestamp(), price, size))
    prints.sort()
    return prints


# --- report ---------------------------------------------------------------------

def summarize(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "no signals"
    frame = frame.copy()
    frame["usd_6"] = [dollar_proxy(p, c) for p, c in zip(frame["pct_move_6"], frame["close"])]
    frame["usd_12"] = [dollar_proxy(p, c) for p, c in zip(frame["pct_move_12"], frame["close"])]
    frame["hit_6"] = frame["atr_move_6"] > 0
    frame["hit_12"] = frame["atr_move_12"] > 0
    parts = []
    order = ["agree", "disagree", "unknown"]
    groups = frame.groupby("tape")
    table = pd.DataFrame({
        "signals": groups.size(),
        "hit%_30m": (100 * groups["hit_6"].mean()).round(0),
        "hit%_60m": (100 * groups["hit_12"].mean()).round(0),
        "avg_ATR_30m": groups["atr_move_6"].mean().round(2),
        "avg_ATR_60m": groups["atr_move_12"].mean().round(2),
        "median_ATR_60m": groups["atr_move_12"].median().round(2),
        "sum_$proxy_60m": groups["usd_12"].sum().round(0),
    }).reindex([g for g in order if g in groups.groups])
    parts.append("Forward move by tape status (every RSI-passing event, 1 contract each):\n" + table.to_string())
    parts.append(
        "\nJohn's gates alone (all rows): n={n}, hit%_60m={hit:.0f}, avg ATR 60m={avg:.2f}, $proxy 60m={usd:.0f}".format(
            n=len(frame), hit=100 * frame["hit_12"].mean(), avg=frame["atr_move_12"].mean(), usd=frame["usd_12"].sum())
    )
    agree = frame[frame["tape"] == "agree"]
    if len(agree):
        parts.append(
            "Tape-agree only: n={n}, hit%_60m={hit:.0f}, avg ATR 60m={avg:.2f}, $proxy 60m={usd:.0f}".format(
                n=len(agree), hit=100 * agree["hit_12"].mean(), avg=agree["atr_move_12"].mean(), usd=agree["usd_12"].sum())
        )
    exits = pd.DataFrame({
        "rule": ["event_only", "tape_rule"],
        "avg_exit_ATR": [frame["exit_event_only_atr"].mean().round(2), frame["exit_tape_rule_atr"].mean().round(2)],
        "exited_early%": [100 * (frame["exit_event_only_bars"] < REVERSAL_HORIZON).mean(),
                          100 * (frame["exit_tape_rule_bars"] < REVERSAL_HORIZON).mean()],
        "avg_hold_bars": [frame["exit_event_only_bars"].mean(), frame["exit_tape_rule_bars"].mean()],
    }).round(1)
    parts.append("\nReversal-exit simulation over 12 bars (all rows):\n" + exits.to_string(index=False))
    if len(agree):
        exits_a = pd.DataFrame({
            "rule": ["event_only", "tape_rule"],
            "avg_exit_ATR": [agree["exit_event_only_atr"].mean().round(2), agree["exit_tape_rule_atr"].mean().round(2)],
            "exited_early%": [100 * (agree["exit_event_only_bars"] < REVERSAL_HORIZON).mean(),
                              100 * (agree["exit_tape_rule_bars"] < REVERSAL_HORIZON).mean()],
        }).round(1)
        parts.append("Reversal-exit simulation, tape-agree entries only:\n" + exits_a.to_string(index=False))
    by_symbol = frame[frame["tape"] == "agree"].groupby("symbol")["atr_move_12"].agg(["size", "mean"]).round(2)
    if len(by_symbol):
        parts.append("\nTape-agree entries by symbol (n, avg ATR 60m):\n" + by_symbol.to_string())
    return "\n".join(parts)


@app.command()
def run(
    days: int = typer.Option(5, help="Completed sessions to replay."),
    symbols: str = typer.Option("", help="Comma-separated override of the whitelist."),
    out: Path = typer.Option(Path("logs") / "backtest_tape.csv", help="Per-signal CSV."),
) -> None:
    """Replay the entry gates on past sessions and report signal quality."""
    setup_logging()
    config = broker.load_config()
    trading, stock, _ = broker.build_clients(config)
    universe = tuple(s.strip().upper() for s in symbols.split(",") if s.strip()) or config.symbols
    now = datetime.now(timezone.utc)
    sessions = completed_sessions(trading, days, now)
    logger.info("sessions: {}", [c.date.isoformat() for c in sessions])
    rows: list[dict] = []
    for session in sessions:
        session_open = session.open.astimezone(timezone.utc)
        session_close = session.close.astimezone(timezone.utc)
        for symbol in universe:
            try:
                df = fetch_bars(stock, symbol, session_open - timedelta(days=4), session_close)
                time.sleep(REQUEST_PAUSE)
                prints = fetch_prints(stock, symbol, session_open - timedelta(minutes=settings.FLOW_LOOKBACK_MINUTES),
                                      session_close)
                time.sleep(REQUEST_PAUSE)
            except broker.BrokerError as error:
                logger.warning("{} {}: {}", session.date, symbol, error)
                continue
            day_rows = evaluate_day(df, prints, session_open, session_close)
            for row in day_rows:
                row["symbol"] = symbol
            rows.extend(day_rows)
            logger.info("{} {}: {} bars, {} prints, {} signals", session.date, symbol, len(df), len(prints), len(day_rows))
    frame = pd.DataFrame(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    typer.echo(f"\n{len(frame)} signals over {len(sessions)} sessions x {len(universe)} symbols -> {out}\n")
    typer.echo(summarize(frame))


if __name__ == "__main__":
    app()
