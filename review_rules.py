"""Rule audit by counterfactual replay: grade every decision of a session against later prices.

    uv run --env-file .env review_rules.py [--journal logs/cycles.jsonl] [--date 2026-09-03]
                                           [--html surge_artifacts/paca-backtest/review_2026-09-03.html] [--deploy]

Every market-open cycle journals every candidate with its gate verdict, its raw
events (kind:direction BEFORE the RSI and tape filters), the tape reading and
the price, plus the entries attempted and the decider's passes. This script
refetches the day's 5-minute bars once per symbol and measures the underlying's
move from the cycle's last completed bar to 60 minutes later and to the session
close, in ATR units signed by the candidate's direction. Groups:

  entered     an order was submitted for that symbol/direction
  no_spread   the decider chose it, the option screener found nothing tradeable
  llm_pass    offered to the decider (gate PASS), it declined
  <gate>      blocked by that rule (flow_disagree, rsi_exhausted, ...), graded on its raw events

Rows without an event direction are graded on the tape sign alone when
|flow| >= 0.10 (the "tape-only" counterfactual). One row per candidate per
cycle: a signal that persists across cycles counts once per cycle, exactly as
the live loop would have to decide on it. Options P&L is not modelled; the
underlying's move is the yardstick. Read-only: nothing here trades.
"""

from __future__ import annotations

import html
import json
import os
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import typer
from loguru import logger

import backtest_tape
import broker

DIRECTION_OF = {"up": "CALL", "down": "PUT", "buy": "CALL", "sell": "PUT"}
TAPE_ONLY_MIN = 0.10
HORIZON_BARS = 12

app = typer.Typer(add_completion=False)


def event_direction(kind: str) -> str | None:
    return DIRECTION_OF.get(kind.rsplit("_", 1)[-1])


def load_cycles(journal: Path, day: str) -> list[dict]:
    rows = []
    for line in journal.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("cycle_id", "").startswith(day.replace("-", "")) and rec.get("market_open") and rec.get("candidates"):
            rows.append(rec)
    return rows


def decision_rows(cycles: list[dict]) -> list[dict]:
    """One row per (cycle, candidate, direction) with a group label; tape-only rows when no direction."""
    out = []
    for rec in cycles:
        started = pd.Timestamp(rec["started_at"])
        if started.tzinfo is None:
            started = started.tz_localize("UTC")
        entered = {}
        for e in rec.get("entries", []):
            receipt = e.get("receipt") or {}
            entered[e["symbol"]] = "entered" if receipt.get("submitted") else ("no_spread" if e.get("rejected") == "no_spread" else f"refused:{e.get('rejected') or 'risk'}")
        entry_dir = {e["symbol"]: e["direction"] for e in rec.get("entries", [])}
        for c in rec["candidates"]:
            base = {"cycle_id": rec["cycle_id"], "started": started, "symbol": c["symbol"], "mid": c.get("mid"),
                    "atr": c.get("atr"), "flow": c.get("flow_imbalance"), "prints": c.get("flow_trades"), "rsi": c.get("rsi"),
                    "ema_fast": c.get("ema_fast_dist"), "ema_slow": c.get("ema_slow_dist")}
            gate = c.get("gate_block")
            raw = c.get("raw_events") or [f"{k}:{event_direction(k)}" for k in c.get("events", [])]
            directions = sorted({r.split(":")[1] for r in raw if ":" in r and r.split(":")[1] in ("CALL", "PUT")})
            if c["symbol"] in entered:
                out.append({**base, "group": entered[c["symbol"]], "direction": entry_dir[c["symbol"]], "events": raw})
            elif gate is None:
                for d in directions or ["?"]:
                    out.append({**base, "group": "llm_pass", "direction": d, "events": raw})
            elif directions:
                for d in directions:
                    out.append({**base, "group": gate, "direction": d, "events": raw})
            flow = c.get("flow_imbalance")
            if flow is not None and abs(flow) >= TAPE_ONLY_MIN and c.get("atr"):
                out.append({**base, "group": "tape_only", "direction": "CALL" if flow > 0 else "PUT", "events": raw,
                            "gate": gate, "aligned": (c.get("ema_fast_dist") or 0) > 0 and (c.get("ema_slow_dist") or 0) > 0
                            if flow > 0 else (c.get("ema_fast_dist") or 0) < 0 and (c.get("ema_slow_dist") or 0) < 0})
    return out


def grade(rows: list[dict], bars_for) -> pd.DataFrame:
    """Add atr60 / atr_close (signed by direction) from the day's 5m bars; drops rows without bars."""
    graded = []
    for r in rows:
        if r["direction"] not in ("CALL", "PUT") or not r.get("atr"):
            continue
        bars = bars_for(r["symbol"])
        if bars is None or bars.empty:
            continue
        completed = bars[bars.index <= r["started"] - timedelta(seconds=300)]
        if completed.empty:
            continue
        i = len(completed) - 1
        et = bars.index.tz_convert("America/New_York")
        session = bars[(et.strftime("%Y-%m-%d") == completed.index[-1].tz_convert("America/New_York").strftime("%Y-%m-%d"))
                       & (et.hour * 60 + et.minute <= 15 * 60 + 55)]
        if session.empty:
            continue
        entry = float(completed["close"].iloc[-1])
        j = min(i + HORIZON_BARS, len(bars) - 1)
        sign = 1 if r["direction"] == "CALL" else -1
        graded.append({**r, "entry": entry, "atr60": sign * (float(bars["close"].iloc[j]) - entry) / r["atr"],
                       "atr_close": sign * (float(session["close"].iloc[-1]) - entry) / r["atr"]})
    return pd.DataFrame(graded)


def stats(g: pd.DataFrame) -> dict:
    return {"n": len(g), "hit60%": round(100 * (g["atr60"] > 0).mean()), "atr60": round(g["atr60"].mean(), 2),
            "hitclose%": round(100 * (g["atr_close"] > 0).mean()), "atr_close": round(g["atr_close"].mean(), 2),
            "symbols": ",".join(sorted(set(g["symbol"]))[:8])}


def report(df: pd.DataFrame, day: str) -> str:
    if df.empty:
        return f"{day}: nothing to grade (no market-open cycles with candidates)"
    pd.set_option("display.width", 200)
    parts = [f"Rule audit {day}: {df['cycle_id'].nunique()} cycles graded, {len(df)} candidate-decisions"]
    decisions = df[df["group"] != "tape_only"]
    order = ["entered", "no_spread", "llm_pass"]
    groups = [g for g in order if g in set(decisions["group"])] + sorted(g for g in set(decisions["group"]) if g not in order)
    table = pd.DataFrame([{"group": g, **stats(decisions[decisions["group"] == g])} for g in groups])
    parts.append("\nBy decision group (underlying move after the cycle, ATR units signed by the trade direction):\n" + table.to_string(index=False))
    tape = df[df["group"] == "tape_only"].copy()
    if len(tape):
        tape["bucket"] = pd.cut(tape["flow"].abs(), [0.10, 0.25, 0.4, 1.01], labels=["0.10-0.25", "0.25-0.40", "0.40+"], right=False)
        t = pd.DataFrame([{"|flow|": str(b), "aligned": a, **stats(g)} for (b, a), g in tape.groupby(["bucket", "aligned"], observed=True)])
        parts.append("\nTape-only counterfactual (direction = sign of the flow, every candidate with |flow| >= 0.10):\n" + t.to_string(index=False))
    taken = decisions[decisions["group"] == "entered"]
    if len(taken):
        cols = ["cycle_id", "symbol", "direction", "entry", "flow", "prints", "rsi", "atr60", "atr_close"]
        parts.append("\nEntries taken:\n" + taken[cols].round(2).to_string(index=False))
    passes = decisions[decisions["group"] == "llm_pass"]
    if len(passes):
        cols = ["cycle_id", "symbol", "direction", "flow", "prints", "rsi", "atr60", "atr_close"]
        parts.append("\nDecider passes (what the declined candidates did next):\n" + passes[cols].round(2).to_string(index=False))
    return "\n".join(parts)


def write_html(text: str, out: Path, day: str) -> None:
    style = ("<style>body{font-family:Segoe UI,Helvetica,Arial,sans-serif;max-width:1180px;margin:0 auto;padding:24px;color:#E4EAEF;background:#0F1519}"
             "h1{font-size:1.5rem;margin:0 0 6px}p{color:#93A1AD;font-size:.92rem;line-height:1.45}a{color:#5CC0D2}"
             "pre{background:#161E25;border:1px solid #26313A;border-radius:6px;padding:14px;font-size:.8rem;line-height:1.35;overflow-x:auto}</style>")
    body = (f"<h1>PACA rule audit — {html.escape(day)}</h1>"
            "<p>Every candidate of every cycle graded against what the underlying did next (60 min and the session close, ATR units "
            "signed by the trade direction). Groups: entered, no_spread (chosen but no tradeable chain), llm_pass (offered, declined), "
            "and one group per blocking rule. The tape-only table asks what following the flow sign alone would have done. "
            'Option P&amp;L is not modelled. <a href="./">picks page</a> · <a href="walkforward.html">walk-forward</a></p>'
            f"<pre>{html.escape(text)}</pre>")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(f"<!doctype html><html><head><meta charset=utf-8><title>PACA rule audit {html.escape(day)}</title>{style}</head><body>{body}</body></html>", encoding="utf-8")


@app.command()
def run(
    journal: Path = typer.Option(Path("logs") / "cycles.jsonl", help="Cycle journal to audit."),
    date: str = typer.Option(datetime.now(timezone.utc).strftime("%Y-%m-%d"), "--date", help="Session date (UTC cycle ids), YYYY-MM-DD."),
    html_out: Path | None = typer.Option(None, "--html", help="Also write the report as a dark HTML page."),
    deploy: bool = typer.Option(False, "--deploy", help="Push the page's folder to SURGE_DOMAIN_BACKTEST."),
) -> None:
    """Grade the session's entries, passes and blocked candidates against later prices."""
    cycles = load_cycles(journal, date)
    rows = decision_rows(cycles)
    config = broker.load_config()
    _, stock, _ = broker.build_clients(config)
    day = datetime.fromisoformat(date).replace(tzinfo=timezone.utc)
    cache: dict[str, pd.DataFrame | None] = {}

    def bars_for(symbol: str):
        if symbol not in cache:
            try:
                cache[symbol] = backtest_tape.fetch_bars(stock, symbol, day + timedelta(hours=8), day + timedelta(hours=22))
            except broker.BrokerError as error:
                logger.warning("{}: {}", symbol, error)
                cache[symbol] = None
        return cache[symbol]

    text = report(grade(rows, bars_for), date)
    typer.echo(text)
    if html_out:
        write_html(text, html_out, date)
        typer.echo(f"\nwrote {html_out}")
        if deploy:
            import dashboard

            domain = os.environ.get("SURGE_DOMAIN_BACKTEST", "").strip()
            surge = dashboard.surge_binary()
            if not domain or surge is None:
                typer.echo("deploy skipped: SURGE_DOMAIN_BACKTEST unset or surge not installed")
                return
            (html_out.parent / "CNAME").write_text(domain + "\n", encoding="utf-8")
            subprocess.run([surge, str(html_out.parent), domain], check=False, timeout=120)
            typer.echo(f"deployed https://{domain}/{html_out.name}")


if __name__ == "__main__":
    app()
