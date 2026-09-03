"""Draw the decider's backtest picks on 5-minute candles (read-only research page).

    uv run --env-file .env backtest_chart.py --csv logs/backtest_tape_30d_llm2.csv [--out surge_artifacts/paca-backtest/index.html] [--deploy]

Reads a backtest_tape.py CSV produced with --llm, refetches the session's 5m
IEX bars for every LLM pick, and writes one static HTML page: a summary table
plus one candlestick chart per pick with the entry bar, the exit under the
"cut 1 ATR / trail 1 ATR" rule set, and the session close. Plotly is loaded
from its CDN like the live candles page. Nothing here touches trading.
"""

from __future__ import annotations

import html
import json
import os
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import typer
from loguru import logger

import backtest_tape
import broker
from cli import setup_logging

DEFAULT_OUT = Path("surge_artifacts") / "paca-backtest" / "index.html"
RULE = "cut1_trail1"

app = typer.Typer(add_completion=False)


def pick_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if "llm_pick" not in frame:
        raise typer.BadParameter("this CSV has no llm_pick column: run backtest_tape.py with --llm first")
    picks = frame[frame["llm_pick"] == True].copy()  # noqa: E712 - CSV booleans
    picks["bar_time"] = pd.to_datetime(picks["bar_time"], utc=True)
    return picks.sort_values("bar_time")


def chart_block(n: int, row: pd.Series, bars: pd.DataFrame) -> str:
    entry_ts = row["bar_time"]
    entry_pos = bars.index.get_indexer([entry_ts])[0]
    if entry_pos < 0:
        return f"<section><h2>{n}. {html.escape(row['symbol'])} {row['date']}</h2><p>bars not found</p></section>"
    held = int(row[f"x_{RULE}_bars"])
    exit_pos = min(entry_pos + held, len(bars) - 1)
    session = bars.iloc[max(0, entry_pos - 30): min(len(bars), entry_pos + 60)]
    et = [ts.tz_convert("America/New_York") for ts in session.index]
    x = [ts.strftime("%Y-%m-%d %H:%M") for ts in et]
    stamp = lambda ts: ts.tz_convert("America/New_York").strftime("%Y-%m-%d %H:%M")  # noqa: E731
    traces = [
        {"type": "candlestick", "x": x, "open": session["open"].tolist(), "high": session["high"].tolist(),
         "low": session["low"].tolist(), "close": session["close"].tolist(), "name": row["symbol"],
         "increasing": {"line": {"color": "#5FCB8F"}}, "decreasing": {"line": {"color": "#F08A7E"}}},
        {"type": "scatter", "mode": "markers", "x": [stamp(entry_ts)], "y": [float(row["close"])],
         "marker": {"symbol": "triangle-up" if row["direction"] == "CALL" else "triangle-down", "size": 16,
                    "color": "#5CC0D2"}, "name": f"LLM entry {row['direction']}"},
        {"type": "scatter", "mode": "markers", "x": [stamp(bars.index[exit_pos])],
         "y": [float(bars["close"].iloc[exit_pos])],
         "marker": {"symbol": "x", "size": 14, "color": "#E4B25C"}, "name": f"exit ({RULE})"},
    ]
    layout = {"height": 380, "margin": {"l": 44, "r": 20, "t": 34, "b": 30},
              "paper_bgcolor": "#161E25", "plot_bgcolor": "#161E25", "font": {"color": "#C9D3DC", "size": 12},
              "xaxis": {"rangeslider": {"visible": False}, "gridcolor": "#26313A", "zerolinecolor": "#26313A",
                        "type": "category", "nticks": 12, "title": {"text": "ET", "font": {"size": 11}}},
              "yaxis": {"gridcolor": "#26313A", "zerolinecolor": "#26313A"},
              "legend": {"orientation": "h", "y": 1.08, "x": 0},
              "title": {"text": f"{row['symbol']} {row['date']} · {row['direction']} · {row['events']}", "font": {"size": 14, "color": "#E4EAEF"}}}
    move = float(row[f"x_{RULE}_atr"])
    thesis = html.escape(str(row.get("llm_thesis", "")))
    verdict = f"+{move:.2f}" if move > 0 else f"{move:.2f}"
    return (
        f"<section><h2>{n}. {html.escape(row['symbol'])} {row['date']} {row['direction']}</h2>"
        f"<p class=meta>flow {row['flow']:+.2f} on {int(row['flow_trades'])} prints · RSI {row['rsi']} · "
        f"exit after {held} bars at <b>{verdict} ATR</b> (60 min: {float(row['atr_move_12']):+.2f}, close: {float(row['atr_move_close']):+.2f})</p>"
        f"<p class=thesis>“{thesis}”</p>"
        f"<div id=chart{n}></div><script>Plotly.newPlot('chart{n}', {json.dumps(traces)}, {json.dumps(layout)}, {{displaylogo:false}});</script></section>"
    )


def build_page(frame: pd.DataFrame, picks: pd.DataFrame, blocks: list[str], csv_name: str) -> str:
    asked = frame[frame.get("llm_asked", False) == True] if "llm_asked" in frame else frame  # noqa: E712
    points = asked.groupby(["date", "bar_time"]).ngroups if len(asked) else 0
    agree = frame[frame["tape"] == "agree"]
    summary = (
        f"<p><b>{len(picks)} picks</b> out of {points} decision points (bars with at least one tape-agree candidate) "
        f"over {frame['date'].nunique()} sessions · source {html.escape(csv_name)}.</p>"
        f"<table><tr><th>group</th><th>n</th><th>hit% 60m</th><th>avg ATR 60m</th><th>avg ATR {RULE}</th><th>avg ATR close</th></tr>"
        + "".join(
            f"<tr><td>{label}</td><td>{len(df)}</td><td>{100 * (df['atr_move_12'] > 0).mean():.0f}</td>"
            f"<td>{df['atr_move_12'].mean():+.2f}</td><td>{df[f'x_{RULE}_atr'].mean():+.2f}</td><td>{df['atr_move_close'].mean():+.2f}</td></tr>"
            for label, df in (("LLM picks", picks), ("all tape-agree candidates", agree)) if len(df)
        )
        + "</table>"
    )
    style = """<style>
:root{--bg:#0F1519;--surface:#161E25;--ink:#E4EAEF;--muted:#93A1AD;--line:#26313A;--accent:#5CC0D2}
body{font-family:Segoe UI,Helvetica,Arial,sans-serif;max-width:1100px;margin:0 auto;padding:24px;color:var(--ink);background:var(--bg)}
h1{font-size:1.5rem;margin:0 0 6px}h2{font-size:1.02rem;margin:22px 0 4px;color:var(--accent)}
table{border-collapse:collapse;font-size:.9rem;margin:10px 0 18px}th,td{border-bottom:1px solid var(--line);padding:6px 10px;text-align:left}th{color:var(--muted);font-weight:600;font-size:.78rem;text-transform:uppercase;letter-spacing:.06em}
.meta{color:var(--muted);font-size:.9rem;margin:2px 0}.thesis{font-style:italic;margin:4px 0 8px;color:#C9D3DC}
section{background:var(--surface);border:1px solid var(--line);border-radius:6px;padding:10px 14px;margin-top:14px}
b{color:#fff}</style>"""
    return (
        "<!doctype html><html><head><meta charset=utf-8><title>PACA backtest: LLM picks on candles</title>"
        '<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>' + style + "</head><body>"
        "<h1>PACA tape/paca — the decider's backtest picks on 5-minute candles</h1>"
        "<p class=meta>Replay of the live entry gates and the real Featherless GLM decider on past sessions (IEX bars and prints). "
        "Entry triangle = the bar the LLM chose; × = exit under stop 1 ATR / trail 1 ATR on closes; options P&amp;L is not modelled.</p>"
        + summary + "".join(blocks) + "</body></html>"
    )


@app.command()
def build(
    csv: Path = typer.Option(..., help="backtest_tape.py CSV produced with --llm."),
    out: Path = typer.Option(DEFAULT_OUT, help="HTML page to write."),
    deploy: bool = typer.Option(False, "--deploy", help="Push the page to SURGE_DOMAIN_BACKTEST."),
) -> None:
    """Build (and optionally publish) the picks-on-candles page."""
    setup_logging()
    frame = pd.read_csv(csv)
    picks = pick_rows(frame)
    config = broker.load_config()
    _, stock, _ = broker.build_clients(config)
    blocks = []
    cache: dict[tuple[str, str], pd.DataFrame] = {}
    for n, (_, row) in enumerate(picks.iterrows(), start=1):
        key = (row["symbol"], row["date"])
        if key not in cache:
            day = datetime.fromisoformat(row["date"]).replace(tzinfo=timezone.utc)
            try:
                cache[key] = backtest_tape.fetch_bars(stock, row["symbol"], day + timedelta(hours=8), day + timedelta(hours=22))
            except broker.BrokerError as error:
                logger.warning("{} {}: {}", row["symbol"], row["date"], error)
                cache[key] = pd.DataFrame()
        bars = cache[key]
        blocks.append(chart_block(n, row, bars) if len(bars) else f"<section><h2>{n}. {row['symbol']} {row['date']}</h2><p>no bars</p></section>")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_page(frame, picks, blocks, csv.name), encoding="utf-8")
    typer.echo(f"wrote {out} ({len(picks)} picks)")
    if deploy:
        domain = os.environ.get("SURGE_DOMAIN_BACKTEST", "").strip()
        import dashboard

        surge = dashboard.surge_binary()
        if not domain or surge is None:
            typer.echo("deploy skipped: SURGE_DOMAIN_BACKTEST unset or surge not installed")
            return
        (out.parent / "CNAME").write_text(domain + "\n", encoding="utf-8")
        import subprocess

        subprocess.run([surge, str(out.parent), domain], check=False, timeout=120)
        typer.echo(f"deployed https://{domain}/")


if __name__ == "__main__":
    app()
