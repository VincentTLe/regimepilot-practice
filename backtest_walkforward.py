"""Walk-forward check of a backtest_tape.py CSV (read-only research, no trading).

    uv run backtest_walkforward.py [--csv logs/backtest_tape_90d.csv] [--llm-csv logs/backtest_tape_30d_llm2.csv]
                                   [--train 60] [--html surge_artifacts/paca-backtest/walkforward.html] [--deploy]

The first `train` sessions are in-sample, the rest out-of-sample. Every cut
(tape status, event class, hour, symbol, exit rules, tape thresholds, a
portfolio replay with the live caps) is printed per half, so a rule only
counts when it holds on BOTH halves — the guard against tuning on noise.
Rows are (bar, symbol, direction) signals that passed the deterministic gates;
forward moves are in ATR units signed in the trade direction. Only the
symbols in settings.yaml are counted.
"""

from __future__ import annotations

import html
import os
import subprocess
from pathlib import Path

import pandas as pd
import typer

import settings
from backtest_tape import EXIT_RULES, dollar_proxy

LIQUID = ("SPY", "QQQ", "IWM", "AAPL", "NVDA", "TSLA", "MSFT", "AMZN")  # names with a thick IEX tape
SWEEP_TRADES = (50, 100, 200)
SWEEP_IMBALANCE = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40)
DEFAULT_HTML = Path("surge_artifacts") / "paca-backtest" / "walkforward.html"

app = typer.Typer(add_completion=False)


def load(csv: Path, train: int) -> tuple[pd.DataFrame, list[str], set[str]]:
    """Live-name rows with ET hour, event class, half label and a $ proxy to the close."""
    df = pd.read_csv(csv)
    df["bar_time"] = pd.to_datetime(df["bar_time"], utc=True)
    et = df["bar_time"].dt.tz_convert("America/New_York")
    df["et_hour"] = et.dt.hour
    df["et_min"] = et.dt.hour * 60 + et.dt.minute
    df["event_class"] = df["events"].str.contains("gap|breakout").map({True: "gap/breakout", False: "macd_only"})
    sessions = sorted(df["date"].unique())
    train_days = set(sessions[:train])
    df["half"] = df["date"].map(lambda d: "1_train" if d in train_days else "2_test")
    df["usd_close"] = [dollar_proxy(a * t / c * 100, c) for a, t, c in zip(df["atr_move_close"], df["atr"], df["close"])]
    return df[df["symbol"].isin(settings.SYMBOLS)].copy(), sessions, train_days


def stats(g: pd.DataFrame) -> dict:
    n = len(g)
    if not n:
        return {"n": 0}
    return {
        "n": n,
        "hit60%": round(100 * (g["atr_move_12"] > 0).mean()),
        "atr60": round(g["atr_move_12"].mean(), 2),
        "hitclose%": round(100 * (g["atr_move_close"] > 0).mean()),
        "atr_close": round(g["atr_move_close"].mean(), 2),
        "cut1_trail1": round(g["x_cut1_trail1_atr"].mean(), 2),
        "usd_close/contract": round(g["usd_close"].mean()),
    }


def table(frame: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    keys = ["half", *by]
    rows = [{**dict(zip(keys, k if isinstance(k, tuple) else (k,))), **stats(g)} for k, g in frame.groupby(keys, sort=True)]
    return pd.DataFrame(rows)


def agree_mask(frame: pd.DataFrame, imbalance: float, min_trades: int) -> pd.Series:
    """Recompute the tape gate for other thresholds from the raw flow columns."""
    flow, trades = frame["flow"], frame["flow_trades"]
    ok = flow.notna() & (trades >= min_trades)
    call = (frame["direction"] == "CALL") & (flow >= imbalance)
    put = (frame["direction"] == "PUT") & (flow <= -imbalance)
    return ok & (call | put)


def simulate(frame: pd.DataFrame, label: str, sessions: list[str], train_days: set[str], col: str = "atr_move_close",
             *, per_bar: int = 2, per_symbol: int = 3, total: int = 15) -> pd.DataFrame:
    """Day P&L in ATR units with the live caps: at each bar take the strongest
    candidates (gap/breakout first, then |flow|) and hold them to the close."""
    frame = frame.sort_values(["date", "bar_time"])
    by_date = {d: g for d, g in frame.groupby("date")}
    days = []
    for date in sessions:
        n_open, per_sym, pnl, usd, entries = 0, {}, 0.0, 0.0, 0
        day = by_date.get(date)
        if day is not None:
            for _, bar in day.groupby("bar_time", sort=True):
                ranked = bar.assign(prio=bar["event_class"].eq("gap/breakout").astype(int), absflow=bar["flow"].abs())
                taken = 0
                for _, r in ranked.sort_values(["prio", "absflow"], ascending=False).iterrows():
                    if taken >= per_bar or n_open >= total or per_sym.get(r["symbol"], 0) >= per_symbol:
                        continue
                    taken += 1
                    n_open += 1
                    per_sym[r["symbol"]] = per_sym.get(r["symbol"], 0) + 1
                    pnl += r[col]
                    usd += dollar_proxy(r[col] * r["atr"] / r["close"] * 100, r["close"])
                    entries += 1
        days.append({"half": "1_train" if date in train_days else "2_test", "entries": entries, "atr": pnl, "usd": usd})
    res = pd.DataFrame(days)
    rows = []
    for half, g in res.groupby("half"):
        rows.append({"variant": label, "half": half, "days": len(g), "entries/day": round(g["entries"].mean(), 1),
                     "avg_day_ATR": round(g["atr"].mean(), 2), "median_day_ATR": round(g["atr"].median(), 2),
                     "pos_days%": round(100 * (g["atr"] > 0).mean()), "worst_day": round(g["atr"].min(), 1),
                     "best_day": round(g["atr"].max(), 1), "ATR/entry": round(g["atr"].sum() / max(1, g["entries"].sum()), 2),
                     "usd/day(1 contract)": round(g["usd"].mean())})
    return pd.DataFrame(rows)


def report(live: pd.DataFrame, sessions: list[str], train_days: set[str], llm: pd.DataFrame | None) -> str:
    n_train = len(train_days)
    parts = [f"sessions: {len(sessions)} (train {sessions[0]}..{sessions[n_train - 1]}, "
             f"test {sessions[n_train]}..{sessions[-1]}); live-name rows={len(live)}"]
    base = agree_mask(live, settings.FLOW_MIN_IMBALANCE, settings.FLOW_MIN_TRADES)
    parts.append(f"sanity: csv tape==agree {int((live['tape'] == 'agree').sum())} vs recomputed "
                 f"({settings.FLOW_MIN_IMBALANCE}, {settings.FLOW_MIN_TRADES}) {int(base.sum())}")
    agree = live[base]

    def section(title: str, frame: pd.DataFrame) -> None:
        parts.append(f"\n== {title}\n" + frame.to_string(index=False))

    section("tape status x half (live names)", table(live, ["tape"]))
    section("event class x half (tape-agree)", table(agree, ["event_class"]))
    section("direction x half (tape-agree)", table(agree, ["direction"]))
    section("ET hour x half (tape-agree)", table(agree, ["et_hour"]))
    section("symbol x half (tape-agree)", table(agree, ["symbol"]))
    section("exit rules x half (tape-agree, avg ATR at exit / avg bars held)", pd.DataFrame([
        {"half": half, "rule": name, "n": len(g), "avg_atr": round(g[f"x_{name}_atr"].mean(), 2),
         "hit%": round(100 * (g[f"x_{name}_atr"] > 0).mean()), "avg_bars": round(g[f"x_{name}_bars"].mean(), 1)}
        for half, g in agree.groupby("half") for name in EXIT_RULES]))
    section("reversal-exit sim x half (all live rows, 12-bar horizon)", pd.DataFrame([
        {"half": half, "n": len(g), "event_only_atr": round(g["exit_event_only_atr"].mean(), 2),
         "tape_rule_atr": round(g["exit_tape_rule_atr"].mean(), 2),
         "event_only_early%": round(100 * (g["exit_event_only_bars"] < 12).mean()),
         "tape_rule_early%": round(100 * (g["exit_tape_rule_bars"] < 12).mean())}
        for half, g in live.groupby("half")]))
    section("tape threshold sweep x half (live names, all events, agree rows only)", pd.DataFrame([
        {"min_trades": min_tr, "imb": imb, "half": half, **stats(g)}
        for min_tr in SWEEP_TRADES for imb in SWEEP_IMBALANCE
        for half, g in live[agree_mask(live, imb, min_tr)].groupby("half")]))
    gap = live["event_class"] == "gap/breakout"
    liquid = live["symbol"].isin(LIQUID)
    variants = {
        f"base({settings.FLOW_MIN_IMBALANCE},{settings.FLOW_MIN_TRADES})": live[base],
        "no_macd_only": live[base & gap],
        "imb_0.25": live[agree_mask(live, 0.25, settings.FLOW_MIN_TRADES)],
        "imb_0.30": live[agree_mask(live, 0.30, settings.FLOW_MIN_TRADES)],
        "trades_100": live[agree_mask(live, settings.FLOW_MIN_IMBALANCE, 100)],
        "skip_11h": live[base & (live["et_hour"] != 11)],
        "liquid8": live[base & liquid],
        "no_entry_after_1530": live[base & (live["et_min"] <= 15 * 60 + 30)],
        "no_macd+imb0.25": live[agree_mask(live, 0.25, settings.FLOW_MIN_TRADES) & gap],
        "no_macd+liquid8": live[base & gap & liquid],
    }
    section("portfolio sim x half (hold to close; caps 2/bar, 3/symbol, 15/day)",
            pd.concat([simulate(f, name, sessions, train_days) for name, f in variants.items()]))
    section("portfolio sim x half, exit rule cut1_trail1 instead of hold-to-close",
            pd.concat([simulate(f, name, sessions, train_days, "x_cut1_trail1_atr") for name, f in list(variants.items())[:3]]))
    if llm is not None and "llm_asked" in llm:
        llm = llm.copy()
        llm["usd_close"] = [dollar_proxy(a * t / c * 100, c) for a, t, c in zip(llm["atr_move_close"], llm["atr"], llm["close"])]
        asked = llm[llm["llm_asked"] == True]  # noqa: E712 - CSV booleans
        picks = asked[asked["llm_pick"] == True]  # noqa: E712
        gap_pick = picks["events"].str.contains("gap|breakout")
        section("LLM replay: picks vs every tape-agree candidate at the asked bars", pd.DataFrame([
            {"group": "LLM picks", **stats(picks)}, {"group": "all agree at asked bars", **stats(asked)},
            {"group": "picks gap/breakout", **stats(picks[gap_pick])}, {"group": "picks macd_only", **stats(picks[~gap_pick])}]))
    return "\n".join(parts)


def write_html(text: str, out: Path, csv_name: str, train: int) -> None:
    style = ("<style>:root{--bg:#0F1519;--ink:#E4EAEF;--muted:#93A1AD;--line:#26313A;--accent:#5CC0D2}"
             "body{font-family:Segoe UI,Helvetica,Arial,sans-serif;max-width:1180px;margin:0 auto;padding:24px;color:var(--ink);background:var(--bg)}"
             "h1{font-size:1.5rem;margin:0 0 6px}p{color:var(--muted);font-size:.92rem;line-height:1.45}a{color:var(--accent)}"
             "pre{background:#161E25;border:1px solid var(--line);border-radius:6px;padding:14px;font-size:.8rem;line-height:1.35;overflow-x:auto}</style>")
    body = (
        "<h1>PACA tape/paca — walk-forward check</h1>"
        f"<p>Source {html.escape(csv_name)}: every bar of the last sessions replayed through the live entry gates on IEX bars + prints. "
        f"The first {train} sessions are in-sample (1_train), the rest out-of-sample (2_test). Moves are in ATR units of the "
        "underlying, signed in the trade direction; a rule is kept only when it holds on both halves. "
        "usd/contract is a rough option proxy (0.4 delta, $12 friction per contract round trip). "
        '<a href="./">back to the picks-on-candles page</a></p>'
        f"<pre>{html.escape(text)}</pre>"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("<!doctype html><html><head><meta charset=utf-8><title>PACA walk-forward</title>" + style
                   + "</head><body>" + body + "</body></html>", encoding="utf-8")


@app.command()
def run(
    csv: Path = typer.Option(Path("logs") / "backtest_tape_90d.csv", help="backtest_tape.py CSV (any length)."),
    llm_csv: Path | None = typer.Option(Path("logs") / "backtest_tape_30d_llm2.csv", help="CSV produced with --llm, for the picks table (skipped if missing)."),
    train: int = typer.Option(60, min=1, help="Sessions in the in-sample half; the rest are out-of-sample."),
    html_out: Path | None = typer.Option(DEFAULT_HTML, "--html", help="Also write the report as a dark HTML page."),
    deploy: bool = typer.Option(False, "--deploy", help="Push the page's folder to SURGE_DOMAIN_BACKTEST."),
) -> None:
    """Print (and optionally publish) the walk-forward report."""
    live, sessions, train_days = load(csv, train)
    if train >= len(sessions):
        raise typer.BadParameter(f"--train {train} leaves no out-of-sample sessions (csv has {len(sessions)})")
    llm = pd.read_csv(llm_csv) if llm_csv and llm_csv.exists() else None
    text = report(live, sessions, train_days, llm)
    typer.echo(text)
    if html_out:
        write_html(text, html_out, csv.name, train)
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
