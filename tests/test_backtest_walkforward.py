"""backtest_walkforward: the tape gate recomputed from raw flow, the capped
portfolio replay, and the CSV split into halves."""

from __future__ import annotations

import math

import pandas as pd

import backtest_walkforward as wf
import settings


def _row(symbol="SPY", direction="CALL", flow=0.3, trades=200, events="breakout_up", bar="2026-08-03T14:30:00+00:00",
         move=1.0, date="2026-08-03"):
    return {"date": date, "bar_time": pd.Timestamp(bar), "symbol": symbol, "direction": direction, "events": events,
            "event_class": "gap/breakout" if ("gap" in events or "breakout" in events) else "macd_only",
            "flow": flow, "flow_trades": trades, "atr": 1.0, "close": 100.0, "atr_move_close": move, "atr_move_12": move,
            "x_cut1_trail1_atr": move, "tape": "agree"}


def test_agree_mask_is_symmetric_and_needs_enough_prints():
    frame = pd.DataFrame([
        _row(direction="CALL", flow=0.2), _row(direction="PUT", flow=-0.2), _row(direction="CALL", flow=-0.2),
        _row(direction="PUT", flow=0.2), _row(direction="CALL", flow=math.nan), _row(direction="CALL", flow=0.9, trades=40),
    ])
    assert wf.agree_mask(frame, 0.15, 50).tolist() == [True, True, False, False, False, False]
    assert wf.agree_mask(frame, 0.25, 50).tolist() == [False] * 6


def test_simulate_applies_the_live_caps_and_ranks_gap_first():
    t1, t2 = "2026-08-03T14:30:00+00:00", "2026-08-03T14:35:00+00:00"
    frame = pd.DataFrame([
        _row(symbol="AAA", events="macd_cross_up", flow=0.9, move=-1.0, bar=t1),   # strongest flow but macd-only: ranked after
        _row(symbol="BBB", events="breakout_up", flow=0.2, move=2.0, bar=t1),      # taken 1st
        _row(symbol="CCC", events="gap_up", flow=0.3, move=3.0, bar=t1),           # taken 2nd (per_bar 2)
        _row(symbol="BBB", events="breakout_up", flow=0.2, move=5.0, bar=t2),      # 2nd BBB entry: per_symbol 3 allows it
    ])
    summary = wf.simulate(frame, "x", ["2026-08-03", "2026-08-04"], {"2026-08-03"}, per_bar=2, per_symbol=3, total=15)
    train = summary[summary["half"] == "1_train"].iloc[0]
    assert train["entries/day"] == 3 and train["avg_day_ATR"] == 10.0  # 2 + 3 + 5, AAA never taken
    test = summary[summary["half"] == "2_test"].iloc[0]
    assert test["days"] == 1 and test["entries/day"] == 0 and test["avg_day_ATR"] == 0  # sessions without signals count


def test_load_labels_halves_and_keeps_whitelist_names_only(tmp_path):
    rows = [_row(symbol="SPY", date="2026-08-03"), _row(symbol="SPY", date="2026-08-04", bar="2026-08-04T14:30:00+00:00"),
            _row(symbol="ZZZ", date="2026-08-04", bar="2026-08-04T14:30:00+00:00")]
    csv = tmp_path / "bt.csv"
    pd.DataFrame(rows).drop(columns=["event_class"]).to_csv(csv, index=False)
    live, sessions, train_days = wf.load(csv, train=1)
    assert sessions == ["2026-08-03", "2026-08-04"] and train_days == {"2026-08-03"}
    assert set(live["symbol"]) == {"SPY"} and "ZZZ" not in settings.SYMBOLS
    assert live["half"].tolist() == ["1_train", "2_test"] and live["et_hour"].tolist() == [10, 10]


def test_write_html_escapes_the_report(tmp_path):
    out = tmp_path / "wf.html"
    wf.write_html("a < b & c", out, "x.csv", 60)
    page = out.read_text(encoding="utf-8")
    assert "a &lt; b &amp; c" in page and "<pre>" in page and "walk-forward" in page
