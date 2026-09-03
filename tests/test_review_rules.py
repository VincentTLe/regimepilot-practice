"""review_rules: decision rows from a journal record, grading against bars, and the report."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd

import review_rules as rr

STARTED = datetime(2026, 9, 3, 15, 35, 45, tzinfo=timezone.utc)  # cycle at 11:35:45 ET reads the 11:30 bar


def bars(start: datetime, closes: list[float]) -> pd.DataFrame:
    stamps = [start + timedelta(minutes=5 * i) for i in range(len(closes))]
    return pd.DataFrame({"close": closes}, index=pd.DatetimeIndex(stamps, name="timestamp"))


def record(**candidates_and_entries):
    return {"cycle_id": "20260903-153545", "started_at": STARTED.isoformat(), "market_open": True,
            "candidates": candidates_and_entries.get("candidates", []), "entries": candidates_and_entries.get("entries", [])}


def cand(symbol, gate=None, events=(), raw=(), flow=None, prints=0, ema=(1.0, 1.0), atr=1.0, rsi=55.0):
    return {"symbol": symbol, "mid": 100.0, "events": list(events), "raw_events": list(raw), "gate_block": gate,
            "flow_imbalance": flow, "flow_trades": prints, "ema_fast_dist": ema[0], "ema_slow_dist": ema[1], "atr": atr, "rsi": rsi}


def test_decision_rows_label_entries_passes_and_blocked_candidates():
    rec = record(
        candidates=[
            cand("TSLA", events=["tape_buy"], raw=["tape_buy:CALL"], flow=0.3, prints=400),          # entered
            cand("GLD", events=["macd_cross_down"], raw=["macd_cross_down:PUT"], flow=-0.5, prints=200),  # offered, declined
            cand("WMT", gate="rsi_exhausted", raw=["gap_up:CALL"], flow=0.05, prints=300),         # blocked, graded on raw events
            cand("SPY", gate="no_event", flow=-0.2, prints=500, ema=(-1.0, -2.0)),                  # tape-only only
            cand("GLD2", gate="flow_unknown", raw=["breakout_up:CALL"], flow=None),                # no tape row
        ],
        entries=[{"symbol": "TSLA", "direction": "CALL", "receipt": {"submitted": True}}],
    )
    rows = rr.decision_rows([rec])
    by = {(r["symbol"], r["group"]): r for r in rows}
    assert by[("TSLA", "entered")]["direction"] == "CALL"
    assert by[("GLD", "llm_pass")]["direction"] == "PUT"
    assert by[("WMT", "rsi_exhausted")]["direction"] == "CALL"
    assert by[("GLD2", "flow_unknown")]["direction"] == "CALL"
    assert ("SPY", "no_event") not in by  # no direction without events
    tape_only = {r["symbol"]: r for r in rows if r["group"] == "tape_only"}
    assert set(tape_only) == {"TSLA", "GLD", "SPY"}  # |flow| >= 0.10 only
    assert tape_only["SPY"]["direction"] == "PUT" and tape_only["SPY"]["aligned"] is True
    assert tape_only["TSLA"]["aligned"] is True and tape_only["GLD"]["aligned"] is False


def test_grade_measures_from_the_last_completed_bar_to_60_minutes_and_the_close():
    # bars start 11:00 ET (15:00 UTC); the cycle at 15:35:45 UTC sees the 15:30 bar as last completed
    day = bars(datetime(2026, 9, 3, 15, 0, tzinfo=timezone.utc), [100 + i for i in range(60)])  # +1 per bar, 60 bars -> ends 19:55 UTC
    rows = [{"cycle_id": "c", "started": pd.Timestamp(STARTED), "symbol": "X", "direction": "CALL", "atr": 2.0, "group": "entered"},
            {"cycle_id": "c", "started": pd.Timestamp(STARTED), "symbol": "X", "direction": "PUT", "atr": 2.0, "group": "llm_pass"}]
    graded = rr.grade(rows, lambda symbol: day)
    call = graded[graded["direction"] == "CALL"].iloc[0]
    assert call["entry"] == 106.0  # the 15:30 bar (7th bar, index 6) closes at 106
    assert call["atr60"] == 6.0  # 12 bars later: 118 -> +12 / atr 2
    assert call["atr_close"] == (159 - 106) / 2  # last bar at or before 15:55 ET
    put = graded[graded["direction"] == "PUT"].iloc[0]
    assert put["atr60"] == -6.0


def test_report_lists_groups_and_tape_only_buckets():
    df = pd.DataFrame([
        {"cycle_id": "c1", "symbol": "TSLA", "group": "entered", "direction": "CALL", "entry": 380.0, "flow": 0.3, "prints": 400, "rsi": 60.0, "atr60": 0.5, "atr_close": 0.8},
        {"cycle_id": "c1", "symbol": "GLD", "group": "llm_pass", "direction": "PUT", "entry": 400.0, "flow": -0.5, "prints": 200, "rsi": 70.0, "atr60": 1.0, "atr_close": 1.5},
        {"cycle_id": "c1", "symbol": "WMT", "group": "rsi_exhausted", "direction": "CALL", "entry": 100.0, "flow": 0.05, "prints": 300, "rsi": 85.0, "atr60": -0.2, "atr_close": -0.4},
        {"cycle_id": "c1", "symbol": "SPY", "group": "tape_only", "direction": "PUT", "entry": 640.0, "flow": -0.2, "prints": 500, "rsi": 50.0, "atr60": 0.3, "atr_close": 0.1, "aligned": True},
    ])
    text = rr.report(df, "2026-09-03")
    assert "entered" in text and "llm_pass" in text and "rsi_exhausted" in text and "0.10-0.25" in text
    assert "Entries taken" in text and "Decider passes" in text
    assert rr.report(pd.DataFrame(), "2026-09-03").startswith("2026-09-03: nothing to grade")
