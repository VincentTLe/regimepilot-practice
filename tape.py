"""Tape sensor: order-flow math from recent trade prints and the L1 quote.

Pure functions, no I/O, no settings. The sensor measures who is aggressing on
the current bars — tick-rule buy volume minus sell volume over their sum —
which the literature (Cont/Kukanov/Stoikov; Chordia/Roll/Subrahmanyam) and our
own probe of 2026-09-02 show tracks the same-bar move but does not forecast the
next bar. So the engine uses it to CONFIRM or VETO entries and to demand
conviction before a reversal exit — never to predict.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Sequence

from data_models import Event


@dataclass(frozen=True)
class FlowStats:
    buy_volume: float
    sell_volume: float
    trades: int
    imbalance: float | None  # (buy − sell) / (buy + sell); None = unknown (too few prints or no signed volume)


@dataclass
class TapeState:
    """Per-underlying memory the loop keeps across cycles.

    `readings` holds the last FLOW_EXIT_BARS imbalance readings; `pending_reversal`
    counts the cycles a held-back reversal stays armed — the opposing event is a
    one-bar pulse, the tape confirmation may only complete a cycle later.
    """

    readings: deque = field(default_factory=deque)
    pending_reversal: int = 0
    peak_marks: dict = field(default_factory=dict)  # "long/short" leg symbols -> highest net mark seen (trailing exit)


def tick_rule(trades: Sequence[tuple[float, float]], min_trades: int) -> FlowStats:
    """Classify each print by the tick rule: uptick = buyer lifted the offer,
    downtick = seller hit the bid, unchanged price carries the previous direction.
    The first print has no direction. `trades` are (price, size) in time order."""
    buy = sell = 0.0
    last_price: float | None = None
    direction = 0
    for price, size in trades:
        if last_price is not None:
            if price > last_price:
                direction = 1
            elif price < last_price:
                direction = -1
            if direction > 0:
                buy += size
            elif direction < 0:
                sell += size
        last_price = price
    total = buy + sell
    count = len(trades)
    imbalance = None if count < min_trades or total <= 0 else (buy - sell) / total
    return FlowStats(buy_volume=buy, sell_volume=sell, trades=count, imbalance=imbalance)


def l1_imbalance(bid_size: float | None, ask_size: float | None) -> float | None:
    """Top-of-book size skew: (bid − ask) / (bid + ask). Advisory only."""
    if bid_size is None or ask_size is None:
        return None
    total = bid_size + ask_size
    return None if total <= 0 else (bid_size - ask_size) / total


def flow_agrees(direction: str, imbalance: float | None, min_imbalance: float) -> bool:
    """Does the tape back a CALL (buyers ≥ +min) or a PUT (sellers ≤ −min)?
    A threshold of 0 disables the check (always agrees); unknown flow never agrees."""
    if min_imbalance <= 0:
        return True
    if imbalance is None:
        return False
    return imbalance >= min_imbalance if direction == "CALL" else imbalance <= -min_imbalance


def tape_event(
    flow: FlowStats | None,
    ema_fast_dist: float | None,
    ema_slow_dist: float | None,
    *,
    min_imbalance: float,
    min_trades: int,
) -> Event | None:
    """Order-flow entry event: the tape alone, no bar pattern required.

    CALL (kind tape_buy) when the imbalance is at least +min_imbalance on at
    least min_trades prints and the last close sits above BOTH EMA anchors;
    PUT (tape_sell) mirrored. min_imbalance 0 turns the event off; an unknown
    tape or unknown anchors never fire.
    """
    if min_imbalance <= 0 or flow is None or flow.imbalance is None or flow.trades < min_trades:
        return None
    if ema_fast_dist is None or ema_slow_dist is None:
        return None
    if flow.imbalance >= min_imbalance and ema_fast_dist > 0 and ema_slow_dist > 0:
        return Event(kind="tape_buy", direction="CALL")
    if flow.imbalance <= -min_imbalance and ema_fast_dist < 0 and ema_slow_dist < 0:
        return Event(kind="tape_sell", direction="PUT")
    return None


def entry_flow_events(
    events: Sequence[Event], imbalance: float | None, min_imbalance: float
) -> tuple[Event, ...]:
    """Events whose direction the tape agrees with (entry candidacy only)."""
    return tuple(e for e in events if flow_agrees(e.direction, imbalance, min_imbalance))


def opposing_streak(option_type: str, readings: Sequence[float | None], min_imbalance: float) -> int:
    """How many of the most recent readings, counting back, oppose the held spread."""
    opposing = "PUT" if option_type == "C" else "CALL"
    streak = 0
    for reading in reversed(list(readings)):
        if reading is None or not flow_agrees(opposing, reading, min_imbalance):
            break
        streak += 1
    return streak


def flow_against(
    option_type: str,
    readings: Sequence[float | None],
    bars: int,
    min_imbalance: float,
) -> bool | None:
    """True when the last `bars` readings ALL oppose a held spread — a call
    spread ("C") is opposed by selling flow, a put spread ("P") by buying flow.
    False when any reading agrees or is neutral, or when fewer than `bars`
    readings exist yet. None when one of the readings is unknown (the caller
    decides the fallback)."""
    window = list(readings)[-bars:] if bars > 0 else []
    if any(reading is None for reading in window):
        return None  # an unknown reading: the caller falls back, never waits on missing data
    if len(window) < bars:
        return False  # not enough consecutive readings yet: no conviction
    opposing = "PUT" if option_type == "C" else "CALL"
    return all(flow_agrees(opposing, reading, min_imbalance) for reading in window)
