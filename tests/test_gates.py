"""Gate tests: deterministic pre-HOLD rules and session labels."""

import socket
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from regimepilot import gates as gates_module
from regimepilot.features import FeaturePacket, build_feature_packet
from regimepilot.gates import (
    GateResult,
    SessionLabels,
    derive_labels,
    evaluate_gates,
)
from regimepilot.models import OhlcvBar

NY = ZoneInfo("America/New_York")
SESSION = date(2026, 8, 24)
PREVIOUS_SESSION = date(2026, 8, 21)
PREVIOUS_CLOSE = 250.0


def et(hour, minute, second=0, day=SESSION):
    return datetime(day.year, day.month, day.day, hour, minute, second, tzinfo=NY)


def bar(hour, minute, close, *, open_=None, day=SESSION):
    return OhlcvBar(
        timestamp=et(hour, minute, day=day),
        open=close if open_ is None else open_,
        high=None if close is None else close,
        low=None if close is None else close,
        close=close,
        volume=1000.0,
    )


def daily(day, close, open_=None):
    return OhlcvBar(
        timestamp=datetime(day.year, day.month, day.day, 0, 0, tzinfo=NY),
        open=open_,
        close=close,
    )


DAILY_BARS = [
    daily(date(2026, 8, 20), 240.0),
    daily(PREVIOUS_SESSION, PREVIOUS_CLOSE),
]


def session_minutes(start=(9, 30), end=(10, 35), day=SESSION):
    stamp = et(*start, day=day)
    last = et(*end, day=day)
    stamps = []
    while stamp <= last:
        stamps.append(stamp)
        stamp += timedelta(minutes=1)
    return stamps


def main_bars():
    bars = []
    for stamp in session_minutes():
        hm = (stamp.hour, stamp.minute)
        close = {(9, 35): 88.0, (10, 35): 110.0}.get(hm, 100.0)
        open_ = 200.0 if hm == (9, 30) else close
        bars.append(bar(stamp.hour, stamp.minute, close, open_=open_))
    return bars


OBSERVED_AT = et(10, 36, 5)


def build(**overrides):
    kwargs = dict(
        observed_at=OBSERVED_AT,
        minute_bars=main_bars(),
        daily_bars=DAILY_BARS,
        bid=99.90,
        ask=100.10,
        market_is_open=True,
        session_close_at=et(16, 0),
    )
    kwargs.update(overrides)
    return build_feature_packet(**kwargs)


def test_a_healthy_packet_passes_all_gates():
    result = evaluate_gates(build())

    assert result.passed is True
    assert result.hold_reason is None
    assert result.labels.momentum_align == "aligned_up"
    assert result.labels.vol_regime == "high"
    assert result.labels.session_phase == "midday"


def test_market_closed_holds():
    result = evaluate_gates(build(market_is_open=False))

    assert result.passed is False
    assert result.hold_reason == "market_closed"


def test_too_close_to_close_holds():
    result = evaluate_gates(build(observed_at=et(15, 45), session_close_at=et(16, 0)))

    assert result.passed is False
    assert result.hold_reason == "too_close_to_close"


def test_stale_data_holds():
    result = evaluate_gates(build(observed_at=et(10, 40, 0)))

    assert result.passed is False
    assert result.hold_reason == "stale_data"


def test_missing_momentum_holds_when_a_return_is_unavailable():
    bars = main_bars()
    for item in bars:
        if item.timestamp == et(9, 35):
            item = bar(9, 35, None)
    # Rebuild without the 09:35 bar so return_60m goes null.
    trimmed = [item for item in bars if item.timestamp != et(9, 35)]
    result = evaluate_gates(build(minute_bars=trimmed))

    assert result.passed is False
    assert result.hold_reason == "missing_momentum"


def test_already_in_position_holds():
    result = evaluate_gates(build(), has_open_option_position=True)

    assert result.passed is False
    assert result.hold_reason == "already_in_position"


def test_mixed_momentum_is_labeled_from_opposite_sign_returns():
    from regimepilot.gates import derive_momentum_align

    assert derive_momentum_align(0.05, -0.08) == "mixed"
    assert derive_momentum_align(-0.05, 0.08) == "mixed"


def test_gate_result_is_frozen_and_closed_to_stray_fields():
    result = evaluate_gates(build())

    with pytest.raises(Exception):
        result.passed = False
    with pytest.raises(Exception):
        GateResult(**{**result.model_dump(), "extra": True})


def test_the_gates_module_never_imports_alpaca():
    source = Path(gates_module.__file__).read_text(encoding="utf-8")
    assert "alpaca" not in source.lower()


def test_gate_evaluation_makes_no_network_call(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("gates must not open a socket")

    monkeypatch.setattr(socket, "socket", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)

    assert evaluate_gates(build()).passed is True


def test_the_gates_module_exposes_no_trading_or_execution_helper():
    forbidden = (
        "submit", "cancel", "replace", "close_position", "close_all", "exercise",
        "order", "buy_call", "buy_put", "position", "size", "risk", "decide",
    )
    offenders = [
        name for name in dir(gates_module) if any(word in name.lower() for word in forbidden)
    ]
    assert offenders == []


def test_labels_survive_a_json_round_trip():
    labels = derive_labels(build())
    assert SessionLabels.model_validate_json(labels.model_dump_json()) == labels
    assert isinstance(build(), FeaturePacket)
