"""Evidence tests: assembly of features, news and gates."""

import traceback
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from regimepilot import evidence as evidence_module
from regimepilot.evidence import EvidenceError, build_evidence, observe_evidence
from regimepilot.features import build_feature_packet
from regimepilot.gates import evaluate_gates
from regimepilot.history import HistoryError
from regimepilot.models import EvidencePacket, GatesEvidence, NewsItem, NewsPacket, OhlcvBar
from regimepilot.news import NewsError, unavailable_news_packet

NY = ZoneInfo("America/New_York")
SESSION = date(2026, 8, 24)
PREVIOUS_SESSION = date(2026, 8, 21)


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


def daily(day, close):
    return OhlcvBar(
        timestamp=datetime(day.year, day.month, day.day, 0, 0, tzinfo=NY),
        close=close,
    )


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


def build_features(**overrides):
    kwargs = dict(
        observed_at=OBSERVED_AT,
        minute_bars=main_bars(),
        daily_bars=[daily(date(2026, 8, 20), 240.0), daily(PREVIOUS_SESSION, 250.0)],
        bid=99.90,
        ask=100.10,
        market_is_open=True,
        session_close_at=et(16, 0),
    )
    kwargs.update(overrides)
    return build_feature_packet(**kwargs)


def build_news(**overrides):
    kwargs = dict(
        observed_at=OBSERVED_AT,
        available=True,
        item_count=1,
        items=(
            NewsItem(
                id=1,
                headline="SPY steady as Fed speakers wait",
                summary="Index ETFs were little changed.",
                age_minutes=15.0,
                symbols=("SPY",),
                source="benzinga",
            ),
        ),
    )
    kwargs.update(overrides)
    return NewsPacket(**kwargs)


class FakeTradingClient:
    pass


class FakeDataClient:
    pass


class FakeNewsClient:
    pass


def test_build_evidence_combines_features_news_and_gates():
    features = build_features()
    news = build_news()
    gates = evaluate_gates(features)
    packet = build_evidence(features, news, gates)

    assert isinstance(packet, EvidencePacket)
    assert packet.gates.passed is True
    assert packet.underlying.return_15m == features.return_15m
    assert packet.news.item_count == 1
    assert packet.account.has_open_option_position is False


def test_observe_evidence_degrades_news_failures_without_aborting(monkeypatch):
    features = build_features()

    monkeypatch.setattr(
        evidence_module,
        "observe_features",
        lambda *args, **kwargs: features,
    )

    def fail_news(*args, **kwargs):
        raise NewsError("failed to read SPY news: RuntimeError")

    monkeypatch.setattr(evidence_module, "observe_news", fail_news)

    packet = observe_evidence(FakeTradingClient(), FakeDataClient(), FakeNewsClient())

    assert packet.news.available is False
    assert packet.news.item_count == 0
    assert packet.gates.passed is True


def test_observe_evidence_wraps_history_failures(monkeypatch):
    def fail_features(*args, **kwargs):
        raise HistoryError("failed to read SPY minute bars: RuntimeError")

    monkeypatch.setattr(evidence_module, "observe_features", fail_features)

    with pytest.raises(EvidenceError) as excinfo:
        observe_evidence(
            FakeTradingClient(),
            FakeDataClient(),
            FakeNewsClient(),
        )

    assert "minute bars" in str(excinfo.value)
    assert "RuntimeError" in str(excinfo.value)
    assert "SUPER-SECRET" not in traceback.format_exc()


def test_evidence_packet_is_frozen_and_serializable():
    features = build_features()
    news = unavailable_news_packet(observed_at=OBSERVED_AT)
    gates = evaluate_gates(features)
    packet = build_evidence(features, news, gates)
    serialized = packet.model_dump_json()

    assert EvidencePacket.model_validate_json(serialized) == packet


def test_the_evidence_module_exposes_no_trading_or_execution_helper():
    forbidden = (
        "submit", "cancel", "replace", "close_position", "close_all", "exercise",
        "order", "buy_call", "buy_put", "position", "size", "risk", "decide",
    )
    offenders = [
        name
        for name in dir(evidence_module)
        if any(word in name.lower() for word in forbidden)
    ]
    assert offenders == []


def test_evidence_gates_carry_no_threshold_based_labels():
    """The briefing sent to the LLM must not include vol_regime / session_phase."""
    assert set(GatesEvidence.model_fields) == {"passed", "hold_reason", "momentum_align"}

    packet = build_evidence(build_features(), build_news(), evaluate_gates(build_features()))
    assert "vol_regime" not in packet.model_dump_json()
    assert "session_phase" not in packet.model_dump_json()
