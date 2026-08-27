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


API_KEY = "SUPER-SECRET-KEY"
SECRET_KEY = "SUPER-SECRET-SECRET"
SPY_CALL = "SPY260902C00765000"


class FakeAccount:
    id = "11112222-3333-4444-5555-666677778888"
    equity = "100000.55"
    options_buying_power = "98000.75"


class FakePosition:
    def __init__(self, symbol=SPY_CALL, asset_class="us_option"):
        self.symbol = symbol
        self.asset_class = asset_class
        self.qty = "1"
        self.side = "long"


class FakeOrder:
    def __init__(self, symbol=SPY_CALL, asset_class="us_option"):
        self.id = "aaaabbbb-cccc-dddd-eeee-ffff00001111"
        self.symbol = symbol
        self.asset_class = asset_class
        self.qty = "1"
        self.side = "buy"
        self.status = "new"
        self.legs = None


class FakeTradingClient:
    """Exactly the three account reads Phase 5A makes. Features are stubbed
    separately, so nothing else is ever called on it."""

    def __init__(self, *, positions=(), orders=()):
        self._positions = list(positions)
        self._orders = list(orders)
        # Deliberately carries credentials so the leak test below is meaningful.
        self.api_key = API_KEY
        self.secret_key = SECRET_KEY

    def get_account(self):
        return FakeAccount()

    def get_all_positions(self):
        return self._positions

    def get_orders(self, request):
        return self._orders


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


def _stub_market_reads(monkeypatch):
    monkeypatch.setattr(evidence_module, "observe_features", lambda *args, **kwargs: build_features())
    monkeypatch.setattr(evidence_module, "observe_news", lambda *args, **kwargs: build_news())


def test_observe_evidence_feeds_the_real_position_state_to_the_gate(monkeypatch):
    """The already_in_position gate reads the paper account, not a placeholder."""
    _stub_market_reads(monkeypatch)
    trading = FakeTradingClient(positions=[FakePosition(SPY_CALL)])

    packet = observe_evidence(trading, FakeDataClient(), FakeNewsClient())

    assert packet.account.has_open_option_position is True
    assert packet.gates.passed is False
    assert packet.gates.hold_reason == "already_in_position"


def test_an_open_spy_option_order_alone_does_not_hold_the_gate(monkeypatch):
    """Open orders are read and kept, but whether they block is a Phase 5B decision."""
    _stub_market_reads(monkeypatch)
    trading = FakeTradingClient(orders=[FakeOrder(SPY_CALL)])

    packet = observe_evidence(trading, FakeDataClient(), FakeNewsClient())

    assert packet.account.has_open_option_position is False
    assert packet.gates.passed is True


def test_observe_evidence_fails_closed_when_the_account_cannot_be_read(monkeypatch):
    """A failed account read is never reported as 'no position'."""
    _stub_market_reads(monkeypatch)
    trading = FakeTradingClient(positions=[FakePosition(SPY_CALL)])

    def explode():
        raise RuntimeError(f"401 unauthorized for key={API_KEY} secret={SECRET_KEY}")

    trading.get_all_positions = explode

    with pytest.raises(EvidenceError) as caught:
        observe_evidence(trading, FakeDataClient(), FakeNewsClient())

    message = str(caught.value)
    assert "positions" in message
    rendered = "".join(
        traceback.format_exception(type(caught.value), caught.value, caught.value.__traceback__)
    )
    for blob in (message, rendered):
        assert API_KEY not in blob
        assert SECRET_KEY not in blob


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
