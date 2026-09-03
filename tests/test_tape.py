"""tape.py: pure order-flow math. No I/O, no settings."""

import pytest

import tape
from data_models import Event

CALL = Event(kind="breakout_up", direction="CALL")
PUT = Event(kind="breakout_down", direction="PUT")


def test_tick_rule_classifies_upticks_downticks_and_carries_zero_ticks():
    # 10.0 first print (no direction) | 10.1 uptick buy 5 | 10.1 flat -> carries buy 3
    # | 10.0 downtick sell 4 | 10.0 flat -> carries sell 2
    trades = [(10.0, 1), (10.1, 5), (10.1, 3), (10.0, 4), (10.0, 2)]
    stats = tape.tick_rule(trades, min_trades=1)
    assert stats.buy_volume == 8 and stats.sell_volume == 6 and stats.trades == 5
    assert stats.imbalance == pytest.approx((8 - 6) / 14)


def test_tick_rule_unknown_when_too_few_prints_or_no_signed_volume():
    assert tape.tick_rule([(10.0, 1), (10.1, 1)], min_trades=5).imbalance is None
    assert tape.tick_rule([], min_trades=0).imbalance is None
    assert tape.tick_rule([(10.0, 7)], min_trades=0).imbalance is None  # one print has no direction


def test_l1_imbalance():
    assert tape.l1_imbalance(300, 100) == pytest.approx(0.5)
    assert tape.l1_imbalance(None, 100) is None
    assert tape.l1_imbalance(0, 0) is None


@pytest.mark.parametrize(
    ("imbalance", "expected"),
    [(0.4, (CALL,)), (-0.4, (PUT,)), (0.05, ()), (0.15, (CALL,)), (-0.15, (PUT,))],
)
def test_entry_flow_events_keeps_only_agreeing_directions(imbalance, expected):
    assert tape.entry_flow_events((CALL, PUT), imbalance, 0.15) == expected


def test_entry_flow_events_threshold_zero_disables_the_gate_and_unknown_keeps_nothing():
    assert tape.entry_flow_events((CALL, PUT), 0.0, 0.0) == (CALL, PUT)
    assert tape.entry_flow_events((CALL, PUT), None, 0.0) == (CALL, PUT)  # gate off: flow not required
    assert tape.entry_flow_events((CALL, PUT), None, 0.15) == ()


def test_flow_against_needs_a_full_streak_of_opposing_readings():
    # a call spread ("C") is opposed by selling flow (<= -min); a put spread by buying flow
    assert tape.flow_against("C", [0.5, -0.3, -0.2], bars=2, min_imbalance=0.15) is True
    assert tape.flow_against("C", [-0.3, 0.1], bars=2, min_imbalance=0.15) is False
    assert tape.flow_against("C", [-0.3], bars=2, min_imbalance=0.15) is False  # not enough readings: no conviction yet
    assert tape.flow_against("C", [-0.3, None], bars=2, min_imbalance=0.15) is None  # unknown reading
    assert tape.flow_against("P", [0.3, 0.16], bars=2, min_imbalance=0.15) is True
    assert tape.flow_against("P", [-0.3, -0.2], bars=2, min_imbalance=0.15) is False


def test_opposing_streak_counts_trailing_opposing_readings():
    assert tape.opposing_streak("C", [0.5, -0.3, -0.2], 0.15) == 2
    assert tape.opposing_streak("C", [-0.3, 0.1], 0.15) == 0
    assert tape.opposing_streak("P", [0.3, None, 0.4], 0.15) == 1
