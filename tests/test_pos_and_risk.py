from datetime import date

import pytest

import pos_and_risk
from data_models import LegPosition, LegQuote, OpenSpread

EXP = date(2026, 9, 11)
TODAY = date(2026, 8, 31)


def leg(symbol="SPY260911C00650000", underlying="SPY", qty=1, price=3.0, strike=650.0, option_type="C"):
    return LegPosition(
        symbol=symbol, underlying=underlying, expiration=EXP, option_type=option_type,
        strike=strike, qty=qty, avg_entry_price=price,
    )


def quote(bid, ask):
    return LegQuote(symbol="X", strike=0.0, bid=bid, ask=ask, implied_vol=0.2,
                    open_interest=500, quote_time=None)


# --- OCC parsing ---

def test_parse_occ_round_trip():
    assert pos_and_risk.parse_occ("SPY260911C00650000") == ("SPY", EXP, "C", 650.0)
    assert pos_and_risk.parse_occ("IWM260911P00230500") == ("IWM", EXP, "P", 230.5)


@pytest.mark.parametrize("bad", ["", "SPY", "SPY260911X00650000", "SPY261341C00650000", "spy 650 call"])
def test_parse_occ_rejects_garbage(bad):
    assert pos_and_risk.parse_occ(bad) is None


# --- pairing ---

def test_pair_spreads_happy_path():
    legs = (
        leg("SPY260911C00650000", qty=2, price=6.0, strike=650.0),
        leg("SPY260911C00655000", qty=-2, price=3.5, strike=655.0),
    )
    spreads, warnings = pos_and_risk.pair_spreads(legs)
    assert warnings == []
    assert len(spreads) == 1
    spread = spreads[0]
    assert spread.qty == 2
    assert spread.net_entry_debit == 2.5
    assert spread.long_symbol == "SPY260911C00650000"


@pytest.mark.parametrize(
    "legs",
    [
        (leg(qty=1),),  # naked single leg
        (leg(qty=2), leg("SPY260911C00655000", qty=-1, strike=655.0)),  # unequal qty
        (leg(qty=1), leg("SPY260911C00655000", qty=1, strike=655.0)),  # two longs
    ],
)
def test_pair_spreads_warns_and_never_touches_odd_shapes(legs):
    spreads, warnings = pos_and_risk.pair_spreads(tuple(legs))
    assert spreads == []
    assert len(warnings) == 1


def test_pair_spreads_non_debit_pair_has_unknown_debit():
    legs = (
        leg("SPY260911C00650000", qty=1, price=2.0, strike=650.0),
        leg("SPY260911C00655000", qty=-1, price=3.0, strike=655.0),  # credit, not ours
    )
    spreads, _ = pos_and_risk.pair_spreads(legs)
    assert spreads[0].net_entry_debit is None


# --- mechanical exits ---

def spread(debit=2.0, expiration=EXP):
    return OpenSpread(
        underlying="SPY", expiration=expiration, option_type="C",
        long_symbol="L", short_symbol="S", qty=1, net_entry_debit=debit,
    )


def test_exit_at_exact_stop_threshold():
    # entry debit 2.00, stop at mark <= 1.00
    decision = pos_and_risk.exit_decision(spread(), quote(1.4, 1.6), quote(0.4, 0.6), TODAY)
    assert decision is not None and decision.reason == "stop" and decision.net_mark == 1.0


def test_exit_at_exact_take_profit_threshold():
    # entry debit 2.00, TP at mark >= 4.00
    decision = pos_and_risk.exit_decision(spread(), quote(4.9, 5.1), quote(0.9, 1.1), TODAY)
    assert decision is not None and decision.reason == "take_profit" and decision.net_mark == 4.0


def test_hold_between_thresholds():
    assert pos_and_risk.exit_decision(spread(), quote(2.9, 3.1), quote(0.9, 1.1), TODAY) is None


def test_expiry_exit_at_dte_boundary():
    near = spread(expiration=date(2026, 9, 2))  # DTE 2
    decision = pos_and_risk.exit_decision(near, quote(2.9, 3.1), quote(0.9, 1.1), TODAY)
    assert decision is not None and decision.reason == "expiry"
    far = spread(expiration=date(2026, 9, 3))  # DTE 3
    assert pos_and_risk.exit_decision(far, quote(2.9, 3.1), quote(0.9, 1.1), TODAY) is None


def test_expiry_exit_survives_missing_marks():
    near = spread(expiration=date(2026, 9, 1))
    decision = pos_and_risk.exit_decision(near, None, None, TODAY)
    assert decision is not None and decision.reason == "expiry" and decision.net_mark is None


def test_missing_marks_or_unknown_debit_hold_instead_of_guessing():
    assert pos_and_risk.exit_decision(spread(), None, quote(1, 1.2), TODAY) is None
    assert pos_and_risk.exit_decision(spread(), quote(1, 1.2), quote(None, None), TODAY) is None
    assert pos_and_risk.exit_decision(spread(debit=None), quote(0.1, 0.2), quote(0.0, 0.1), TODAY) is None


# --- risk sizing ---

def test_open_premium_at_risk_sums_or_refuses():
    known = [spread(debit=2.0), spread(debit=1.0)]
    assert pos_and_risk.open_premium_at_risk(known) == 300.0
    assert pos_and_risk.open_premium_at_risk([spread(debit=None)]) is None


def test_size_entry_caps_on_100k_equity():
    # per-entry cap 0.5% of 100k = $500; debit $2.00 -> $200/contract -> qty 2
    qty, reason = pos_and_risk.size_entry(2.0, 100_000.0, 0.0, 0.0)
    assert (qty, reason) == (2, None)


def test_size_entry_refusals():
    assert pos_and_risk.size_entry(2.0, None, 0.0, 0.0) == (0, "unknown_equity")
    assert pos_and_risk.size_entry(2.0, 0.0, 0.0, 0.0) == (0, "unknown_equity")
    assert pos_and_risk.size_entry(2.0, 100_000.0, None, 0.0) == (0, "unknown_open_risk")
    assert pos_and_risk.size_entry(0.0, 100_000.0, 0.0, 0.0) == (0, "bad_debit")
    assert pos_and_risk.size_entry(6.0, 100_000.0, 0.0, 0.0) == (0, "risk_caps")  # $600 > $500


def test_size_entry_cycle_and_total_room():
    # cycle cap 1% = $1000; already spent $900 -> only $100 left -> qty 0 at $2 debit
    assert pos_and_risk.size_entry(2.0, 100_000.0, 0.0, 900.0) == (0, "risk_caps")
    # total cap 10% = $10k; open risk $9,900 -> $100 room -> refused
    assert pos_and_risk.size_entry(2.0, 100_000.0, 9_900.0, 0.0) == (0, "risk_caps")
    # open risk exactly at cap -> refused
    assert pos_and_risk.size_entry(2.0, 100_000.0, 10_000.0, 0.0) == (0, "risk_caps")
