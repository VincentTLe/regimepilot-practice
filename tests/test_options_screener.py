from datetime import date, datetime, timedelta, timezone

import pytest

import options_screener as screener
from models import LegQuote, OpenSpread

NOW = datetime(2026, 8, 31, 15, 0, tzinfo=timezone.utc)
TODAY = NOW.date()
EXP = date(2026, 9, 11)


def leg(
    symbol="OPT", strike=100.0, bid=2.0, ask=2.05, iv=0.20, oi=500,
    stamp=NOW, **_,
):
    return LegQuote(
        symbol=symbol, strike=strike, bid=bid, ask=ask,
        implied_vol=iv, open_interest=oi, quote_time=stamp,
    )


# --- expiration ---

def test_pick_expiration_nearest_at_least_5_dte_weeklies_included():
    friday_weekly = date(2026, 9, 4)  # DTE 4 -> too near
    next_weekly = date(2026, 9, 8)  # DTE 8 -> nearest eligible
    monthly = date(2026, 9, 18)
    assert screener.pick_expiration({friday_weekly, next_weekly, monthly}, TODAY) == next_weekly
    assert screener.pick_expiration({friday_weekly}, TODAY) is None
    boundary = date(2026, 9, 5)  # DTE exactly 5 qualifies
    assert screener.pick_expiration({boundary}, TODAY) == boundary


# --- leg quality ---

@pytest.mark.parametrize(
    ("bad_leg", "reason"),
    [
        (leg(oi=99), "low_open_interest"),
        (leg(oi=None), "low_open_interest"),
        (leg(bid=None), "no_quote"),
        (leg(stamp=None), "no_quote"),
        (leg(bid=2.2, ask=2.1), "crossed_quote"),
        (leg(bid=0.0), "crossed_quote"),
        (leg(stamp=NOW + timedelta(seconds=5)), "future_quote"),
        (leg(stamp=NOW - timedelta(seconds=11)), "stale_quote"),
        (leg(bid=1.0, ask=1.2), "wide_spread"),  # ~1818 bps
        (leg(iv=None), "missing_iv"),
        (leg(iv=0.0), "missing_iv"),
    ],
)
def test_check_leg_rejections(bad_leg, reason):
    assert screener.check_leg(bad_leg, NOW) == reason


def test_check_leg_accepts_good_quote():
    assert screener.check_leg(leg(), NOW) is None


# --- enumeration ---

def chain(strikes_and_quotes):
    return {strike: q for strike, q in strikes_and_quotes.items()}


def good_chain():
    # strikes 95..105, tight quotes, all legs pass
    return {
        95.0: leg("C95", 95.0, bid=6.0, ask=6.1, iv=0.20, oi=800),
        100.0: leg("C100", 100.0, bid=3.4, ask=3.5, iv=0.21, oi=900),
        105.0: leg("C105", 105.0, bid=1.5, ask=1.55, iv=0.25, oi=700),
    }


def test_enumerate_call_spread_sides_and_pricing():
    spreads, rejections = screener.enumerate_spreads(good_chain(), "CALL", 100.0, EXP, "SPY", NOW)
    assert rejections == {}
    pair = {(s.long.symbol, s.short.symbol) for s in spreads}
    # bull call: long the lower strike, short the higher
    assert ("C95", "C100") in pair and ("C100", "C105") in pair and ("C95", "C105") in pair
    near = next(s for s in spreads if (s.long.symbol, s.short.symbol) == ("C95", "C100"))
    assert near.net_debit == round(6.1 - 3.4, 2)
    assert near.width == 5.0


def put_chain():
    # put premiums rise with strike
    return {
        95.0: leg("P95", 95.0, bid=1.5, ask=1.55, iv=0.22, oi=700),
        100.0: leg("P100", 100.0, bid=3.4, ask=3.5, iv=0.21, oi=900),
        105.0: leg("P105", 105.0, bid=6.0, ask=6.1, iv=0.20, oi=800),
    }


def test_enumerate_put_spread_reverses_sides():
    spreads, _ = screener.enumerate_spreads(put_chain(), "PUT", 100.0, EXP, "SPY", NOW)
    pair = {(s.long.symbol, s.short.symbol) for s in spreads}
    # bear put: long the higher strike, short the lower
    assert ("P105", "P100") in pair and ("P100", "P95") in pair


def test_strike_band_excludes_far_strikes():
    quotes = good_chain()
    quotes[125.0] = leg("C125", 125.0)  # outside +10% of spot 100
    spreads, _ = screener.enumerate_spreads(quotes, "CALL", 100.0, EXP, "SPY", NOW)
    assert all("C125" not in (s.long.symbol, s.short.symbol) for s in spreads)


def test_width_capped_at_three_steps():
    quotes = {
        90.0 + 5 * i: leg(f"C{i}", 90.0 + 5 * i, bid=8.0 - i, ask=8.1 - i, oi=500)
        for i in range(5)  # strikes 90..110
    }
    spreads, _ = screener.enumerate_spreads(quotes, "CALL", 100.0, EXP, "SPY", NOW)
    assert max((s.width for s in spreads), default=0) <= 15.0  # 3 steps of 5


def test_debit_sanity_rejections():
    quotes = {
        95.0: leg("A", 95.0, bid=8.2, ask=8.4, iv=0.2),
        100.0: leg("B", 100.0, bid=3.1, ask=3.2, iv=0.2),
    }
    # debit 8.4 - 3.1 = 5.3 >= width 5 -> rejected
    spreads, rejections = screener.enumerate_spreads(quotes, "CALL", 100.0, EXP, "SPY", NOW)
    assert spreads == [] and rejections.get("bad_debit") == 1


def test_bad_leg_blocks_pairs_but_not_others():
    quotes = good_chain()
    quotes[100.0] = leg("C100", 100.0, bid=3.4, ask=3.5, iv=None)  # kills any pair using 100
    spreads, rejections = screener.enumerate_spreads(quotes, "CALL", 100.0, EXP, "SPY", NOW)
    assert rejections.get("missing_iv") == 1
    assert {(s.long.symbol, s.short.symbol) for s in spreads} == {("C95", "C105")}


# --- ranking ---

def test_rank_flattest_skew_first_with_oi_tiebreak():
    flat = leg("F", 100.0, iv=0.20, oi=100)
    flat2 = leg("F2", 105.0, iv=0.20, oi=100)
    steep = leg("S", 110.0, iv=0.35, oi=10_000)
    from models import SpreadQuote

    def sq(long, short, skew):
        return SpreadQuote(
            underlying="SPY", direction="CALL", expiration=EXP,
            long=long, short=short, width=5.0, net_debit=1.0, skew=skew,
        )

    steeper = sq(flat, steep, 0.15)
    flatter = sq(flat, flat2, 0.0)
    ranked = screener.rank_spreads([steeper, flatter])
    assert ranked[0] is flatter
    # tie on skew -> higher combined OI wins
    tie_low_oi = sq(flat, flat2, 0.05)
    tie_high_oi = sq(steep, steep, 0.05)
    assert screener.rank_spreads([tie_low_oi, tie_high_oi])[0] is tie_high_oi


# --- order plans ---

def open_spread():
    return OpenSpread(
        underlying="SPY", expiration=EXP, option_type="C",
        long_symbol="LSYM", short_symbol="SSYM", qty=2, net_entry_debit=2.0,
    )


def test_entry_plan_is_deterministic_and_debit_positive():
    spreads, _ = screener.enumerate_spreads(good_chain(), "CALL", 100.0, EXP, "SPY", NOW)
    top = screener.rank_spreads(spreads)[0]
    plan = screener.build_entry_plan(top, 2, "20260831-150000")
    assert plan.client_order_id == "sp-20260831-150000-enter-SPY"
    assert plan.kind == "enter" and plan.qty == 2 and plan.limit_price > 0
    assert plan.legs[0].intent == "buy_to_open" and plan.legs[1].intent == "sell_to_open"
    again = screener.build_entry_plan(top, 2, "20260831-150000")
    assert again.client_order_id == plan.client_order_id  # duplicate prevention key


def test_exit_plan_credit_is_negative_net_price():
    long_q = leg("LSYM", 95.0, bid=6.0, ask=6.2)
    short_q = leg("SSYM", 100.0, bid=3.0, ask=3.2)
    plan = screener.build_exit_plan(open_spread(), long_q, short_q, "20260831-150000")
    assert plan is not None
    assert plan.limit_price == round(3.2 - 6.0, 2) == -2.8  # credit -> negative
    assert plan.legs[0].intent == "sell_to_close" and plan.legs[1].intent == "buy_to_close"
    assert plan.qty == 2
    assert plan.client_order_id == "sp-20260831-150000-exit-SPY-260911C"


def test_exit_plan_refuses_missing_quotes():
    assert screener.build_exit_plan(open_spread(), leg(bid=None), leg(), "cid") is None
