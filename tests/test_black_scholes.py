"""Black-Scholes pricing tests. No network, no vendor SDK."""

import math
import socket

import pytest

from regimepilot.black_scholes import (
    ImpliedVolError,
    call_price,
    implied_volatility,
    norm_cdf,
    option_delta,
    option_price,
    put_price,
)


def test_norm_cdf_matches_known_points():
    assert norm_cdf(0.0) == pytest.approx(0.5, abs=1e-9)
    assert norm_cdf(1.959964) == pytest.approx(0.975, abs=1e-4)


def test_call_price_positive_and_below_spot():
    price = call_price(spot=100.0, strike=100.0, time_years=7 / 365, rate=0.04, sigma=0.20)
    assert 0 < price < 100.0


def test_put_call_parity_holds():
    spot, strike, t, r, sigma = 450.0, 452.0, 7 / 365, 0.04, 0.18
    call = call_price(spot, strike, t, r, sigma)
    put = put_price(spot, strike, t, r, sigma)
    # C - P = S - K * e^(-rT), the textbook parity identity.
    lhs = call - put
    rhs = spot - strike * math.exp(-r * t)
    assert lhs == pytest.approx(rhs, abs=1e-8)


def test_deep_itm_call_approaches_intrinsic_value():
    price = call_price(spot=200.0, strike=100.0, time_years=7 / 365, rate=0.04, sigma=0.15)
    intrinsic = 200.0 - 100.0 * math.exp(-0.04 * 7 / 365)
    assert price == pytest.approx(intrinsic, abs=0.05)


def test_option_price_dispatches_by_type():
    kwargs = dict(spot=450.0, strike=450.0, time_years=7 / 365, rate=0.04, sigma=0.2)
    assert option_price("call", **kwargs) == call_price(**kwargs)
    assert option_price("put", **kwargs) == put_price(**kwargs)


def test_option_price_rejects_unknown_type():
    with pytest.raises(ValueError):
        option_price("straddle", spot=1, strike=1, time_years=0.1, rate=0.0, sigma=0.2)


def test_option_delta_call_between_zero_and_one():
    delta = option_delta("call", spot=450.0, strike=450.0, time_years=7 / 365, rate=0.04, sigma=0.2)
    assert 0.0 < delta < 1.0


def test_option_delta_put_between_minus_one_and_zero():
    delta = option_delta("put", spot=450.0, strike=450.0, time_years=7 / 365, rate=0.04, sigma=0.2)
    assert -1.0 < delta < 0.0


def test_implied_volatility_recovers_input_sigma():
    true_sigma = 0.22
    market_price = call_price(spot=450.0, strike=455.0, time_years=7 / 365, rate=0.04, sigma=true_sigma)
    recovered = implied_volatility(
        "call", market_price, spot=450.0, strike=455.0, time_years=7 / 365, rate=0.04
    )
    assert recovered == pytest.approx(true_sigma, abs=1e-4)


def test_implied_volatility_raises_when_price_unreachable():
    with pytest.raises(ImpliedVolError):
        implied_volatility(
            "call", market_price=-5.0, spot=450.0, strike=455.0, time_years=7 / 365, rate=0.04
        )


def test_black_scholes_never_touches_the_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("black_scholes must not open a socket")

    monkeypatch.setattr(socket, "socket", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)

    call_price(spot=450.0, strike=450.0, time_years=7 / 365, rate=0.04, sigma=0.2)
    implied_volatility("call", 5.0, spot=450.0, strike=450.0, time_years=7 / 365, rate=0.04)
