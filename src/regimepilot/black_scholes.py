"""Pure Black-Scholes pricing for a single European-style option.

Used only where no live quote exists: the backtester approximates a
historical option premium from the historical underlying price, a strike, a
time to expiration and an assumed volatility, because Alpaca's historical
option chain is not available to this project. Nothing here touches the
network or a vendor SDK, and nothing here is used on the live trading path --
``chain.py`` and ``selector.py`` price contracts from real quotes, never from
this module.

The math is textbook Black-Scholes-Merton with no dividend yield, which is
the same simplifying assumption SPY 0-10 DTE traders live with day to day
(SPY's dividend yield is a few basis points a week, immaterial at this
horizon). ``norm_cdf`` is implemented with ``math.erf`` so this module adds no
dependency beyond the standard library.
"""

from __future__ import annotations

import math

from regimepilot.models import Observation

MIN_SIGMA = 1e-6
MAX_SIGMA = 5.0
MIN_TIME_YEARS = 1e-6


def norm_cdf(x: float) -> float:
    """Standard normal CDF via the error function. No scipy dependency."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_pdf(x: float) -> float:
    """Standard normal PDF."""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _d1_d2(spot: float, strike: float, time_years: float, rate: float, sigma: float) -> tuple[float, float]:
    sigma = max(sigma, MIN_SIGMA)
    time_years = max(time_years, MIN_TIME_YEARS)
    d1 = (math.log(spot / strike) + (rate + 0.5 * sigma * sigma) * time_years) / (
        sigma * math.sqrt(time_years)
    )
    d2 = d1 - sigma * math.sqrt(time_years)
    return d1, d2


def call_price(spot: float, strike: float, time_years: float, rate: float, sigma: float) -> float:
    """Black-Scholes price of a European call. No dividend yield."""
    d1, d2 = _d1_d2(spot, strike, time_years, rate, sigma)
    return spot * norm_cdf(d1) - strike * math.exp(-rate * time_years) * norm_cdf(d2)


def put_price(spot: float, strike: float, time_years: float, rate: float, sigma: float) -> float:
    """Black-Scholes price of a European put. No dividend yield."""
    d1, d2 = _d1_d2(spot, strike, time_years, rate, sigma)
    return strike * math.exp(-rate * time_years) * norm_cdf(-d2) - spot * norm_cdf(-d1)


def option_price(
    option_type: str, spot: float, strike: float, time_years: float, rate: float, sigma: float
) -> float:
    """Dispatch to ``call_price`` or ``put_price`` by ``option_type`` ("call"/"put")."""
    if option_type == "call":
        return call_price(spot, strike, time_years, rate, sigma)
    if option_type == "put":
        return put_price(spot, strike, time_years, rate, sigma)
    raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")


def option_delta(
    option_type: str, spot: float, strike: float, time_years: float, rate: float, sigma: float
) -> float:
    """Black-Scholes delta, for reporting the simulated book's directional exposure."""
    d1, _ = _d1_d2(spot, strike, time_years, rate, sigma)
    if option_type == "call":
        return norm_cdf(d1)
    if option_type == "put":
        return norm_cdf(d1) - 1.0
    raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")


def option_vega(spot: float, strike: float, time_years: float, rate: float, sigma: float) -> float:
    """Black-Scholes vega (price change per 1.00 = 100 vol points), same for call and put."""
    d1, _ = _d1_d2(spot, strike, time_years, rate, sigma)
    return spot * norm_pdf(d1) * math.sqrt(max(time_years, MIN_TIME_YEARS))


class ImpliedVolError(ValueError):
    """Raised when no volatility in [MIN_SIGMA, MAX_SIGMA] reproduces the market price."""


def implied_volatility(
    option_type: str,
    market_price: float,
    spot: float,
    strike: float,
    time_years: float,
    rate: float,
    *,
    tolerance: float = 1e-6,
    max_iterations: int = 100,
) -> float:
    """Solve for sigma such that the model price matches ``market_price``.

    Bisection, not Newton-Raphson: Black-Scholes price is monotonic in sigma,
    so bisection always converges and never diverges on a bad initial guess,
    at the cost of a few more iterations. That trade is worth it here because
    this runs once per simulated entry in a backtest loop, not on a hot path.
    """
    lo, hi = MIN_SIGMA, MAX_SIGMA
    price_lo = option_price(option_type, spot, strike, time_years, rate, lo) - market_price
    price_hi = option_price(option_type, spot, strike, time_years, rate, hi) - market_price
    if price_lo > 0 or price_hi < 0:
        raise ImpliedVolError(
            f"market_price={market_price} is not reachable by any sigma in [{lo}, {hi}] "
            f"for spot={spot}, strike={strike}, time_years={time_years}"
        )

    for _ in range(max_iterations):
        mid = (lo + hi) / 2
        price_mid = option_price(option_type, spot, strike, time_years, rate, mid) - market_price
        if abs(price_mid) < tolerance:
            return mid
        if price_mid > 0:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


class OptionPricingInputs(Observation):
    """The inputs used for one simulated option price. Kept for journal replay."""

    option_type: str
    spot: float
    strike: float
    time_years: float
    rate: float
    sigma: float
