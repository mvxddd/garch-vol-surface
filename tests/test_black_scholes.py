"""Pricing and implied-volatility inversion."""
from __future__ import annotations

import numpy as np
import pytest

from volsurface.models.black_scholes import (black76_price, greeks,
                                             implied_forward_from_parity,
                                             implied_vol, vega)

RNG = np.random.default_rng(20260902)


def test_put_call_parity():
    F, K, T, sig, r = 100.0, np.array([80., 90., 100., 110., 125.]), 0.5, 0.22, 0.04
    call = black76_price(F, K, T, sig, r=r, is_call=True)
    put = black76_price(F, K, T, sig, r=r, is_call=False)
    assert np.allclose(call - put, np.exp(-r * T) * (F - K), atol=1e-12)


def test_price_is_monotone_in_vol():
    p = black76_price(100.0, 105.0, 1.0, np.linspace(0.05, 1.5, 50), r=0.03)
    assert np.all(np.diff(p) > 0)


def test_zero_vol_collapses_to_intrinsic():
    assert black76_price(100.0, 90.0, 1.0, 0.0, r=0.0, is_call=True) == pytest.approx(10.0)
    assert black76_price(100.0, 90.0, 1.0, 0.0, r=0.0, is_call=False) == pytest.approx(0.0)
    # An expired option is worth intrinsic regardless of the vol passed in.
    assert black76_price(100.0, 90.0, 0.0, 0.9, r=0.0, is_call=True) == pytest.approx(10.0)


def test_vega_matches_finite_difference():
    F, K, T, sig, r = 100.0, np.array([85., 100., 120.]), 0.75, 0.25, 0.03
    h = 1e-6
    fd = (black76_price(F, K, T, sig + h, r=r) - black76_price(F, K, T, sig - h, r=r)) / (2 * h)
    assert np.allclose(fd, vega(F, K, T, sig, r=r), rtol=1e-4)


def test_delta_bounds_and_signs():
    g = greeks(100.0, np.array([80., 100., 130.]), 1.0, 0.25, r=0.03, is_call=True)
    assert np.all((g["delta"] > 0) & (g["delta"] < 1))
    gp = greeks(100.0, np.array([80., 100., 130.]), 1.0, 0.25, r=0.03, is_call=False)
    assert np.all((gp["delta"] < 0) & (gp["delta"] > -1))


@pytest.mark.parametrize("is_call", [True, False])
def test_implied_vol_round_trip(is_call):
    """Price with a known vol, invert, and get the same vol back."""
    n = 4000
    F = RNG.uniform(50, 300, n)
    K = F * np.exp(RNG.uniform(-0.30, 0.22, n))     # inside the pipeline's filters
    T = RNG.uniform(14 / 365, 2.0, n)
    sigma = RNG.uniform(0.06, 1.2, n)
    price = black76_price(F, K, T, sigma, r=0.04, is_call=is_call)
    iv = implied_vol(price, F, K, T, r=0.04, is_call=is_call)

    solved = np.isfinite(iv)
    assert solved.mean() > 0.97, "inverter should solve almost every sane quote"
    assert np.max(np.abs(iv[solved] - sigma[solved])) < 1e-3


def test_implied_vol_rejects_arbitrage_violating_prices():
    # Below intrinsic, and above the discounted forward: both inadmissible.
    assert np.isnan(implied_vol(0.001, 100.0, 80.0, 1.0, r=0.0, is_call=True))
    assert np.isnan(implied_vol(150.0, 100.0, 100.0, 1.0, r=0.0, is_call=True))


def test_implied_vol_rejects_unidentifiable_quotes():
    """A price with no vol sensitivity must return NaN, not an arbitrary root."""
    # Deep OTM, almost no time: vega is ~0, so any vol reprices it.
    assert np.isnan(implied_vol(1e-10, 100.0, 200.0, 0.01, r=0.04, is_call=True))


def test_implied_forward_recovers_the_forward():
    F, T, r = 137.4, 0.35, 0.045
    K = np.linspace(0.85 * F, 1.15 * F, 25)
    call = black76_price(F, K, T, 0.24, r=r, is_call=True)
    put = black76_price(F, K, T, 0.24, r=r, is_call=False)
    fwd, r2 = implied_forward_from_parity(K, call, put, r=r, T=T)
    assert fwd == pytest.approx(F, rel=1e-9)
    assert r2 > 0.999


def test_implied_forward_flags_a_bad_regression():
    K = np.array([90.0, 100.0, 110.0])
    junk = np.array([1.0, 1.0, 1.0])            # no parity structure at all
    fwd, r2 = implied_forward_from_parity(K, junk, junk, r=0.04, T=0.5)
    assert (not np.isfinite(fwd)) or r2 < 0.99
