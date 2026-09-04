"""American exercise: pricing, early-exercise value, and inversion."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from volsurface.config import OptionsConfig
from volsurface.data.clean import prepare_quotes
from volsurface.data.synthetic import synthetic_option_chain
from volsurface.models.american import (binomial_price, convergence_check,
                                        early_exercise_premium,
                                        implied_vol_american)
from volsurface.models.black_scholes import (black76_price, bs_price_spot,
                                             implied_vol)


# --------------------------------------------------------------------------- #
# The tree itself
# --------------------------------------------------------------------------- #
def test_european_tree_converges_to_the_closed_form():
    """CRR is O(1/steps): each doubling should roughly halve the error."""
    table = convergence_check(step_counts=(32, 64, 128, 256, 512))
    errors = table["abs_error"].to_numpy()
    assert np.all(np.diff(errors) < 0)                  # monotonically better
    assert errors[-1] < 0.01
    ratios = errors[:-1] / errors[1:]
    assert np.all(ratios > 1.6), ratios                 # ~2x per doubling


def test_american_call_without_dividends_equals_european():
    """
    The classic result: with no dividend it is never optimal to exercise a call
    early, so the American and European prices must coincide exactly.
    """
    K = np.array([80.0, 100.0, 130.0])
    american = binomial_price(100, K, 1.0, 0.25, r=0.06, q=0.0, is_call=True,
                              steps=256, american=True)
    european = binomial_price(100, K, 1.0, 0.25, r=0.06, q=0.0, is_call=True,
                              steps=256, american=False)
    assert np.allclose(american, european, atol=1e-12)


def test_early_exercise_premium_is_never_negative():
    rng = np.random.default_rng(3)
    n = 120
    S = rng.uniform(50, 200, n)
    K = S * np.exp(rng.uniform(-0.3, 0.3, n))
    T = rng.uniform(0.05, 2.0, n)
    sigma = rng.uniform(0.10, 0.60, n)
    is_call = rng.random(n) < 0.5
    prem = early_exercise_premium(S, K, T, sigma, r=0.05, q=0.02,
                                  is_call=is_call, steps=128)
    assert np.all(prem >= -1e-9)


def test_put_premium_grows_with_moneyness_and_maturity():
    K = np.array([90.0, 100.0, 110.0, 120.0])
    prem = early_exercise_premium(100.0, K, 1.0, 0.25, r=0.06, q=0.0,
                                  is_call=False, steps=256)
    assert np.all(np.diff(prem) > 0)                     # deeper ITM => more

    T = np.array([0.25, 0.5, 1.0, 2.0])
    prem_t = early_exercise_premium(100.0, 110.0, T, 0.25, r=0.06, q=0.0,
                                    is_call=False, steps=256)
    assert np.all(np.diff(prem_t) > 0)                   # longer => more


def test_deep_itm_american_put_is_worth_at_least_intrinsic():
    """An American put can always be exercised, so it never trades below it."""
    price = float(binomial_price(60.0, 100.0, 1.0, 0.20, r=0.05, q=0.0,
                                 is_call=False, steps=256))
    assert price >= 40.0 - 1e-9


def test_degenerate_inputs_collapse_to_intrinsic():
    assert float(binomial_price(100, 90, 0.0, 0.2, is_call=True)) == pytest.approx(10.0)
    assert float(binomial_price(100, 90, 1.0, 0.0, is_call=False)) == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# Inversion
# --------------------------------------------------------------------------- #
def test_american_inversion_round_trip():
    S = np.full(6, 100.0)
    K = np.array([85.0, 92.0, 100.0, 108.0, 115.0, 125.0])
    T = np.array([0.25, 0.5, 0.75, 1.0, 1.5, 2.0])
    sigma = np.array([0.15, 0.20, 0.24, 0.28, 0.33, 0.40])
    price = binomial_price(S, K, T, sigma, r=0.04, q=0.02, is_call=False, steps=256)
    iv = implied_vol_american(price, S, K, T, r=0.04, q=0.02, is_call=False,
                              steps=256)
    assert np.nanmax(np.abs(iv - sigma)) < 1e-4


def test_american_inversion_rejects_prices_below_intrinsic():
    assert np.isnan(implied_vol_american(5.0, 100.0, 120.0, 1.0, is_call=False))
    assert np.isnan(implied_vol_american(150.0, 100.0, 100.0, 1.0, is_call=True))


def test_american_iv_is_never_above_european_iv():
    """
    The sign that matters. An American option is worth at least its European
    twin at the same vol, so reproducing one fixed market price needs a *lower*
    vol. If this test ever flips, the surface is biased upward everywhere.
    """
    S, r, q = 773.0, 0.042, 0.013
    for dte in (30, 90, 180, 378):
        T = dte / 365
        F = S * np.exp((r - q) * T)
        for K, is_call in ((S * 0.95, False), (S * 1.05, True)):
            price = float(black76_price(F, K, T, 0.18, r=r, is_call=is_call))
            iv_eu = float(implied_vol(price, F, K, T, r=r, is_call=is_call))
            iv_am = float(implied_vol_american(price, S, K, T, r=r, q=q,
                                               is_call=is_call, steps=256))
            assert iv_am <= iv_eu + 1e-4, (dte, K, is_call, iv_eu, iv_am)


def test_the_bias_is_material_for_long_dated_puts():
    """Not just signed correctly — big enough to matter at a year."""
    S, r, q, T = 773.0, 0.042, 0.013, 378 / 365
    F = S * np.exp((r - q) * T)
    K = S * 0.97
    price = float(black76_price(F, K, T, 0.18, r=r, is_call=False))
    iv_eu = float(implied_vol(price, F, K, T, r=r, is_call=False))
    iv_am = float(implied_vol_american(price, S, K, T, r=r, q=q, is_call=False,
                                       steps=256))
    assert (iv_eu - iv_am) * 100 > 0.2        # more than 0.2 vol points


def test_calls_are_barely_affected_when_carry_is_positive():
    """With q < r, early exercise of a call is worth ~nothing at any tenor."""
    S, r, q, T = 773.0, 0.042, 0.013, 1.0
    F = S * np.exp((r - q) * T)
    K = S * 1.05
    price = float(black76_price(F, K, T, 0.18, r=r, is_call=True))
    iv_eu = float(implied_vol(price, F, K, T, r=r, is_call=True))
    iv_am = float(implied_vol_american(price, S, K, T, r=r, q=q, is_call=True,
                                       steps=256))
    assert abs(iv_eu - iv_am) * 100 < 0.05


# --------------------------------------------------------------------------- #
# Pipeline integration
# --------------------------------------------------------------------------- #
def test_pipeline_accepts_american_exercise():
    spot, asof = 450.0, pd.Timestamp("2026-09-02")
    chain = synthetic_option_chain(spot, asof=asof)
    cfg = OptionsConfig(exercise_style="american", binomial_steps=64)
    quotes, _, _ = prepare_quotes(chain, cfg, spot=spot, asof=asof)
    assert not quotes.empty
    assert "early_exercise_premium" in quotes.columns
    assert (quotes["early_exercise_premium"] >= -1e-6).all()
    assert quotes["iv"].between(0.01, 3.0).all()


def test_european_remains_the_default():
    """The faster path stays default; American is an explicit opt-in."""
    assert OptionsConfig().exercise_style == "european"
