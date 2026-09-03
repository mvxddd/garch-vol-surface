"""SVI calibration, arbitrage diagnostics and surface interpolation."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from volsurface.config import OptionsConfig, SurfaceConfig
from volsurface.data.clean import prepare_quotes
from volsurface.data.synthetic import (_svi_slice_params, _svi_total_variance,
                                       synthetic_option_chain)
from volsurface.models.surface import build_surface
from volsurface.models.svi import check_calendar_arbitrage, fit_svi

SPOT = 450.0
ASOF = pd.Timestamp("2026-09-02")


@pytest.fixture(scope="module")
def surface():
    chain = synthetic_option_chain(SPOT, asof=ASOF)
    cfg = OptionsConfig()
    quotes, forwards, _ = prepare_quotes(chain, cfg, spot=SPOT, asof=ASOF)
    return build_surface(quotes, forwards, SPOT, SurfaceConfig(), asof=ASOF,
                         risk_free_rate=cfg.risk_free_rate)


def test_svi_recovers_known_parameters():
    """Fit SVI to data generated from SVI: the parameters must come back."""
    T = 0.25
    a, b, rho, m, sig = _svi_slice_params(T)
    k = np.linspace(-0.25, 0.18, 40)
    iv = np.sqrt(_svi_total_variance(k, a, b, rho, m, sig) / T)

    fit = fit_svi(k, iv, T)
    assert fit.rmse_vol < 5e-4               # under 0.05 vol points
    assert fit.rho == pytest.approx(rho, abs=0.05)
    assert fit.b == pytest.approx(b, rel=0.10)
    assert fit.butterfly_free


def test_svi_rejects_degenerate_input():
    with pytest.raises(ValueError):
        fit_svi(np.array([0.0, 0.1]), np.array([0.2, 0.2]), 0.5)
    with pytest.raises(ValueError):
        fit_svi(np.linspace(-0.2, 0.2, 10), np.full(10, 0.2), T=0.0)


def test_calibrated_slices_are_butterfly_free(surface):
    for sl in surface.slices:
        assert getattr(sl, "butterfly_free", True), f"arbitrage at T={sl.T}"
        assert sl.min_durrleman_g >= -1e-10


def test_implied_density_is_non_negative(surface):
    k = np.linspace(-0.4, 0.3, 300)
    for sl in surface.slices:
        assert np.all(sl.risk_neutral_density(k) >= -1e-12)


def test_no_calendar_arbitrage(surface):
    assert check_calendar_arbitrage(list(surface.slices))["has_arbitrage"] is False


def test_total_variance_is_non_decreasing_in_maturity(surface):
    """The property that makes the interpolated surface calendar-arb-free."""
    k = np.linspace(-0.30, 0.20, 21)[:, None]
    T = np.linspace(0.02, 1.5, 120)[None, :]
    w = surface.total_variance(k, T)
    assert np.all(np.diff(w, axis=1) >= -1e-12)


def test_surface_reprices_its_own_quotes(surface):
    q = surface.quotes
    model_iv = surface.iv(q["k"].to_numpy(), q["T"].to_numpy())
    rmse = float(np.sqrt(np.mean((model_iv - q["iv"]) ** 2)))
    assert rmse < 0.005, f"surface RMSE {rmse*100:.2f} vol points is too large"


def test_atm_term_structure_is_in_contango(surface):
    ts = surface.atm_term_structure()
    assert ts["atm_iv"].is_monotonic_increasing


def test_delta_strike_inversion_is_self_consistent(surface):
    """The strike returned for a delta must actually have that delta."""
    from volsurface.models.black_scholes import greeks

    T = float(surface.maturities[len(surface.maturities) // 2])
    for delta, is_call in ((0.25, True), (0.25, False), (0.10, False)):
        K = surface.strike_for_delta(delta, T, is_call)
        iv = float(surface.iv_strike(K, T))
        got = greeks(float(surface.forward(T)), K, T, iv,
                     r=surface.r, is_call=is_call)["forward_delta"]
        assert abs(abs(float(got)) - delta) < 1e-3


def test_grid_is_finite_and_ordered(surface):
    k, t, iv = surface.grid(n_k=41, n_t=21)
    assert iv.shape == (21, 41)
    assert np.all(np.isfinite(iv))
    # Equity skew: downside strikes are bid over the forward at every maturity.
    # (Only the left half is monotone — every SVI smile has a minimum and turns
    # back up in the right wing, which is a feature, not a defect.)
    left = k <= -0.02
    assert np.all(np.diff(iv[:, left], axis=1) <= 1e-9)
    for row in iv:
        assert row[0] > row[np.argmin(np.abs(k))]
