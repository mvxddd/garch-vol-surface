"""GARCH estimation, forecast horizons, and evaluation metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from volsurface.analytics.metrics import (diebold_mariano, forecast_metrics,
                                          mincer_zarnowitz, qlike)
from volsurface.config import GarchConfig
from volsurface.data.synthetic import synthetic_prices
from volsurface.data.prices import compute_returns
from volsurface.models import garch as G

pytest.importorskip("arch")


@pytest.fixture(scope="module")
def returns():
    px = synthetic_prices("2016-01-01", "2026-01-01", seed=7)
    return compute_returns(px)


def test_arch_effects_are_detected(returns):
    """The generator has volatility clustering; the test must find it."""
    assert G.arch_effect_test(returns)["lm_pvalue"] < 1e-6


def test_fit_is_stationary_and_sane(returns):
    fit = G.fit_garch(returns, vol_model="Garch", p=1, q=1, dist="t")
    assert fit.converged
    assert 0.80 < fit.persistence < 1.0
    assert 0.03 < fit.long_run_vol < 1.0        # a plausible annualised vol


def test_egarch_long_run_vol_is_in_variance_space(returns):
    """EGARCH's intercept lives in log-variance; a naive read gives ~0% vol."""
    fit = G.fit_garch(returns, vol_model="EGARCH", p=1, o=1, q=1, dist="skewt")
    assert 0.03 < fit.long_run_vol < 1.0


def test_short_sample_is_rejected(returns):
    with pytest.raises(ValueError):
        G.fit_garch(returns.iloc[:100])


def test_forecast_term_structure_mean_reverts(returns):
    """Forecasts must converge toward the long-run vol as the horizon grows."""
    fit = G.fit_garch(returns, vol_model="Garch", p=1, q=1, dist="t")
    ts = G.forecast_term_structure(fit, horizons=(1, 21, 126, 252))
    assert ts["garch_vol_ann"].notna().all()
    near, far = ts["garch_vol_ann"].iloc[0], ts["garch_vol_ann"].iloc[-1]
    lr = fit.long_run_vol
    assert abs(far - lr) <= abs(near - lr) + 1e-9


def test_forward_realized_vol_has_no_lookahead(returns):
    """rv_fwd at t must use only returns strictly after t."""
    rv = G.realized_vol(returns, 5, annualize=False, forward=True)
    manual = np.sqrt(np.mean(returns.to_numpy()[1:6] ** 2))
    assert rv.iloc[0] == pytest.approx(manual)
    assert rv.iloc[-1] != rv.iloc[-1] or np.isnan(rv.iloc[-1])   # tail is NaN


def test_walk_forward_is_out_of_sample(returns):
    cfg = GarchConfig(oos_fraction=0.15, forecast_horizons=(1, 5), refit_every=60)
    wf = G.walk_forward_forecast(returns, cfg, vol_model="Garch", p=1, q=1, dist="t")
    assert not wf.empty
    assert set(wf.columns) >= {"date", "horizon", "forecast_vol", "realized_vol"}
    assert wf["forecast_vol"].between(0.001, 5).all()
    # Forecasts should correlate positively with what actually happened.
    for h in (1, 5):
        sub = wf[wf["horizon"] == h]
        assert sub[["forecast_vol", "realized_vol"]].corr().iloc[0, 1] > 0.1


def test_qlike_is_minimised_at_the_truth():
    """
    QLIKE is minimised where the forecast variance equals E[RV^2], so the
    realised proxy must be built to have exactly that second moment:
    E[(sigma|z|)^2] = sigma^2 because E[z^2] = 1.
    """
    rng = np.random.default_rng(3)
    true = 0.20
    realized = true * np.abs(rng.standard_normal(200_000))
    losses = {s: qlike(np.full_like(realized, s), realized)
              for s in (0.10, 0.15, 0.20, 0.28, 0.40)}
    assert min(losses, key=losses.get) == 0.20


def test_mincer_zarnowitz_on_a_perfect_forecast():
    f = np.linspace(0.1, 0.5, 300)
    mz = mincer_zarnowitz(f, f)
    assert mz["alpha"] == pytest.approx(0.0, abs=1e-9)
    assert mz["beta"] == pytest.approx(1.0, abs=1e-9)
    assert mz["r_squared"] == pytest.approx(1.0, abs=1e-9)


def test_forecast_metrics_detect_bias():
    a = np.full(500, 0.20)
    m = forecast_metrics(a + 0.05, a)
    assert m["bias"] == pytest.approx(0.05)
    assert m["rmse"] == pytest.approx(0.05)


def test_diebold_mariano_direction_and_significance():
    rng = np.random.default_rng(11)
    good = np.abs(rng.standard_normal(1000)) * 0.5
    bad = good + 0.5                      # strictly worse on every observation
    dm = diebold_mariano(good, bad, horizon=1)
    assert dm["dm_stat"] < 0              # negative => model A wins
    assert dm["p_value"] < 0.01
