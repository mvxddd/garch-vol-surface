"""Cleaning funnel and end-to-end pipeline behaviour."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from volsurface import Config, run_pipeline
from volsurface.config import OptionsConfig
from volsurface.data.clean import prepare_quotes
from volsurface.data.synthetic import synthetic_option_chain

SPOT, ASOF = 450.0, pd.Timestamp("2026-09-02")


@pytest.fixture(scope="module")
def prepared():
    chain = synthetic_option_chain(SPOT, asof=ASOF)
    return prepare_quotes(chain, OptionsConfig(), spot=SPOT, asof=ASOF)


def test_cleaning_keeps_only_otm_quotes(prepared):
    quotes, _, _ = prepared
    calls = quotes[quotes["option_type"] == "call"]
    puts = quotes[quotes["option_type"] == "put"]
    assert (calls["strike"] >= calls["forward"]).all()
    assert (puts["strike"] < puts["forward"]).all()


def test_engineered_features_are_consistent(prepared):
    quotes, _, _ = prepared
    assert np.allclose(quotes["k"], np.log(quotes["strike"] / quotes["forward"]))
    assert np.allclose(quotes["total_variance"], quotes["iv"] ** 2 * quotes["T"])
    assert (quotes["vega"] > 0).all()


def test_inverted_iv_matches_the_generating_vol(prepared):
    """End-to-end inversion accuracy against the synthetic ground truth."""
    quotes, _, _ = prepared
    err = (quotes["iv"] - quotes["provider_iv"]).abs()
    assert err.median() < 5e-4
    assert err.max() < 5e-3          # widest wing, where the spread is largest


def test_forward_recovers_the_generating_carry(prepared):
    _, forwards, _ = prepared
    assert (forwards["forward_source"] == "parity").all()
    assert forwards["implied_q"].between(0.010, 0.016).all()   # generator used 1.3%


def test_funnel_is_monotonically_non_increasing(prepared):
    _, _, funnel = prepared
    counts = funnel.to_frame()["n_quotes"].to_numpy()
    assert np.all(np.diff(counts) <= 0)


def test_all_filters_removed_raises_a_clear_error():
    chain = synthetic_option_chain(SPOT, asof=ASOF)
    cfg = OptionsConfig(min_bid=1e9)          # nothing can pass
    with pytest.raises(ValueError, match="filtered out"):
        prepare_quotes(chain, cfg, spot=SPOT, asof=ASOF)


@pytest.mark.slow
def test_pipeline_runs_end_to_end(tmp_path):
    cfg = Config()
    cfg.data.provider = "synthetic"
    cfg.data.use_cache = False
    cfg.data.start = "2019-01-01"
    cfg.garch.specs = (("GARCH(1,1)-t", "Garch", 1, 0, 1, "t"),)
    cfg.garch.oos_fraction = 0.10
    cfg.garch.refit_every = 60
    cfg.output_dir = tmp_path
    cfg.figure_dir = tmp_path / "figures"
    cfg.report_dir = tmp_path / "reports"

    res = run_pipeline(cfg, make_figures=False)

    assert res.status["load_prices"] == "ok"
    assert all(v == "ok" for k, v in res.status.items() if k != "figures"), res.errors
    head = res.headline()
    assert head["n_expiries"] >= 5
    assert 0.01 < head["atm_30d_iv"] < 2.0
    assert head["calendar_arbitrage"] is False

    written = res.save(cfg.report_dir)
    assert (cfg.report_dir / "run_report.json").exists()
    assert len(written) > 8


def test_pipeline_survives_a_broken_option_feed(monkeypatch):
    """If the chain cannot be loaded at all, the GARCH half must still run."""
    import volsurface.pipeline as P

    def boom(*a, **k):
        raise RuntimeError("simulated provider outage")

    monkeypatch.setattr(P, "load_option_chain", boom)
    cfg = Config()
    cfg.data.provider = "synthetic"
    cfg.data.use_cache = False
    cfg.data.start = "2021-01-01"
    cfg.garch.specs = (("GARCH(1,1)-t", "Garch", 1, 0, 1, "t"),)

    res = run_pipeline(cfg, make_figures=False, run_walk_forward=False)
    assert res.status["fit_garch"] == "ok"
    assert res.status["build_surface"] == "failed"
    assert "simulated provider outage" in res.errors["build_surface"]
    assert res.best_fit is not None            # the useful half survived
