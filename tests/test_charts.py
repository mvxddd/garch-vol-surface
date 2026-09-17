"""
Every chart renders, in every language, on data it will actually meet.

Why a whole file for this
-------------------------
Twenty chart functions, and before this file five of them were exercised. The
rest were verified by looking at them once — which catches a bad layout and
misses everything that breaks later. Two real failures in this project came from
exactly that gap: a dashboard build that died on a figure, and a `strict=True`
added to a `zip` that no test would have flagged if it had been wrong.

A chart is also the one component where "it ran" is nearly the whole
requirement. Nobody diffs a PNG; what matters is that the function survives real
data, degenerate data, and both translations. So these are smoke tests on
purpose, and they are deliberately exhaustive about *which* functions they
cover rather than deep about any one of them.
"""
from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from volsurface import i18n
from volsurface import portfolio as PF
from volsurface.analytics import skew as SK
from volsurface.backtest import (
    CostModel,
    compare_strategies,
    short_straddle_backtest,
)
from volsurface.config import (
    AnalyticsConfig,
    GarchConfig,
    OptionsConfig,
    SurfaceConfig,
)
from volsurface.data.clean import prepare_quotes
from volsurface.data.prices import compute_returns
from volsurface.data.synthetic import (
    synthetic_option_chain,
    synthetic_prices,
)
from volsurface.models import garch as G
from volsurface.models.surface import build_surface
from volsurface.viz import plots as P
from volsurface.viz import use_theme

SPOT, ASOF = 450.0, pd.Timestamp("2026-09-02")
LANGUAGES = ["en", "ru"]


@pytest.fixture(autouse=True)
def _clean_up():
    use_theme("light")
    yield
    plt.close("all")
    i18n.set_language("en")


@pytest.fixture(scope="module")
def surface():
    cfg = OptionsConfig()
    quotes, forwards, funnel = prepare_quotes(
        synthetic_option_chain(SPOT, asof=ASOF), cfg, spot=SPOT, asof=ASOF)
    surf = build_surface(quotes, forwards, SPOT, SurfaceConfig(), asof=ASOF,
                         risk_free_rate=cfg.risk_free_rate)
    surf._funnel = funnel          # carried along for the funnel chart
    return surf


@pytest.fixture(scope="module")
def market():
    """Returns and an implied-vol proxy, for the time-series charts."""
    prices = synthetic_prices("2018-01-01", "2026-01-01", seed=5)
    returns = compute_returns(prices)
    realised = np.sqrt(returns.pow(2).rolling(21).mean() * 252)
    implied = (realised.shift(1) * 1.2).bfill().clip(lower=0.05)
    return prices, returns, implied


@pytest.fixture(scope="module")
def garch_fit(market):
    _prices, returns, _implied = market
    return G.fit_garch(returns, vol_model="Garch", p=1, q=1, dist="t")


@pytest.fixture(scope="module")
def walk_forward(market):
    _prices, returns, _implied = market
    cfg = GarchConfig(oos_fraction=0.12, forecast_horizons=(1, 5),
                      refit_every=90)
    return G.walk_forward_forecast(returns, cfg, vol_model="Garch", p=1, q=1,
                                   dist="t", name="GARCH(1,1)-t")


@pytest.fixture(scope="module")
def backtest(market):
    _prices, returns, implied = market
    return short_straddle_backtest(implied, returns, costs=CostModel(0.3, 1.0))


def _assert_rendered(fig, tmp_path, name):
    """A figure that saves to a non-trivial PNG is a figure that drew."""
    assert fig is not None
    out = tmp_path / f"{name}.png"
    fig.savefig(out, dpi=60)
    assert out.stat().st_size > 5_000, f"{name} produced a suspiciously empty PNG"
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Surface charts
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("lang", LANGUAGES)
def test_surface_charts_render(surface, lang, tmp_path):
    i18n.set_language(lang)
    for name, fn in (
        ("heatmap", P.plot_surface_heatmap),
        ("smile_grid", P.plot_smile_grid),
        ("smile_overlay", P.plot_smile_overlay),
        ("density", P.plot_risk_neutral_density),
        ("residuals", P.plot_fit_residuals),
    ):
        _assert_rendered(fn(surface), tmp_path, f"{name}_{lang}")


@pytest.mark.parametrize("lang", LANGUAGES)
def test_term_structure_and_skew_render(surface, lang, tmp_path):
    i18n.set_language(lang)
    term = SK.term_structure_metrics(surface)
    _assert_rendered(P.plot_term_structure(term), tmp_path, f"term_{lang}")
    _assert_rendered(P.plot_skew_term(SK.skew_metrics(surface)), tmp_path,
                     f"skew_{lang}")


def test_term_structure_accepts_a_garch_overlay(surface, garch_fit, tmp_path):
    term = SK.term_structure_metrics(surface)
    garch_ts = G.forecast_term_structure(garch_fit, horizons=(1, 21, 63, 252))
    _assert_rendered(P.plot_term_structure(term, garch_ts), tmp_path, "term_garch")


def test_interactive_surface_builds(surface):
    """The plotly figure has no PNG to weigh, so check its traces instead."""
    fig = P.plot_surface_3d(surface, n_k=21, n_t=11)
    assert len(fig.data) >= 1
    assert fig.data[0].z is not None and np.isfinite(fig.data[0].z).any()


# --------------------------------------------------------------------------- #
# GARCH charts
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("lang", LANGUAGES)
def test_garch_charts_render(market, garch_fit, walk_forward, lang, tmp_path):
    i18n.set_language(lang)
    _prices, returns, _implied = market
    _assert_rendered(P.plot_conditional_vol(returns, garch_fit), tmp_path,
                     f"cond_{lang}")
    _assert_rendered(P.plot_forecast_vs_realized(walk_forward), tmp_path,
                     f"fcast_{lang}")


def test_conditional_vol_marks_the_out_of_sample_split(market, garch_fit,
                                                       walk_forward, tmp_path):
    _prices, returns, _implied = market
    oos_start = pd.Timestamp(walk_forward["date"].min())
    _assert_rendered(P.plot_conditional_vol(returns, garch_fit,
                                            oos_start=oos_start),
                     tmp_path, "cond_oos")


@pytest.mark.parametrize("metric", ["qlike", "rmse"])
def test_scorecard_renders_for_each_metric(walk_forward, metric, tmp_path):
    from volsurface.analytics.metrics import evaluate_walk_forward

    table = evaluate_walk_forward(walk_forward)
    _assert_rendered(P.plot_model_scorecard(table, metric), tmp_path,
                     f"score_{metric}")


# --------------------------------------------------------------------------- #
# Premium, screen, data quality
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("lang", LANGUAGES)
def test_premium_charts_render(surface, garch_fit, market, lang, tmp_path):
    from volsurface.analytics.vrp import current_vrp, historical_vrp

    i18n.set_language(lang)
    _prices, returns, implied = market
    garch_ts = G.forecast_term_structure(garch_fit, horizons=(1, 21, 63, 126))
    _assert_rendered(P.plot_vrp_term(current_vrp(surface, garch_ts, (21, 63))),
                     tmp_path, f"vrp_term_{lang}")
    _assert_rendered(P.plot_vrp_history(historical_vrp(implied, returns, 21)),
                     tmp_path, f"vrp_hist_{lang}")


@pytest.mark.parametrize("lang", LANGUAGES)
def test_quality_and_screen_charts_render(surface, lang, tmp_path):
    i18n.set_language(lang)
    _assert_rendered(P.plot_quote_funnel(surface._funnel.to_frame()), tmp_path,
                     f"funnel_{lang}")
    anomalies = SK.detect_anomalies(surface, AnalyticsConfig())
    _assert_rendered(P.plot_anomalies(anomalies), tmp_path, f"anom_{lang}")


@pytest.mark.parametrize("lang", LANGUAGES)
def test_anomaly_chart_handles_an_empty_screen(lang, tmp_path):
    """A clean surface is the normal outcome and must still draw something."""
    i18n.set_language(lang)
    fig = P.plot_anomalies(pd.DataFrame())
    assert fig is not None
    plt.close(fig)
    fig = P.plot_anomalies(None)
    assert fig is not None
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Portfolio and backtest
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("lang", LANGUAGES)
def test_portfolio_charts_render(surface, lang, tmp_path):
    i18n.set_language(lang)
    report = PF.risk_report(PF.example_portfolio(surface), surface)
    for by in ("tenor", "strike"):
        _assert_rendered(P.plot_vega_ladder(report[f"vega_by_{by}"], by=by),
                         tmp_path, f"ladder_{by}_{lang}")
    _assert_rendered(P.plot_stress_grid(report["stress"]), tmp_path,
                     f"stress_{lang}")


def test_vega_ladder_handles_a_book_with_no_options(surface):
    """A delta-only book has no vega to ladder; the chart must say so, not die."""
    book = PF.Portfolio([PF.Position("underlying", 100)])
    priced = PF.price_portfolio(book, surface)
    fig = P.plot_vega_ladder(PF.vega_ladder(priced, by="tenor"))
    assert fig is not None
    plt.close(fig)


@pytest.mark.parametrize("lang", LANGUAGES)
def test_backtest_charts_render(backtest, market, lang, tmp_path):
    i18n.set_language(lang)
    _prices, returns, implied = market
    _assert_rendered(P.plot_equity_curve(backtest), tmp_path, f"equity_{lang}")
    _assert_rendered(P.plot_trade_distribution(backtest), tmp_path,
                     f"dist_{lang}")
    table = compare_strategies(implied, returns, holding_days=21)
    _assert_rendered(P.plot_strategy_comparison(table), tmp_path, f"cmp_{lang}")


# --------------------------------------------------------------------------- #
# Degenerate inputs
# --------------------------------------------------------------------------- #
def test_charts_refuse_empty_input_clearly(surface):
    """An empty frame must raise something readable, not an IndexError."""
    with pytest.raises(ValueError):
        P.plot_forecast_vs_realized(pd.DataFrame(
            columns=["date", "horizon", "forecast_vol", "realized_vol", "model"]))


def test_single_expiry_surface_draws_every_surface_chart(tmp_path):
    """One expiry is the degenerate case for anything that spans maturities."""
    cfg = OptionsConfig()
    chain = synthetic_option_chain(SPOT, asof=ASOF, expiry_days=(30,))
    quotes, forwards, _ = prepare_quotes(chain, cfg, spot=SPOT, asof=ASOF)
    surf = build_surface(quotes, forwards, SPOT, SurfaceConfig(), asof=ASOF,
                         risk_free_rate=cfg.risk_free_rate)
    for name, fn in (("heat", P.plot_surface_heatmap),
                     ("grid", P.plot_smile_grid),
                     ("overlay", P.plot_smile_overlay),
                     ("density", P.plot_risk_neutral_density)):
        _assert_rendered(fn(surf), tmp_path, f"single_{name}")


def test_both_themes_produce_a_figure(surface, tmp_path):
    for theme in ("light", "dark"):
        use_theme(theme)
        _assert_rendered(P.plot_surface_heatmap(surface), tmp_path,
                         f"theme_{theme}")
