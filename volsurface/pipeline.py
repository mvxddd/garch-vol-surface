"""
End-to-end orchestration: config in, calibrated surface + research output out.

The pipeline is deliberately **fault-tolerant by stage**. Volatility modelling
and surface construction are independent research questions that happen to
share an underlying; if the option chain is unavailable (weekend, delisting,
rate limit) the GARCH half of the study must still run and report, and vice
versa. Every stage records its own status so a partial run is explicit rather
than silently missing sections.
"""
from __future__ import annotations

import json
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .analytics import metrics as M
from .analytics import skew as SK
from .analytics import vrp as VRP
from .config import Config
from .data import load_option_chain, load_prices, prepare_quotes
from .data.prices import compute_returns, load_vol_index, synthetic_vol_index
from .i18n import set_language
from .models import garch as G
from .models.surface import build_surface
from .utils import get_logger, timer

LOG = get_logger("volsurface.pipeline")


@dataclass
class PipelineResult:
    """Everything the pipeline produced, plus a status record per stage."""

    config: Config
    status: dict[str, str] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)

    # --- volatility modelling ---
    prices: pd.DataFrame | None = None
    returns: pd.Series | None = None
    arch_test: dict[str, float] = field(default_factory=dict)
    garch_fits: dict[str, Any] = field(default_factory=dict)
    best_fit: Any = None
    model_table: pd.DataFrame | None = None
    residual_diagnostics: dict[str, float] = field(default_factory=dict)
    walk_forward: pd.DataFrame | None = None
    evaluation: pd.DataFrame | None = None
    dm_tests: pd.DataFrame | None = None
    garch_term_structure: pd.DataFrame | None = None

    # --- surface ---
    spot: float | None = None
    chain: pd.DataFrame | None = None
    quotes: pd.DataFrame | None = None
    forwards: pd.DataFrame | None = None
    funnel: pd.DataFrame | None = None
    surface: Any = None
    slice_table: pd.DataFrame | None = None
    fit_quality: pd.DataFrame | None = None

    # --- research output ---
    skew: pd.DataFrame | None = None
    term_structure: pd.DataFrame | None = None
    vrp: pd.DataFrame | None = None
    vrp_history: pd.DataFrame | None = None
    vrp_stats: dict[str, float] = field(default_factory=dict)
    anomalies: pd.DataFrame | None = None
    figures: dict[str, str] = field(default_factory=dict)

    # ---------------------------------------------------------------- #
    def headline(self) -> dict[str, Any]:
        """The handful of numbers that summarise the whole run."""
        out: dict[str, Any] = {"ticker": self.config.data.ticker}
        if self.prices is not None and not self.prices.empty:
            out["asof"] = str(pd.Timestamp(self.prices.index[-1]).date())
            out["spot"] = round(float(self.prices["Close"].iloc[-1]), 2)
            out["synthetic_data"] = bool(self.prices.attrs.get("synthetic", False))
        if self.best_fit is not None:
            out["best_model"] = self.best_fit.name
            out["persistence"] = round(self.best_fit.persistence, 4)
            out["long_run_vol"] = round(self.best_fit.long_run_vol, 4)
        if self.evaluation is not None and not self.evaluation.empty:
            best = self.evaluation.sort_values("qlike").iloc[0]
            out["best_oos_model"] = f"{best['model']} @ {int(best['horizon'])}d"
            out["best_oos_qlike"] = round(float(best["qlike"]), 4)
        if self.surface is not None:
            out["n_expiries"] = len(self.surface.slices)
            out["n_quotes"] = 0 if self.quotes is None else int(len(self.quotes))
            out["atm_30d_iv"] = round(float(self.surface.atm_vol(30 / 365)), 4)
            cal = self.surface.calendar_arbitrage()
            out["calendar_arbitrage"] = bool(cal["has_arbitrage"])
        if self.vrp is not None and not self.vrp.empty:
            row = self.vrp.iloc[0]
            out["vrp_vol_points"] = round(float(row["vrp_vol_points"]), 2)
            out["vrp_signal"] = row["signal"]
        if self.anomalies is not None:
            out["n_anomalies"] = int(len(self.anomalies))
        return out

    def to_json(self) -> str:
        payload = {"headline": self.headline(), "status": self.status,
                   "errors": self.errors, "config": self.config.to_dict()}
        return json.dumps(payload, indent=2, default=str)

    def save(self, directory: str | Path | None = None) -> dict[str, str]:
        """Write every table to CSV plus a JSON run report. Returns the paths."""
        directory = Path(directory or self.config.report_dir)
        directory.mkdir(parents=True, exist_ok=True)
        written: dict[str, str] = {}

        tables = {
            "model_comparison": self.model_table,
            "walk_forward_forecasts": self.walk_forward,
            "forecast_evaluation": self.evaluation,
            "diebold_mariano": self.dm_tests,
            "garch_term_structure": self.garch_term_structure,
            "iv_quotes": self.quotes,
            "forwards": self.forwards,
            "quote_funnel": self.funnel,
            "svi_slices": self.slice_table,
            "surface_fit_quality": self.fit_quality,
            "skew_metrics": self.skew,
            "term_structure": self.term_structure,
            "vrp_snapshot": self.vrp,
            "vrp_history": self.vrp_history,
            "anomalies": self.anomalies,
        }
        for name, df in tables.items():
            if df is not None and len(df):
                path = directory / f"{name}.csv"
                df.to_csv(path, index=isinstance(df.index, pd.DatetimeIndex))
                written[name] = str(path)

        report = directory / "run_report.json"
        report.write_text(self.to_json())
        written["run_report"] = str(report)
        LOG.info("Wrote %d output files to %s", len(written), directory)
        return written


def _stage(result: PipelineResult, name: str, fn, required: bool = False):
    """Run one stage, recording success/failure without killing the pipeline."""
    try:
        with timer(name, LOG):
            fn()
        result.status[name] = "ok"
    except Exception as exc:
        result.status[name] = "failed"
        result.errors[name] = f"{type(exc).__name__}: {exc}"
        LOG.error("Stage '%s' failed: %s", name, exc)
        LOG.debug(traceback.format_exc())
        if required:
            raise


def run_pipeline(cfg: Config | None = None, make_figures: bool = True,
                 run_walk_forward: bool = True) -> PipelineResult:
    """
    Execute the full study.

    Parameters
    ----------
    make_figures : write the chart set to `cfg.figure_dir`.
    run_walk_forward : the honest out-of-sample evaluation. It re-estimates the
        MLE every `cfg.garch.refit_every` days for every specification, which is
        the slowest part of the run (tens of seconds to a few minutes) — set
        False for a quick surface-only pass.
    """
    cfg = cfg or Config()
    cfg.ensure_dirs()
    # Apply the display language up front, not just before rendering: a caller
    # who sets cfg.language and then builds charts himself (make_figures=False)
    # must still get his language, and the CLI summary is printed after this
    # returns.
    set_language(cfg.language)
    res = PipelineResult(config=cfg)

    # ------------------------------------------------------------------ #
    # 1. Prices and returns  (required — everything else depends on them)
    # ------------------------------------------------------------------ #
    def _prices():
        res.prices = load_prices(cfg.data)
        res.returns = compute_returns(res.prices, field=cfg.data.price_field)
        res.spot = float(res.prices["Close"].iloc[-1])
        res.arch_test = G.arch_effect_test(res.returns)
        if res.arch_test.get("lm_pvalue", 1.0) > 0.05:
            LOG.warning("ARCH-LM p-value is %.3f: little evidence of volatility "
                        "clustering — a GARCH model may not be warranted here.",
                        res.arch_test["lm_pvalue"])

    _stage(res, "load_prices", _prices, required=True)

    # ------------------------------------------------------------------ #
    # 2. GARCH estimation and model selection
    # ------------------------------------------------------------------ #
    def _garch():
        res.garch_fits = G.fit_model_suite(res.returns, cfg.garch)
        res.best_fit = G.select_best(res.garch_fits)
        res.model_table = G.comparison_table(res.garch_fits)
        res.residual_diagnostics = G.residual_diagnostics(res.best_fit)
        res.garch_term_structure = G.forecast_term_structure(
            res.best_fit, horizons=(1, 5, 21, 42, 63, 126, 252))

    _stage(res, "fit_garch", _garch)

    # ------------------------------------------------------------------ #
    # 3. Walk-forward out-of-sample validation (+ naive benchmarks)
    # ------------------------------------------------------------------ #
    def _oos():
        frames = []
        for name, vol_model, p, o, q, dist in cfg.garch.specs:
            try:
                frames.append(G.walk_forward_forecast(
                    res.returns, cfg.garch, vol_model=vol_model, p=p, o=o, q=q,
                    dist=dist, name=name))
            except Exception as exc:
                LOG.error("Walk-forward failed for %s: %s", name, exc)
        if not frames:
            raise RuntimeError("No specification produced walk-forward forecasts.")
        wf = pd.concat(frames, ignore_index=True)

        # Benchmarks the models have to beat to be worth their complexity.
        bench = M.naive_benchmarks(res.returns, cfg.garch.forecast_horizons,
                                   wf["date"].unique())
        res.walk_forward = pd.concat([wf, bench], ignore_index=True)
        res.evaluation = M.evaluate_walk_forward(res.walk_forward)
        res.dm_tests = M.compare_models_dm(res.walk_forward, loss="qlike")

    if run_walk_forward:
        _stage(res, "walk_forward", _oos)
    else:
        res.status["walk_forward"] = "skipped"

    # ------------------------------------------------------------------ #
    # 4. Option chain -> cleaned IV quotes -> calibrated surface
    # ------------------------------------------------------------------ #
    def _surface():
        res.chain = load_option_chain(cfg.data, cfg.options, spot=res.spot,
                                      prices=res.prices)
        res.quotes, res.forwards, funnel = prepare_quotes(
            res.chain, cfg.options, spot=res.spot)
        res.funnel = funnel.to_frame()
        res.surface = build_surface(res.quotes, res.forwards, res.spot,
                                    cfg.surface,
                                    risk_free_rate=cfg.options.risk_free_rate)
        res.slice_table = res.surface.slice_table()
        res.fit_quality = res.surface.fit_quality()

    _stage(res, "build_surface", _surface)

    # ------------------------------------------------------------------ #
    # 5. Skew, term structure, VRP, anomaly screen
    # ------------------------------------------------------------------ #
    def _analytics():
        if res.surface is None:
            raise RuntimeError("No surface: skipping surface analytics.")
        res.skew = SK.skew_metrics(res.surface)
        res.term_structure = SK.term_structure_metrics(res.surface)
        if res.garch_term_structure is not None:
            res.vrp = VRP.current_vrp(res.surface, res.garch_term_structure,
                                      cfg.analytics.vrp_horizons_days)

    _stage(res, "surface_analytics", _analytics)

    def _vrp_history():
        iv = load_vol_index(cfg.data)
        if iv is None:
            if not cfg.data.allow_synthetic_fallback:
                raise RuntimeError("No implied-vol index available.")
            iv = synthetic_vol_index(res.returns, seed=cfg.data.synthetic_seed)
            LOG.warning("Using a SYNTHETIC implied-vol index for the historical "
                        "VRP study — indicative only.")
        res.vrp_history = VRP.historical_vrp(iv, res.returns, horizon_days=21)
        res.vrp_stats = VRP.vrp_summary(res.vrp_history)
        LOG.info("Historical VRP: mean %.2f vol points, positive %.0f%% of days, "
                 "Newey-West t = %.1f",
                 res.vrp_stats.get("mean_vrp_vol_points", np.nan),
                 res.vrp_stats.get("pct_positive", np.nan),
                 res.vrp_stats.get("newey_west_t_stat", np.nan))

    _stage(res, "vrp_history", _vrp_history)

    def _screen():
        if res.surface is None:
            raise RuntimeError("No surface to screen.")
        res.anomalies = SK.detect_anomalies(res.surface, cfg.analytics,
                                            vrp=res.vrp_history)

    _stage(res, "anomaly_screen", _screen)

    # ------------------------------------------------------------------ #
    # 6. Figures
    # ------------------------------------------------------------------ #
    if make_figures:
        _stage(res, "figures", lambda: res.figures.update(make_all_figures(res)))
    else:
        res.status["figures"] = "skipped"

    LOG.info("Pipeline finished — %s",
             ", ".join(f"{k}:{v}" for k, v in res.status.items()))
    return res


def make_all_figures(res: PipelineResult) -> dict[str, str]:
    """Render every chart the run supports, skipping the ones it cannot."""
    import matplotlib
    matplotlib.use("Agg", force=False)         # safe in headless / CI contexts
    import matplotlib.pyplot as plt

    from . import viz
    from .viz import plots as P

    set_language(res.config.language)   # also correct when called standalone
    viz.use_theme("light")
    out_dir = Path(res.config.figure_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    made: dict[str, str] = {}

    def _try(name: str, fn, ext: str = "png"):
        path = out_dir / f"{name}.{ext}"
        try:
            fn(path)
            made[name] = str(path)
        except Exception as exc:
            LOG.warning("Figure '%s' skipped: %s", name, exc)
        finally:
            plt.close("all")

    if res.surface is not None:
        _try("iv_surface_3d", lambda p: P.plot_surface_3d(res.surface, path=p), "html")
        _try("iv_surface_heatmap", lambda p: P.plot_surface_heatmap(res.surface, path=p))
        _try("smile_grid", lambda p: P.plot_smile_grid(res.surface, path=p))
        _try("smile_overlay", lambda p: P.plot_smile_overlay(res.surface, path=p))
        _try("risk_neutral_density",
             lambda p: P.plot_risk_neutral_density(res.surface, path=p))
        _try("fit_residuals", lambda p: P.plot_fit_residuals(res.surface, path=p))
    if res.term_structure is not None:
        _try("term_structure",
             lambda p: P.plot_term_structure(res.term_structure,
                                             res.garch_term_structure, path=p))
    if res.skew is not None:
        _try("skew_term", lambda p: P.plot_skew_term(res.skew, path=p))
    if res.funnel is not None:
        _try("quote_funnel", lambda p: P.plot_quote_funnel(res.funnel, path=p))
    if res.best_fit is not None and res.returns is not None:
        oos_start = None
        if res.walk_forward is not None and not res.walk_forward.empty:
            oos_start = pd.Timestamp(res.walk_forward["date"].min())
        _try("conditional_vol",
             lambda p: P.plot_conditional_vol(res.returns, res.best_fit,
                                              oos_start=oos_start, path=p))
    if res.walk_forward is not None and not res.walk_forward.empty:
        best = (res.evaluation.sort_values("qlike")["model"].iloc[0]
                if res.evaluation is not None and not res.evaluation.empty else None)
        _try("forecast_vs_realized",
             lambda p: P.plot_forecast_vs_realized(res.walk_forward, model=best, path=p))
    if res.evaluation is not None and not res.evaluation.empty:
        _try("model_scorecard",
             lambda p: P.plot_model_scorecard(res.evaluation, "qlike", path=p))
    if res.vrp is not None and not res.vrp.empty:
        _try("vrp_term", lambda p: P.plot_vrp_term(res.vrp, path=p))
    if res.vrp_history is not None and not res.vrp_history.empty:
        _try("vrp_history", lambda p: P.plot_vrp_history(res.vrp_history, path=p))
    if res.anomalies is not None:
        _try("anomalies", lambda p: P.plot_anomalies(res.anomalies, path=p))

    LOG.info("Rendered %d figures into %s", len(made), out_dir)
    return made
