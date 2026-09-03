"""
GARCH-family volatility modelling: estimation, model selection, walk-forward
out-of-sample forecasting and forecast evaluation.

Conventions used everywhere in this module
------------------------------------------
* Input `returns` are **daily log returns in decimal** (0.01 = 1%).
* `arch` is fitted on returns scaled by 100 (percent). MLE on decimal returns
  is badly conditioned — the optimiser routinely stalls with omega ~ 1e-6.
  Every forecast is unscaled before it leaves this module.
* Every volatility that leaves this module is an **annualised decimal vol**
  (0.18 = 18% annualised), so it is directly comparable to a market IV.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd

from ..config import TRADING_DAYS, GarchConfig
from ..utils import get_logger

LOG = get_logger("volsurface.garch")

# `arch` is imported lazily so that the rest of the library (pricing, surface)
# works in environments where it is not installed.
try:
    from arch import arch_model
    from arch.univariate.base import ARCHModelResult
    _HAS_ARCH = True
except ImportError:                                  # pragma: no cover
    arch_model = None                                # type: ignore[assignment]
    ARCHModelResult = Any                            # type: ignore[misc,assignment]
    _HAS_ARCH = False


# --------------------------------------------------------------------------- #
# Realised volatility targets
# --------------------------------------------------------------------------- #
def realized_vol(returns: pd.Series, window: int, annualize: bool = True,
                 forward: bool = False) -> pd.Series:
    """
    Close-to-close realised volatility over `window` days.

    Uses the zero-mean estimator sqrt(252/h * Σ r²) — the standard convention
    for volatility forecast evaluation, because subtracting an estimated mean
    adds far more noise than the drift it removes at daily frequency.

    `forward=True` returns the vol realised over the *next* `window` days,
    aligned to the decision date — this is the correct out-of-sample target.
    """
    if window < 1:
        raise ValueError("window must be >= 1")
    r2 = returns.astype(float) ** 2
    if forward:
        # Shift first so that the window covers t+1 .. t+h and never leaks
        # today's return into a "forward" target (a classic look-ahead bug).
        fwd = r2.shift(-1)
        total = fwd[::-1].rolling(window).sum()[::-1]
    else:
        total = r2.rolling(window).sum()
    var = total / window
    if annualize:
        var = var * TRADING_DAYS
    return np.sqrt(var).rename(
        f"rv_{'fwd' if forward else 'trail'}_{window}d"
    )


def parkinson_vol(high: pd.Series, low: pd.Series, window: int) -> pd.Series:
    """
    Parkinson range estimator — ~5x more efficient than close-to-close and a
    useful sanity check that a close-only RV target is not badly biased.
    """
    hl = np.log(high.astype(float) / low.astype(float)) ** 2
    var = hl.rolling(window).mean() / (4.0 * np.log(2.0))
    return np.sqrt(var * TRADING_DAYS).rename(f"parkinson_{window}d")


# --------------------------------------------------------------------------- #
# Estimation
# --------------------------------------------------------------------------- #
@dataclass
class GarchFit:
    """One estimated specification plus its diagnostics."""

    name: str
    vol_model: str
    p: int
    o: int
    q: int
    dist: str
    result: Any                    # arch ARCHModelResult
    scale: float
    loglik: float
    aic: float
    bic: float
    persistence: float
    params: dict[str, float]
    converged: bool

    @property
    def long_run_vol(self) -> float:
        """
        Unconditional (long-run) annualised vol implied by the parameters —
        the level every GARCH forecast mean-reverts to, and the single most
        informative number in the fit.

        GARCH/GJR live in variance space:      var_inf = omega / (1 - persistence)
        EGARCH lives in *log*-variance space:  var_inf = exp(omega / (1 - Σbeta))

        Getting that distinction wrong silently reports a ~0% long-run vol for
        EGARCH, which is why it is handled explicitly here.
        """
        omega = self.params.get("omega")
        uv: float | None = None
        if omega is not None and self.persistence < 1.0:
            if self.vol_model.lower() == "egarch":
                uv = float(np.exp(omega / (1.0 - self.persistence)))
            else:
                uv = float(omega / (1.0 - self.persistence))
        if uv is None or not np.isfinite(uv) or uv <= 0:
            # Non-stationary or exotic spec: fall back to the sample variance
            # of the fitted conditional volatility.
            uv = float(np.nanmean(np.asarray(self.result.conditional_volatility) ** 2))
        return float(np.sqrt(max(uv, 1e-12) * TRADING_DAYS) / self.scale)

    def summary_row(self) -> dict[str, Any]:
        return {
            "model": self.name,
            "loglik": self.loglik,
            "aic": self.aic,
            "bic": self.bic,
            "persistence": self.persistence,
            "long_run_vol": self.long_run_vol,
            "converged": self.converged,
            "n_params": len(self.params),
        }


def _requires_simulation(vol_model: str) -> bool:
    """EGARCH (and other non-affine recursions) have no analytic multi-step."""
    return vol_model.lower() in {"egarch", "aparch", "figarch"}


def fit_garch(
    returns: pd.Series,
    vol_model: str = "Garch",
    p: int = 1,
    o: int = 0,
    q: int = 1,
    dist: str = "normal",
    mean_model: str = "Constant",
    scale: float = 100.0,
    name: str | None = None,
) -> GarchFit:
    """
    Fit one GARCH-family specification by maximum likelihood.

    Raises
    ------
    ImportError  if the `arch` package is unavailable.
    ValueError   if the sample is too short for stable estimation (< 250 obs).
    RuntimeError if the optimiser fails outright.
    """
    if not _HAS_ARCH:
        raise ImportError("`arch` is required: pip install arch")

    y = pd.Series(returns).dropna().astype(float)
    if len(y) < 250:
        raise ValueError(
            f"Need >= 250 observations for a credible GARCH fit, got {len(y)}"
        )

    label = name or f"{vol_model}({p},{o},{q})-{dist}"
    am = arch_model(
        y * scale, mean=mean_model, vol=vol_model, p=p, o=o, q=q, dist=dist,
        rescale=False,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")           # arch is chatty about scaling
        try:
            res = am.fit(disp="off", show_warning=False, options={"maxiter": 2000})
        except Exception as exc:                  # optimiser blew up
            raise RuntimeError(f"MLE failed for {label}: {exc}") from exc

    params = {k: float(v) for k, v in res.params.items()}
    # Persistence: alpha + gamma/2 + beta for GARCH/GJR; beta for EGARCH.
    if vol_model.lower() == "egarch":
        persistence = sum(v for k, v in params.items() if k.startswith("beta"))
    else:
        persistence = (
            sum(v for k, v in params.items() if k.startswith("alpha"))
            + 0.5 * sum(v for k, v in params.items() if k.startswith("gamma"))
            + sum(v for k, v in params.items() if k.startswith("beta"))
        )

    converged = bool(getattr(res, "convergence_flag", 0) == 0)
    if persistence >= 1.0:
        LOG.warning(
            "%s is non-stationary (persistence=%.4f): forecasts will not "
            "mean-revert — treat long-horizon output with suspicion.",
            label, persistence,
        )

    return GarchFit(
        name=label, vol_model=vol_model, p=p, o=o, q=q, dist=dist,
        result=res, scale=scale, loglik=float(res.loglikelihood),
        aic=float(res.aic), bic=float(res.bic), persistence=float(persistence),
        params=params, converged=converged,
    )


def fit_model_suite(returns: pd.Series, cfg: GarchConfig) -> dict[str, GarchFit]:
    """
    Fit every specification in `cfg.specs`, tolerating individual failures.

    A failed spec is logged and skipped: one badly-conditioned EGARCH must not
    take down a pipeline that has three other working models.
    """
    fits: dict[str, GarchFit] = {}
    for name, vol_model, p, o, q, dist in cfg.specs:
        try:
            fits[name] = fit_garch(
                returns, vol_model=vol_model, p=p, o=o, q=q, dist=dist,
                mean_model=cfg.mean_model, scale=cfg.return_scale, name=name,
            )
            LOG.info("Fitted %-22s loglik=%10.2f  BIC=%9.2f  persistence=%.4f",
                     name, fits[name].loglik, fits[name].bic, fits[name].persistence)
        except Exception as exc:
            LOG.error("Skipping %s — %s", name, exc)
    if not fits:
        raise RuntimeError("Every GARCH specification failed to estimate.")
    return fits


def select_best(fits: dict[str, GarchFit], criterion: str = "bic") -> GarchFit:
    """Pick the preferred spec by BIC (default) or AIC, preferring convergence."""
    key = criterion.lower()
    if key not in {"bic", "aic", "loglik"}:
        raise ValueError("criterion must be one of {'bic','aic','loglik'}")
    candidates = [f for f in fits.values() if f.converged] or list(fits.values())
    sign = -1.0 if key == "loglik" else 1.0
    best = min(candidates, key=lambda f: sign * getattr(f, key))
    LOG.info("Selected %s by %s", best.name, key.upper())
    return best


def comparison_table(fits: dict[str, GarchFit]) -> pd.DataFrame:
    """Side-by-side information criteria, sorted best-first by BIC."""
    df = pd.DataFrame([f.summary_row() for f in fits.values()])
    return df.sort_values("bic").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Forecasting
# --------------------------------------------------------------------------- #
def _horizon_annual_vol(variance_path: np.ndarray, scale: float) -> float:
    """
    Convert a path of daily variance forecasts (in scaled units) into a single
    annualised decimal vol for the whole horizon.

        sigma_ann = sqrt( mean(daily var) * 252 ) / scale

    Averaging the *variance* (not the vol) is what makes this the correct
    comparison for an option with that maturity: an option prices off the
    expected integrated variance to expiry.
    """
    v = np.asarray(variance_path, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan")
    return float(np.sqrt(v.mean() * TRADING_DAYS) / scale)


def forecast_term_structure(
    fit: GarchFit,
    horizons: Sequence[int] = (1, 5, 21, 63, 126, 252),
    simulations: int = 2000,
    seed: int = 12345,
) -> pd.DataFrame:
    """
    GARCH-implied volatility **term structure** from the end of the sample.

    This is the model's own analogue of the implied-vol term structure, and it
    is what gets compared against ATM IV per expiry in the VRP analysis.
    """
    horizons = sorted({int(h) for h in horizons if int(h) >= 1})
    h_max = max(horizons)
    method = "simulation" if _requires_simulation(fit.vol_model) else "analytic"

    kwargs: dict[str, Any] = {"horizon": h_max, "reindex": False, "method": method}
    if method == "simulation":
        kwargs.update(simulations=simulations, rng=np.random.default_rng(seed).standard_normal)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fc = fit.result.forecast(**kwargs)

    path = np.asarray(fc.variance.values)[-1]              # h_max daily variances
    rows = []
    for h in horizons:
        rows.append({
            "horizon_days": h,
            "horizon_years": h / TRADING_DAYS,
            "garch_vol_ann": _horizon_annual_vol(path[:h], fit.scale),
        })
    out = pd.DataFrame(rows)
    out.attrs["model"] = fit.name
    return out


def walk_forward_forecast(
    returns: pd.Series,
    cfg: GarchConfig,
    vol_model: str = "Garch",
    p: int = 1,
    o: int = 0,
    q: int = 1,
    dist: str = "t",
    name: str | None = None,
    simulations: int = 1000,
) -> pd.DataFrame:
    """
    Honest out-of-sample walk-forward evaluation.

    At each out-of-sample date t we use **only** data up to and including t to
    produce a forecast for t+1..t+h, then compare against the volatility that
    actually realised. Parameters are re-estimated every `cfg.refit_every`
    days; between refits the previous parameters are *fixed* and the variance
    recursion is simply filtered forward on the new data (`ARCHModel.fix`).
    This is exactly how a desk runs it — nightly filtering, periodic
    recalibration — and it is ~20x cheaper than refitting daily.

    Returns
    -------
    Long-format DataFrame: date, horizon, forecast_vol, realized_vol, model.
    Rows whose realised window extends past the end of the sample are dropped.
    """
    if not _HAS_ARCH:
        raise ImportError("`arch` is required: pip install arch")

    y = pd.Series(returns).dropna().astype(float)
    n = len(y)
    horizons = sorted({int(h) for h in cfg.forecast_horizons})
    h_max = max(horizons)
    label = name or f"{vol_model}({p},{o},{q})-{dist}"

    start_idx = int(n * (1.0 - cfg.oos_fraction))
    if start_idx < 250:
        raise ValueError(
            f"In-sample window of {start_idx} obs is too short; supply more "
            f"history or reduce oos_fraction."
        )

    method = "simulation" if _requires_simulation(vol_model) else "analytic"
    rng = np.random.default_rng(2024)
    scale = cfg.return_scale

    fixed_params = None
    records: list[dict[str, Any]] = []
    n_refits = 0

    for i in range(start_idx, n):
        # ---- training slice (expanding or rolling) --------------------------
        lo = 0 if cfg.window == "expanding" else max(0, i + 1 - cfg.rolling_window_size)
        y_train = y.iloc[lo: i + 1]
        if len(y_train) < 250:
            continue

        need_refit = (fixed_params is None) or ((i - start_idx) % cfg.refit_every == 0)
        am = arch_model(y_train * scale, mean=cfg.mean_model, vol=vol_model,
                        p=p, o=o, q=q, dist=dist, rescale=False)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if need_refit:
                    res = am.fit(disp="off", show_warning=False,
                                 options={"maxiter": 1000})
                    fixed_params = res.params
                    n_refits += 1
                else:
                    # Re-filter the recursion on the extended sample without
                    # touching the parameters.
                    res = am.fix(fixed_params)

                kwargs: dict[str, Any] = {"horizon": h_max, "reindex": False,
                                          "method": method}
                if method == "simulation":
                    kwargs.update(simulations=simulations,
                                  rng=rng.standard_normal)
                fc = res.forecast(**kwargs)
                path = np.asarray(fc.variance.values)[-1]
        except Exception as exc:
            LOG.debug("Forecast failed at %s (%s) — skipping date", y.index[i], exc)
            continue

        date = y.index[i]
        for h in horizons:
            if i + h >= n:                    # realised window not yet complete
                continue
            fut = y.iloc[i + 1: i + 1 + h].to_numpy()
            rv = float(np.sqrt(np.mean(fut ** 2) * TRADING_DAYS))
            records.append({
                "date": date,
                "horizon": h,
                "forecast_vol": _horizon_annual_vol(path[:h], scale),
                "realized_vol": rv,
                "model": label,
            })

    LOG.info("Walk-forward %s: %d forecast dates, %d MLE refits, %d rows",
             label, n - start_idx, n_refits, len(records))
    out = pd.DataFrame.from_records(records)
    if not out.empty:
        out = out.sort_values(["horizon", "date"]).reset_index(drop=True)
    return out


# --------------------------------------------------------------------------- #
# Diagnostics on the fitted model
# --------------------------------------------------------------------------- #
def residual_diagnostics(fit: GarchFit, lags: int = 20) -> dict[str, float]:
    """
    Standardised-residual diagnostics.

    A correctly specified volatility model leaves standardised residuals that
    are (i) serially uncorrelated and (ii) free of remaining ARCH effects, so
    both Ljung-Box p-values should be comfortably above 0.05.
    """
    out: dict[str, float] = {}
    try:
        warnings.simplefilter("ignore", FutureWarning)
        z = np.asarray(fit.result.std_resid)
        z = z[np.isfinite(z)]
        from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch

        lb = acorr_ljungbox(z, lags=[lags], return_df=True)
        lb2 = acorr_ljungbox(z ** 2, lags=[lags], return_df=True)
        out["ljungbox_z_pvalue"] = float(lb["lb_pvalue"].iloc[0])
        out["ljungbox_z2_pvalue"] = float(lb2["lb_pvalue"].iloc[0])
        out["arch_lm_pvalue"] = float(het_arch(z, nlags=lags)[1])
        out["skew_z"] = float(pd.Series(z).skew())
        out["kurtosis_z"] = float(pd.Series(z).kurtosis())
    except Exception as exc:                             # statsmodels optional
        LOG.warning("Residual diagnostics unavailable: %s", exc)
    return out


def arch_effect_test(returns: pd.Series, lags: int = 12) -> dict[str, float]:
    """
    Engle's ARCH-LM test on raw returns — the empirical justification for
    fitting a GARCH model at all. A tiny p-value = volatility clustering.
    """
    try:
        from statsmodels.stats.diagnostic import het_arch

        r = pd.Series(returns).dropna().to_numpy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            stat, pval, fstat, fpval = het_arch(r, nlags=lags)
        return {"lm_stat": float(stat), "lm_pvalue": float(pval),
                "f_stat": float(fstat), "f_pvalue": float(fpval), "lags": lags}
    except Exception as exc:
        LOG.warning("ARCH-LM test unavailable: %s", exc)
        return {}
