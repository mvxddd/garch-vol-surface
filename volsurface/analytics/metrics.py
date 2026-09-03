"""
Volatility-forecast evaluation.

Why not just RMSE? Because realised variance is a noisy, right-skewed proxy for
the latent volatility. RMSE on vol over-weights the few explosive days and can
rank a genuinely better model last. Desks therefore look at a panel:

* **RMSE / MAE** on volatility — interpretable in vol points.
* **QLIKE** on variance — the standard robust loss for volatility forecasting
  (Patton 2011): it is robust to noise in the RV proxy, penalises
  under-prediction harder than over-prediction, and is what practitioners rank
  on. Lower is better.
* **Mincer-Zarnowitz** regression RV = a + b·F: an unbiased, efficient
  forecast has a = 0, b = 1. b < 1 is the classic "GARCH over-reacts" finding.
* **Diebold-Mariano** for whether one model beats another *significantly*.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..utils import get_logger

LOG = get_logger("volsurface.metrics")


def _clean_pair(forecast, realized) -> tuple[np.ndarray, np.ndarray]:
    f = np.asarray(forecast, dtype=float)
    a = np.asarray(realized, dtype=float)
    if f.shape != a.shape:
        raise ValueError(f"shape mismatch: forecast {f.shape} vs realized {a.shape}")
    ok = np.isfinite(f) & np.isfinite(a) & (f > 0) & (a >= 0)
    return f[ok], a[ok]


def qlike(forecast_vol, realized_vol) -> float:
    """
    QLIKE loss on the variance scale:  mean[ log(σ²_f) + RV² / σ²_f ].

    Robust to the fact that RV is only a proxy for the true latent variance.
    """
    f, a = _clean_pair(forecast_vol, realized_vol)
    if f.size == 0:
        return float("nan")
    vf, va = f ** 2, a ** 2
    return float(np.mean(np.log(vf) + va / vf))


def mincer_zarnowitz(forecast_vol, realized_vol) -> dict[str, float]:
    """
    OLS of realised on forecast volatility. Returns alpha, beta, R² and the
    t-statistic for H0: beta = 1 (the efficiency test that actually matters).
    """
    f, a = _clean_pair(forecast_vol, realized_vol)
    n = f.size
    if n < 10:
        return {"alpha": np.nan, "beta": np.nan, "r_squared": np.nan,
                "t_beta_eq_1": np.nan, "n": n}

    X = np.column_stack([np.ones(n), f])
    coef, *_ = np.linalg.lstsq(X, a, rcond=None)
    resid = a - X @ coef
    dof = max(n - 2, 1)
    sigma2 = float(resid @ resid) / dof
    xtx_inv = np.linalg.pinv(X.T @ X)
    se_beta = float(np.sqrt(sigma2 * xtx_inv[1, 1]))
    ss_tot = float(np.sum((a - a.mean()) ** 2))
    r2 = 1.0 - float(resid @ resid) / ss_tot if ss_tot > 0 else np.nan

    return {
        "alpha": float(coef[0]),
        "beta": float(coef[1]),
        "r_squared": r2,
        "t_beta_eq_1": (coef[1] - 1.0) / se_beta if se_beta > 0 else np.nan,
        "n": n,
    }


def forecast_metrics(forecast_vol, realized_vol) -> dict[str, float]:
    """Full evaluation panel for one (model, horizon) pair."""
    f, a = _clean_pair(forecast_vol, realized_vol)
    if f.size == 0:
        return {k: np.nan for k in
                ("rmse", "mae", "mape", "bias", "qlike", "r2", "beta",
                 "alpha", "t_beta_eq_1", "corr", "n")}

    err = f - a
    mz = mincer_zarnowitz(f, a)
    with np.errstate(divide="ignore", invalid="ignore"):
        mape = float(np.mean(np.abs(err) / np.where(a > 0, a, np.nan)) * 100)

    return {
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(np.abs(err))),
        "mape": mape,
        "bias": float(np.mean(err)),                 # >0 => forecasts too high
        "qlike": qlike(f, a),
        "r2": mz["r_squared"],
        "alpha": mz["alpha"],
        "beta": mz["beta"],
        "t_beta_eq_1": mz["t_beta_eq_1"],
        "corr": float(np.corrcoef(f, a)[0, 1]) if f.size > 2 else np.nan,
        "n": int(f.size),
    }


def diebold_mariano(
    loss_a: np.ndarray, loss_b: np.ndarray, horizon: int = 1,
    harvey_correction: bool = True,
) -> dict[str, float]:
    """
    Diebold-Mariano test of equal predictive accuracy (A vs B).

    Negative DM statistic => model A has lower loss (A is better).
    Overlapping h-step forecasts are autocorrelated, so the long-run variance
    uses a Newey-West kernel truncated at h-1, plus the Harvey-Leybourne-Newbold
    small-sample correction (a t-distribution, not a normal, in finite samples).
    """
    from scipy import stats

    d = np.asarray(loss_a, dtype=float) - np.asarray(loss_b, dtype=float)
    d = d[np.isfinite(d)]
    n = d.size
    if n < 10:
        return {"dm_stat": np.nan, "p_value": np.nan, "mean_diff": np.nan, "n": n}

    d_bar = float(d.mean())
    dm_lag = max(int(horizon) - 1, 0)
    gamma0 = float(np.mean((d - d_bar) ** 2))
    lrv = gamma0
    for lag in range(1, dm_lag + 1):
        cov = float(np.mean((d[lag:] - d_bar) * (d[:-lag] - d_bar)))
        lrv += 2.0 * (1.0 - lag / (dm_lag + 1.0)) * cov          # Bartlett weight
    lrv = max(lrv, 1e-16)

    dm = d_bar / np.sqrt(lrv / n)
    if harvey_correction and n > dm_lag:
        h = dm_lag + 1
        adj = np.sqrt(max((n + 1 - 2 * h + h * (h - 1) / n) / n, 1e-12))
        dm *= adj
    p = float(2 * (1 - stats.t.cdf(abs(dm), df=n - 1)))
    return {"dm_stat": float(dm), "p_value": p, "mean_diff": d_bar, "n": n}


def compare_models_dm(
    walk_forward: pd.DataFrame, loss: str = "qlike",
) -> pd.DataFrame:
    """
    Pairwise DM tests between every model in a walk-forward frame, per horizon.

    `walk_forward` must have columns
    [date, horizon, forecast_vol, realized_vol, model].
    """
    required = {"date", "horizon", "forecast_vol", "realized_vol", "model"}
    missing = required - set(walk_forward.columns)
    if missing:
        raise ValueError(f"walk_forward is missing columns: {sorted(missing)}")

    def _loss(f: np.ndarray, a: np.ndarray) -> np.ndarray:
        if loss == "qlike":
            vf, va = f ** 2, a ** 2
            return np.log(vf) + va / vf
        if loss == "mse":
            return (f - a) ** 2
        raise ValueError("loss must be 'qlike' or 'mse'")

    rows = []
    for horizon, grp in walk_forward.groupby("horizon"):
        wide = grp.pivot_table(index="date", columns="model",
                               values=["forecast_vol", "realized_vol"])
        models = list(wide["forecast_vol"].columns)
        realized = wide["realized_vol"].mean(axis=1).to_numpy()
        losses = {
            m: _loss(wide["forecast_vol"][m].to_numpy(), realized) for m in models
        }
        for i, a in enumerate(models):
            for b in models[i + 1:]:
                mask = np.isfinite(losses[a]) & np.isfinite(losses[b])
                res = diebold_mariano(losses[a][mask], losses[b][mask], horizon=horizon)
                rows.append({
                    "horizon": horizon, "model_a": a, "model_b": b,
                    "loss": loss, **res,
                    "winner": (a if res["mean_diff"] < 0 else b)
                    if np.isfinite(res["mean_diff"]) else None,
                    "significant_5pct": bool(res["p_value"] < 0.05)
                    if np.isfinite(res["p_value"]) else False,
                })
    return pd.DataFrame(rows)


def evaluate_walk_forward(walk_forward: pd.DataFrame) -> pd.DataFrame:
    """Metric panel for every (model, horizon) in a walk-forward frame."""
    if walk_forward.empty:
        return pd.DataFrame()
    rows = []
    for (model, horizon), grp in walk_forward.groupby(["model", "horizon"]):
        rows.append({
            "model": model, "horizon": horizon,
            **forecast_metrics(grp["forecast_vol"], grp["realized_vol"]),
        })
    out = pd.DataFrame(rows).sort_values(["horizon", "qlike"])
    return out.reset_index(drop=True)


def naive_benchmarks(returns: pd.Series, horizons, oos_index,
                     rw_window: int = 21, min_vol: float = 0.03) -> pd.DataFrame:
    """
    Benchmark forecasts every GARCH model must beat to be worth anything:

    * **Historical RV** — trailing realised vol over a fixed `rw_window`
      (21 days by default). Note the window is fixed rather than matched to the
      forecast horizon: a *one-day* trailing realised vol is not a volatility
      forecast anyone would use, and scoring it as one produces a straw man
      that flatters every other model.
    * **EWMA** — RiskMetrics lambda = 0.94

    A GARCH that cannot beat EWMA out-of-sample has told you something
    important, and reporting it is what separates a real study from a demo.

    `min_vol` floors both benchmarks at 3% annualised: a near-zero forecast
    sends QLIKE — which contains RV/sigma^2 — to a meaningless number that
    swamps the rest of the comparison. The floor is a property of the
    *benchmark*, not of the loss; no desk would quote a 0.1% vol forecast.
    """
    r = pd.Series(returns).dropna().astype(float)
    lam = 0.94
    ewma_var = r.pow(2).ewm(alpha=1 - lam, adjust=False).mean()
    from ..config import TRADING_DAYS

    rows = []
    idx = pd.Index(oos_index)
    pos = {d: i for i, d in enumerate(r.index)}
    arr = r.to_numpy()
    n = len(arr)
    for h in horizons:
        trail = np.sqrt(r.pow(2).rolling(h).mean() * TRADING_DAYS)
        for d in idx:
            i = pos.get(d)
            if i is None or i + h >= n:
                continue
            rv = float(np.sqrt(np.mean(arr[i + 1: i + 1 + h] ** 2) * TRADING_DAYS))
            for name, raw in (
                (f"Historical RV ({rw_window}d)",
                 float(trail.iloc[i]) if i < len(trail) else np.nan),
                ("EWMA(0.94)", float(np.sqrt(ewma_var.iloc[i] * TRADING_DAYS))),
            ):
                val = max(raw, min_vol) if np.isfinite(raw) else raw
                if np.isfinite(val):
                    rows.append({"date": d, "horizon": h, "forecast_vol": val,
                                 "realized_vol": rv, "model": name})
    return pd.DataFrame(rows)
