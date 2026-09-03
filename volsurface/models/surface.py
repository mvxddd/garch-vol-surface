"""
Implied-volatility surface construction and querying.

The surface is stored as a set of **per-expiry smiles** plus a rule for moving
between them in maturity. That structure is not an implementation detail — it
is how a surface is quoted, risk-managed and arbitrage-checked on a desk:

* within an expiry, the smile is a calibrated curve in log-moneyness;
* across expiries, we interpolate **total variance** w = IV^2 * T linearly in T.

Interpolating total variance (rather than volatility) is what keeps the
interpolated surface calendar-arbitrage-free: if w is non-decreasing in T at
the calibrated nodes, linear interpolation between them is non-decreasing too.
Interpolating IV directly does *not* have that property and routinely creates
free calendar spreads between listed expiries.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np
import pandas as pd

from ..config import CALENDAR_DAYS, SurfaceConfig
from ..utils import get_logger
from .black_scholes import greeks
from .svi import SVIParams, check_calendar_arbitrage, fit_svi

LOG = get_logger("volsurface.surface")


class Smile(Protocol):
    """Anything that can return total variance at a log-moneyness."""

    T: float

    def total_variance(self, k) -> np.ndarray: ...


@dataclass
class SplineSlice:
    """
    Weighted smoothing-spline smile — the non-parametric fallback.

    Faster and unconditionally fittable, but with **no arbitrage guarantee**:
    use it when SVI fails to converge on a messy chain, and check the
    diagnostics before trusting the result.
    """

    T: float
    knots_k: np.ndarray
    _spline: object
    rmse_vol: float = np.nan
    n_quotes: int = 0

    def total_variance(self, k) -> np.ndarray:
        k = np.asarray(k, dtype=float)
        # Clamp to the fitted range: splines extrapolate catastrophically.
        kc = np.clip(k, self.knots_k.min(), self.knots_k.max())
        return np.maximum(np.asarray(self._spline(kc), dtype=float), 1e-10)

    def implied_vol(self, k, T: float | None = None) -> np.ndarray:
        t = float(T if T is not None else self.T)
        return np.sqrt(self.total_variance(k) / max(t, 1e-12))

    def as_dict(self) -> dict[str, float]:
        return {"T": self.T, "rmse_vol": self.rmse_vol, "n_quotes": self.n_quotes,
                "atm_vol": float(self.implied_vol(0.0)), "method": "spline"}


def _fit_spline_slice(k: np.ndarray, iv: np.ndarray, T: float,
                      weights: np.ndarray | None = None,
                      smoothing: float | None = None) -> SplineSlice:
    from scipy.interpolate import UnivariateSpline

    order = np.argsort(k)
    k_s, iv_s = np.asarray(k)[order], np.asarray(iv)[order]
    w_obs = iv_s ** 2 * T
    wts = (np.ones_like(w_obs) if weights is None
           else np.sqrt(np.clip(np.asarray(weights)[order], 1e-6, None)))
    # Collapse duplicate strikes — UnivariateSpline requires strictly increasing x.
    k_u, idx = np.unique(k_s, return_index=True)
    w_u, wt_u = w_obs[idx], wts[idx]
    deg = 3 if len(k_u) >= 6 else max(len(k_u) - 1, 1)
    s = smoothing if smoothing is not None else len(k_u) * float(np.var(w_u)) * 1e-3
    spl = UnivariateSpline(k_u, w_u, w=wt_u, k=deg, s=s, ext="const")
    fitted = np.sqrt(np.maximum(spl(k_u), 1e-12) / T)
    rmse = float(np.sqrt(np.mean((fitted - np.sqrt(w_u / T)) ** 2)))
    return SplineSlice(T=float(T), knots_k=k_u, _spline=spl, rmse_vol=rmse,
                       n_quotes=int(len(k_s)))


# --------------------------------------------------------------------------- #
# The surface
# --------------------------------------------------------------------------- #
class VolSurface:
    """
    A queryable implied-volatility surface.

    Query API (all vectorised, all in annualised decimal vol):
        surface.iv(k, T)                 by log-moneyness and year fraction
        surface.iv_strike(K, T)          by absolute strike
        surface.atm_vol(T)               the k = 0 point
        surface.grid(...)                a (k x T) mesh for plotting
        surface.strike_for_delta(...)    invert delta -> strike (for RR/BF)
    """

    def __init__(self, slices: Sequence[Smile], forwards: pd.DataFrame,
                 spot: float, asof: pd.Timestamp, method: str = "svi",
                 quotes: pd.DataFrame | None = None,
                 risk_free_rate: float = 0.0):
        self.slices = sorted(slices, key=lambda s: s.T)
        if not self.slices:
            raise ValueError("A surface needs at least one calibrated smile.")
        self.forwards = forwards.sort_values("T").reset_index(drop=True)
        self.spot = float(spot)
        self.asof = pd.Timestamp(asof)
        self.method = method
        self.quotes = quotes
        self.r = float(risk_free_rate)
        self._T = np.array([s.T for s in self.slices], dtype=float)

    # -- maturity handling -------------------------------------------------- #
    @property
    def maturities(self) -> np.ndarray:
        return self._T.copy()

    def forward(self, T) -> np.ndarray:
        """
        Forward at an arbitrary maturity: linear in **log F** against T, which
        is equivalent to a piecewise-constant carry rate between listed
        expiries — the standard curve-building convention.
        """
        T = np.asarray(T, dtype=float)
        t_nodes = self.forwards["T"].to_numpy()
        log_f = np.log(self.forwards["forward"].to_numpy())
        if t_nodes.size == 1:
            return np.full_like(T, float(np.exp(log_f[0])))
        return np.exp(np.interp(T, t_nodes, log_f))

    def total_variance(self, k, T) -> np.ndarray:
        """
        Total variance at (k, T), interpolated linearly in T between slices.

        Outside the calibrated range we extrapolate conservatively:
        * short end — hold volatility constant (w scales linearly with T), so
          we never invent a variance level for a tenor no one quoted;
        * long end — hold the last *forward* variance constant, the flat-
          forward convention, with a floor at zero so w stays non-decreasing.
        """
        k_arr = np.asarray(k, dtype=float)
        T_arr = np.asarray(T, dtype=float)
        k_b, T_b = np.broadcast_arrays(k_arr, T_arr)
        flat_k, flat_T = k_b.ravel(), T_b.ravel()

        # w_i(k) for every slice, evaluated at every requested k: (n_slices, n_pts)
        w_nodes = np.vstack([np.asarray(s.total_variance(flat_k), dtype=float)
                             for s in self.slices])
        t_nodes = self._T
        cols = np.arange(flat_k.size)

        if t_nodes.size == 1:
            out = w_nodes[0] * (flat_T / max(t_nodes[0], 1e-12))
            return np.maximum(out.reshape(k_b.shape), 1e-10)

        # Interior: linear in T between the two bracketing slices (vectorised —
        # a per-point np.interp loop is 50x slower on a plotting-sized mesh).
        idx = np.clip(np.searchsorted(t_nodes, flat_T, side="right") - 1,
                      0, t_nodes.size - 2)
        t_lo, t_hi = t_nodes[idx], t_nodes[idx + 1]
        frac = (flat_T - t_lo) / np.maximum(t_hi - t_lo, 1e-12)
        w_lo, w_hi = w_nodes[idx, cols], w_nodes[idx + 1, cols]
        out = w_lo + frac * (w_hi - w_lo)

        # Short end: hold volatility constant, so w scales linearly with T.
        short = flat_T <= t_nodes[0]
        if short.any():
            out[short] = w_nodes[0, cols[short]] * (flat_T[short] / max(t_nodes[0], 1e-12))

        # Long end: flat-forward variance, floored at zero so w never decreases.
        long = flat_T >= t_nodes[-1]
        if long.any():
            slope = np.maximum(
                (w_nodes[-1, cols[long]] - w_nodes[-2, cols[long]])
                / max(t_nodes[-1] - t_nodes[-2], 1e-12), 0.0)
            out[long] = w_nodes[-1, cols[long]] + slope * (flat_T[long] - t_nodes[-1])

        return np.maximum(out.reshape(k_b.shape), 1e-10)

    # -- primary queries ---------------------------------------------------- #
    def iv(self, k, T) -> np.ndarray:
        """Implied volatility at log-moneyness `k` and year fraction `T`."""
        T_arr = np.asarray(T, dtype=float)
        w = self.total_variance(k, T)
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.sqrt(w / np.where(T_arr > 0, T_arr, np.nan))
        return out

    def iv_strike(self, K, T) -> np.ndarray:
        """Implied volatility at an absolute strike."""
        K = np.asarray(K, dtype=float)
        F = self.forward(T)
        return self.iv(np.log(K / F), T)

    def atm_vol(self, T) -> np.ndarray:
        """Forward-ATM volatility (k = 0) — the level of the surface."""
        return self.iv(0.0, T)

    def atm_term_structure(self, days: Sequence[int] | None = None) -> pd.DataFrame:
        """ATM vol by tenor: the term structure, on listed nodes by default."""
        if days is None:
            t = self._T.copy()
        else:
            t = np.asarray(days, dtype=float) / CALENDAR_DAYS
        return pd.DataFrame({
            "days": np.round(t * CALENDAR_DAYS).astype(int),
            "T": t,
            "forward": self.forward(t),
            "atm_iv": self.atm_vol(t),
        })

    def grid(self, k_bounds: tuple[float, float] = (-0.35, 0.25),
             n_k: int = 81, n_t: int = 41,
             t_bounds: tuple[float, float] | None = None
             ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Dense (k, T) mesh of implied vols for 3-D plotting.

        Returns (k_grid, T_grid, IV) with IV shaped (n_t, n_k).
        """
        t_lo, t_hi = t_bounds or (float(self._T.min()), float(self._T.max()))
        k_grid = np.linspace(*k_bounds, n_k)
        t_grid = np.linspace(t_lo, t_hi, n_t)
        KK, TT = np.meshgrid(k_grid, t_grid)
        return k_grid, t_grid, self.iv(KK, TT)

    # -- delta-space queries (for skew metrics) ----------------------------- #
    def strike_for_delta(self, delta: float, T: float, is_call: bool,
                         max_iter: int = 50, tol: float = 1e-8) -> float:
        """
        Strike whose Black-76 spot delta equals `delta`, solved consistently
        with the surface (the vol at the answer must be the vol used to find
        it). Fixed-point iteration converges in a handful of steps because the
        smile is far flatter in k than the delta map is.
        """
        from scipy.stats import norm

        if not (0.0 < delta < 1.0) or T <= 0:
            return float("nan")
        F = float(self.forward(T))
        sigma = float(self.atm_vol(T))
        k = 0.0
        for _ in range(max_iter):
            sigma = float(self.iv(k, T))
            if not np.isfinite(sigma) or sigma <= 0:
                return float("nan")
            d = delta if is_call else 1.0 - delta
            k_new = -sigma * np.sqrt(T) * norm.ppf(d) + 0.5 * sigma ** 2 * T
            if abs(k_new - k) < tol:
                k = k_new
                break
            k = k_new
        return float(F * np.exp(k))

    def iv_for_delta(self, delta: float, T: float, is_call: bool) -> float:
        K = self.strike_for_delta(delta, T, is_call)
        if not np.isfinite(K):
            return float("nan")
        return float(self.iv_strike(K, T))

    # -- diagnostics -------------------------------------------------------- #
    def slice_table(self) -> pd.DataFrame:
        """Per-expiry calibration diagnostics (params, fit error, arb checks)."""
        rows = []
        for s in self.slices:
            row = s.as_dict() if hasattr(s, "as_dict") else {"T": s.T}
            row["days"] = int(round(s.T * CALENDAR_DAYS))
            row["forward"] = float(self.forward(s.T))
            rows.append(row)
        cols_first = ["days", "T", "forward", "atm_vol", "n_quotes", "rmse_vol"]
        df = pd.DataFrame(rows)
        ordered = [c for c in cols_first if c in df.columns]
        return df[ordered + [c for c in df.columns if c not in ordered]]

    def calendar_arbitrage(self) -> dict[str, object]:
        """Calendar-spread check across calibrated slices (SVI slices only)."""
        svi = [s for s in self.slices if isinstance(s, SVIParams)]
        if len(svi) >= 2:
            return check_calendar_arbitrage(svi)
        # Generic numeric check for non-SVI slices.
        grid = np.linspace(-0.3, 0.2, 61)
        violations, worst = [], 0.0
        for near, far in zip(self.slices[:-1], self.slices[1:]):
            diff = far.total_variance(grid) - near.total_variance(grid)
            if (diff < -1e-6).any():
                depth = float(-diff.min())
                worst = max(worst, depth)
                violations.append({"T_near": near.T, "T_far": far.T,
                                   "max_depth_total_var": depth,
                                   "n_k_violating": int((diff < -1e-6).sum())})
        return {"has_arbitrage": bool(violations), "violations": violations,
                "max_violation": worst}

    def fit_quality(self) -> pd.DataFrame:
        """Residuals of the calibrated surface against the quotes it was fit to."""
        if self.quotes is None or self.quotes.empty:
            return pd.DataFrame()
        q = self.quotes.copy()
        q["iv_model"] = self.iv(q["k"].to_numpy(), q["T"].to_numpy())
        q["resid_vol"] = q["iv_model"] - q["iv"]
        # Is the model inside the market's own bid/ask, in vol terms?
        if {"iv_bid", "iv_ask"}.issubset(q.columns):
            q["inside_spread"] = (
                (q["iv_model"] >= q[["iv_bid", "iv_ask"]].min(axis=1))
                & (q["iv_model"] <= q[["iv_bid", "iv_ask"]].max(axis=1))
            )
        agg = {"resid_vol": ["mean", "std", lambda s: float(np.sqrt(np.mean(s ** 2)))],
               "iv": "size"}
        out = q.groupby(["expiry"]).agg(agg)
        out.columns = ["bias", "std", "rmse", "n"]
        if "inside_spread" in q.columns:
            out["pct_inside_spread"] = (q.groupby("expiry")["inside_spread"]
                                        .mean().mul(100).round(1))
        return out.reset_index()


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def build_surface(quotes: pd.DataFrame, forwards: pd.DataFrame, spot: float,
                  cfg: SurfaceConfig, asof: pd.Timestamp | None = None,
                  risk_free_rate: float = 0.0) -> VolSurface:
    """
    Calibrate one smile per expiry and assemble them into a `VolSurface`.

    If SVI fails on an expiry (too few quotes, pathological data), that expiry
    falls back to a smoothing spline rather than taking down the whole build —
    with a loud warning, because a spline slice carries no arbitrage guarantee.
    """
    required = {"expiry", "T", "k", "iv"}
    missing = required - set(quotes.columns)
    if missing:
        raise ValueError(f"quotes is missing columns: {sorted(missing)}")

    asof = pd.Timestamp(asof or quotes["asof"].iloc[0])
    method = cfg.method.lower()
    slices: list[Smile] = []

    for (expiry, T), grp in quotes.groupby(["expiry", "T"], sort=True):
        k = grp["k"].to_numpy()
        iv = grp["iv"].to_numpy()
        w = grp["vega"].to_numpy() if (cfg.vega_weighting and "vega" in grp) else None
        try:
            if method == "svi":
                slices.append(fit_svi(k, iv, float(T), weights=w,
                                      n_starts=cfg.svi_n_starts,
                                      max_iter=cfg.svi_max_iter))
            else:
                slices.append(_fit_spline_slice(k, iv, float(T), weights=w))
        except Exception as exc:
            LOG.error("Smile calibration failed for %s (T=%.3f): %s — falling "
                      "back to a spline slice (no arbitrage guarantee).",
                      pd.Timestamp(expiry).date(), T, exc)
            try:
                slices.append(_fit_spline_slice(k, iv, float(T), weights=w))
            except Exception as exc2:
                LOG.error("Spline fallback also failed for %s: %s — expiry dropped.",
                          pd.Timestamp(expiry).date(), exc2)

    if not slices:
        raise RuntimeError("No expiry could be calibrated; cannot build a surface.")

    surface = VolSurface(slices, forwards=forwards, spot=spot, asof=asof,
                         method=method, quotes=quotes, risk_free_rate=risk_free_rate)

    cal = surface.calendar_arbitrage()
    n_bad_bfly = sum(1 for s in slices
                     if isinstance(s, SVIParams) and not s.butterfly_free)
    LOG.info("Surface built: %d slices (%s), calendar arb=%s, "
             "%d slice(s) with residual butterfly arb",
             len(slices), method, cal["has_arbitrage"], n_bad_bfly)
    return surface
