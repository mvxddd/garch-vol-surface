"""
Raw-SVI smile calibration with no-arbitrage constraints (Gatheral, 2004).

Why parametric SVI instead of just splining the quotes
------------------------------------------------------
A spline through mid-IVs will happily produce a smile whose implied risk-
neutral density is negative — i.e. a surface that offers free money on a
butterfly. That surface cannot be used to price or risk anything. Raw SVI

    w(k) = a + b [ rho (k - m) + sqrt((k - m)^2 + sigma^2) ]      w = IV^2 * T

is the market standard because (i) five parameters reproduce essentially every
observed equity smile, (ii) its wings satisfy Lee's moment formula by
construction, and (iii) there are explicit conditions on (a,b,rho,m,sigma) that
guarantee a non-negative density. This module calibrates it under those
conditions and reports the diagnostics that prove it.

Parameter roles (useful when reading a calibration table):
    a     — overall variance level
    b     — angle between the wings (total smile slope)
    rho   — skew / rotation: negative = the equity crash skew
    m     — horizontal shift of the smile's minimum
    sigma — how rounded the vertex is (ATM curvature)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..utils import get_logger

LOG = get_logger("volsurface.svi")

_MIN_W = 1e-8


@dataclass(frozen=True)
class SVIParams:
    """One calibrated smile, plus the diagnostics needed to trust it."""

    a: float
    b: float
    rho: float
    m: float
    sigma: float
    T: float
    rmse_vol: float = np.nan       # fit error in vol points
    n_quotes: int = 0
    butterfly_free: bool = True
    min_durrleman_g: float = np.nan

    # -- the smile itself --------------------------------------------------- #
    def total_variance(self, k) -> np.ndarray:
        k = np.asarray(k, dtype=float)
        km = k - self.m
        return np.maximum(
            self.a + self.b * (self.rho * km + np.sqrt(km ** 2 + self.sigma ** 2)),
            _MIN_W,
        )

    def implied_vol(self, k, T: float | None = None) -> np.ndarray:
        """Implied vol at log-moneyness `k` (annualised, decimal)."""
        t = float(T if T is not None else self.T)
        if t <= 0:
            return np.full_like(np.asarray(k, dtype=float), np.nan)
        return np.sqrt(self.total_variance(k) / t)

    # -- derivatives (closed form; needed for the arbitrage tests) ---------- #
    def dw_dk(self, k) -> np.ndarray:
        km = np.asarray(k, dtype=float) - self.m
        root = np.sqrt(km ** 2 + self.sigma ** 2)
        return self.b * (self.rho + km / np.maximum(root, _MIN_W))

    def d2w_dk2(self, k) -> np.ndarray:
        km = np.asarray(k, dtype=float) - self.m
        return self.b * self.sigma ** 2 / np.maximum(
            (km ** 2 + self.sigma ** 2) ** 1.5, _MIN_W
        )

    # -- arbitrage diagnostics --------------------------------------------- #
    def durrleman_g(self, k) -> np.ndarray:
        """
        Durrleman's function. g(k) >= 0 for every k  <=>  no butterfly
        arbitrage <=> the implied risk-neutral density is non-negative.

            g = (1 - k w'/(2w))^2 - (w'^2/4)(1/w + 1/4) + w''/2
        """
        k = np.asarray(k, dtype=float)
        w = self.total_variance(k)
        wp, wpp = self.dw_dk(k), self.d2w_dk2(k)
        term1 = (1.0 - k * wp / (2.0 * w)) ** 2
        term2 = (wp ** 2 / 4.0) * (1.0 / w + 0.25)
        return term1 - term2 + wpp / 2.0

    def risk_neutral_density(self, k) -> np.ndarray:
        """
        Implied density in log-strike space:

            p(k) = g(k) / sqrt(2 pi w(k)) * exp(-d2(k)^2 / 2)

        Plotting this is the fastest visual test of a surface's sanity — any
        dip below zero is an arbitrage the calibration failed to exclude.
        """
        k = np.asarray(k, dtype=float)
        w = self.total_variance(k)
        d2 = -k / np.sqrt(w) - 0.5 * np.sqrt(w)
        return self.durrleman_g(k) / np.sqrt(2 * np.pi * w) * np.exp(-0.5 * d2 ** 2)

    def as_dict(self) -> dict[str, float]:
        return {"a": self.a, "b": self.b, "rho": self.rho, "m": self.m,
                "sigma": self.sigma, "T": self.T, "rmse_vol": self.rmse_vol,
                "n_quotes": self.n_quotes, "butterfly_free": self.butterfly_free,
                "min_durrleman_g": self.min_durrleman_g,
                "atm_vol": float(self.implied_vol(0.0)),
                "atm_skew": float(self.dw_dk(0.0) / (2 * np.sqrt(
                    max(self.total_variance(0.0), _MIN_W) * self.T))
                    if self.T > 0 else np.nan)}


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #
def _svi_w(theta, k):
    """Vectorised raw-SVI total variance for a parameter vector."""
    a, b, rho, m, sigma = theta
    km = np.asarray(k, dtype=float) - m
    return a + b * (rho * km + np.sqrt(km ** 2 + sigma ** 2))


def _durrleman_grid(theta, k):
    """Durrleman g(k) evaluated directly from a parameter vector (no object)."""
    a, b, rho, m, sigma = theta
    km = np.asarray(k, dtype=float) - m
    root = np.sqrt(km ** 2 + sigma ** 2)
    w = np.maximum(a + b * (rho * km + root), _MIN_W)
    wp = b * (rho + km / np.maximum(root, _MIN_W))
    wpp = b * sigma ** 2 / np.maximum(root ** 3, _MIN_W)
    return (1 - k * wp / (2 * w)) ** 2 - (wp ** 2 / 4) * (1 / w + 0.25) + wpp / 2


def _residuals(theta, k, w_obs, weights, T, k_dense, penalty):
    """
    Weighted fit residuals with hard-constraint penalties appended.

    The penalties are part of the residual *vector* (not a separate objective)
    so `least_squares` can exploit the Jacobian structure. They enforce:

      1. minimum total variance  a + b*sigma*sqrt(1-rho^2) >= 0   (positive vol)
      2. Lee's wing bound        b (1 + |rho|) <= 4/T              (no wing arb)
      3. Durrleman               g(k) >= 0 on a dense grid         (no butterfly)

    `penalty` is escalated by the caller until the constraints actually bind —
    a fixed weight either distorts a clean fit or fails to bite on a messy one.
    """
    a, b, rho, m, sigma = theta
    res = weights * (_svi_w(theta, k) - w_obs)

    min_var = a + b * sigma * np.sqrt(max(1.0 - rho ** 2, 0.0))
    pen_var = penalty * max(0.0, -min_var)
    pen_wing = penalty * max(0.0, b * (1.0 + abs(rho)) - 4.0 / max(T, 1e-6))
    pen_bfly = penalty * np.maximum(0.0, -_durrleman_grid(theta, k_dense))

    return np.concatenate([res, [pen_var, pen_wing], pen_bfly])


def fit_svi(
    k: np.ndarray,
    iv: np.ndarray,
    T: float,
    weights: np.ndarray | None = None,
    n_starts: int = 8,
    max_iter: int = 4000,
    extrapolation_margin: float = 0.10,
    outlier_mad: float = 4.0,
) -> SVIParams:
    """
    Calibrate raw SVI to one expiry.

    Parameters
    ----------
    k       : log-moneyness log(K/F) of each quote
    iv      : implied volatility of each quote (decimal, annualised)
    T       : year fraction to expiry
    weights : per-quote weights — pass **vega** so the fit respects the quotes
              a market maker can actually trade, not the noisy 5-delta wings.
    extrapolation_margin :
              how far beyond the quoted strike range the no-arbitrage
              constraints are enforced (and later checked). Enforcing them out
              to infinity would over-constrain the fit; enforcing them only on
              the quotes would let the interpolated surface go negative between
              them. A 0.10 log-moneyness buffer is the practical compromise.

    Method
    ------
    1. Multi-start bounded least squares over the economically meaningful
       region of (rho, sigma) — the objective is non-convex.
    2. One IRLS-style re-weighting pass that down-weights quotes more than
       `outlier_mad` MADs from the first fit. This is the robustness that a
       soft-L1 loss would give, *without* the side effect of also squashing the
       constraint penalties (which is what made an earlier version of this
       function silently return arbitrageable smiles).
    3. Penalty continuation: escalate the constraint weight until Durrleman's
       condition holds, keeping the loosest penalty that produces a clean fit.

    Raises
    ------
    ValueError   if fewer than 5 usable quotes or T <= 0.
    RuntimeError if every start fails.
    """
    from scipy.optimize import least_squares

    k = np.asarray(k, dtype=float)
    iv = np.asarray(iv, dtype=float)
    ok = np.isfinite(k) & np.isfinite(iv) & (iv > 0)
    k, iv = k[ok], iv[ok]
    if k.size < 5:
        raise ValueError(f"Need >= 5 quotes to calibrate SVI, got {k.size}")
    if T <= 0:
        raise ValueError("T must be positive")

    w_obs = (iv ** 2) * T
    if weights is None:
        wts = np.ones_like(w_obs)
    else:
        wts = np.asarray(weights, dtype=float)[ok]
        wts = np.where(np.isfinite(wts) & (wts > 0), wts, 0.0)
        wts = np.sqrt(np.maximum(wts / max(wts.max(), 1e-12), 1e-3))

    margin = float(extrapolation_margin)
    k_dense = np.linspace(k.min() - margin, k.max() + margin, 80)

    lo = np.array([-2.0 * w_obs.max() - 1e-4, 1e-6, -0.999, 2 * k.min() - 1.0, 1e-4])
    hi = np.array([2.0 * w_obs.max() + 1e-2, 4.0 / max(T, 1e-6), 0.999,
                   2 * k.max() + 1.0, 5.0])

    m0 = float(k[np.argmin(w_obs)])
    w_min = float(w_obs.min())
    w_atm = float(np.interp(0.0, k, w_obs))
    starts = [(rho0, sig0)
              for rho0 in (-0.85, -0.6, -0.3, 0.0, 0.4)
              for sig0 in (0.05, 0.2, 0.6)][:max(int(n_starts), 1)]

    def _solve(cur_weights, penalty, x_init=None):
        best_sol, best_cost = None, np.inf
        seeds = [x_init] if x_init is not None else []
        seeds += [np.array([w_min * 0.7, min(max(w_atm, 1e-4) * 2.0, hi[1] * 0.5),
                            r0, m0, s0]) for r0, s0 in starts]
        for x0 in seeds:
            x0 = np.clip(np.asarray(x0, dtype=float), lo + 1e-9, hi - 1e-9)
            try:
                sol = least_squares(
                    _residuals, x0, bounds=(lo, hi), max_nfev=max_iter,
                    args=(k, w_obs, cur_weights, T, k_dense, penalty),
                )
            except Exception as exc:                       # pragma: no cover
                LOG.debug("SVI start failed at T=%.4f: %s", T, exc)
                continue
            if sol.cost < best_cost:
                best_sol, best_cost = sol, sol.cost
        return best_sol

    # --- stage 1: initial fit -------------------------------------------- #
    sol = _solve(wts, penalty=50.0)
    if sol is None:
        raise RuntimeError(f"SVI calibration failed for T={T:.4f}")

    # --- stage 2: one robustness pass ------------------------------------ #
    resid = _svi_w(sol.x, k) - w_obs
    mad = float(np.median(np.abs(resid - np.median(resid))))
    if mad > 0:
        keep = np.abs(resid - np.median(resid)) <= outlier_mad * mad
        if keep.sum() >= 5 and (~keep).any():
            LOG.info("T=%.3f: down-weighting %d outlier quote(s)", T, int((~keep).sum()))
            robust_w = np.where(keep, wts, wts * 0.05)
            sol = _solve(robust_w, penalty=50.0, x_init=sol.x) or sol

    # --- stage 3: penalty continuation until arbitrage-free -------------- #
    k_check = np.linspace(k.min() - margin, k.max() + margin, 400)
    for penalty in (50.0, 500.0, 5_000.0, 50_000.0):
        cand = _solve(wts, penalty=penalty, x_init=sol.x)
        if cand is None:
            continue
        sol = cand
        if float(np.min(_durrleman_grid(sol.x, k_check))) >= -1e-10:
            break

    a, b, rho, m, sigma = (float(v) for v in sol.x)
    fitted = np.sqrt(np.maximum(_svi_w(sol.x, k), _MIN_W) / T)
    rmse = float(np.sqrt(np.mean((fitted - iv) ** 2)))
    min_g = float(np.min(_durrleman_grid(sol.x, k_check)))

    out = SVIParams(a=a, b=b, rho=rho, m=m, sigma=sigma, T=T, rmse_vol=rmse,
                    n_quotes=int(k.size), butterfly_free=bool(min_g >= -1e-10),
                    min_durrleman_g=min_g)
    if not out.butterfly_free:
        LOG.warning("T=%.3f: residual butterfly arbitrage (min g = %.2e) even at "
                    "maximum penalty — the quotes themselves are inconsistent.",
                    T, min_g)
    if rmse > 0.02:
        LOG.warning("T=%.3f: SVI fit RMSE is %.2f vol points — inspect the quotes "
                    "for this expiry.", T, rmse * 100)
    return out


def check_calendar_arbitrage(slices: list[SVIParams],
                             k_grid: np.ndarray | None = None,
                             tol: float = 1e-6) -> dict[str, object]:
    """
    Calendar-spread arbitrage test.

    Total variance must be non-decreasing in maturity at every fixed
    log-moneyness: w(k, T1) <= w(k, T2) for T1 < T2. A violation means a
    calendar spread with a guaranteed payoff — usually a stale front-month
    quote rather than a genuine opportunity, but always worth flagging.
    """
    ordered = sorted([s for s in slices if np.isfinite(s.T)], key=lambda s: s.T)
    if len(ordered) < 2:
        return {"has_arbitrage": False, "violations": [], "max_violation": 0.0}

    grid = np.linspace(-0.35, 0.25, 121) if k_grid is None else np.asarray(k_grid)
    violations, worst = [], 0.0
    for near, far in zip(ordered[:-1], ordered[1:]):
        diff = far.total_variance(grid) - near.total_variance(grid)
        bad = diff < -tol
        if bad.any():
            depth = float(-diff[bad].min())
            worst = max(worst, depth)
            violations.append({
                "T_near": near.T, "T_far": far.T,
                "n_k_violating": int(bad.sum()),
                "max_depth_total_var": depth,
                "k_range": (float(grid[bad].min()), float(grid[bad].max())),
            })
    return {"has_arbitrage": bool(violations), "violations": violations,
            "max_violation": worst}
