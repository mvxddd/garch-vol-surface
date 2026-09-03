"""
Synthetic market generator — the project's offline safety net.

Yahoo rate-limits, delists, and occasionally returns empty frames; option
chains vanish on holidays. A pipeline that dies in those cases is useless for a
demo, an interview, or CI. So every loader falls back to this module, which
produces data with the *stylistic properties that matter*:

* returns with volatility clustering and a fat left tail (GJR-GARCH dynamics);
* an option surface generated from a genuine SVI parameterisation with a
  downward equity skew and an upward-sloping term structure;
* realistic bid/ask spreads that widen in the wings and in short maturities.

The output is deliberately labelled `synthetic=True` everywhere so it can never
be mistaken for real market data in a report.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import CALENDAR_DAYS, TRADING_DAYS
from ..models.black_scholes import black76_price
from ..utils import get_logger

LOG = get_logger("volsurface.synthetic")


def synthetic_prices(
    start: str = "2016-01-01",
    end: str | None = None,
    s0: float = 450.0,
    seed: int = 42,
    mu: float = 0.08,
) -> pd.DataFrame:
    """Daily OHLCV path driven by a GJR-GARCH(1,1,1) with Student-t shocks."""
    idx = pd.bdate_range(start=start, end=end or pd.Timestamp.today().normalize())
    n = len(idx)
    rng = np.random.default_rng(seed)

    omega, alpha, gamma, beta = 1.2e-6, 0.04, 0.09, 0.90
    nu = 6.0
    h = np.empty(n)
    r = np.empty(n)
    h[0] = omega / max(1 - alpha - gamma / 2 - beta, 1e-6)
    drift = mu / TRADING_DAYS
    for t in range(n):
        if t > 0:
            h[t] = (omega + (alpha + gamma * (r[t - 1] < 0)) * r[t - 1] ** 2
                    + beta * h[t - 1])
        z = rng.standard_t(nu) / np.sqrt(nu / (nu - 2))     # unit variance
        r[t] = drift - 0.5 * h[t] + np.sqrt(h[t]) * z

    close = s0 * np.exp(np.cumsum(r))
    intraday = np.sqrt(h) * rng.uniform(0.4, 1.1, n)
    high = close * np.exp(np.abs(intraday))
    low = close * np.exp(-np.abs(intraday * rng.uniform(0.5, 1.0, n)))
    open_ = np.r_[s0, close[:-1]] * np.exp(rng.normal(0, 0.0015, n))

    df = pd.DataFrame(
        {"Open": open_, "High": np.maximum.reduce([high, close, open_]),
         "Low": np.minimum.reduce([low, close, open_]), "Close": close,
         "Volume": rng.lognormal(17.5, 0.35, n).round()},
        index=idx,
    )
    df.index.name = "Date"
    df.attrs["synthetic"] = True
    LOG.warning("Using SYNTHETIC price data (%d rows) — not real market data.", n)
    return df


def _svi_slice_params(T: float, base_vol: float = 0.16,
                      rho: float = -0.70) -> tuple[float, float, float, float, float]:
    """
    Raw-SVI parameters for one tenor, chosen to be **arbitrage-free by
    construction** and to reproduce the three empirical regularities of an
    equity surface.

    Generating the ground truth from SVI itself (rather than, say, a quadratic
    in log-moneyness) matters for two reasons:

    * a quadratic smile has unbounded curvature and genuinely *is* butterfly-
      arbitrageable in the wings at long tenors — so it would make the
      arbitrage diagnostics fire on data that is supposed to be clean;
    * it turns the calibration test into an honest round-trip: fit SVI to
      prices generated from SVI and you should recover the parameters.

    Calibration targets: ~16% ATM front-month rising to ~20% at one year
    (contango), rho = -0.70 (crash skew), and ~7-9 vol points of skew per
    one-standard-deviation move.
    """
    atm = base_vol + 0.045 * (1.0 - np.exp(-1.8 * T))
    w_atm = atm ** 2 * T                       # ATM total variance
    b = 0.145 * T ** 0.60                      # wing angle, grows with tenor
    m = -0.02 * np.sqrt(T)                     # slight left shift of the vertex
    sigma = 0.35 * np.sqrt(T) + 0.05           # vertex roundedness
    # Solve `a` so that w(0) is exactly the target ATM total variance.
    a = w_atm - b * (rho * (-m) + np.sqrt(m ** 2 + sigma ** 2))
    return a, b, rho, m, sigma


def _svi_total_variance(k: np.ndarray, a, b, rho, m, sigma) -> np.ndarray:
    """Raw SVI: w(k) = a + b [ rho (k-m) + sqrt((k-m)^2 + sigma^2) ]."""
    km = np.asarray(k, dtype=float) - m
    return a + b * (rho * km + np.sqrt(km ** 2 + sigma ** 2))


def synthetic_option_chain(
    spot: float,
    asof: pd.Timestamp | None = None,
    r: float = 0.042,
    q: float = 0.013,
    seed: int = 42,
    expiry_days: tuple[int, ...] = (14, 30, 60, 91, 122, 182, 273, 365),
    n_strikes: int = 33,
    base_vol: float = 0.16,
) -> pd.DataFrame:
    """
    Build a full synthetic chain by pricing an arbitrage-free SVI surface with
    Black-76 and adding realistic microstructure noise.

    The strike ladder is scaled by sigma*sqrt(T) rather than a fixed percentage
    band, because that is how listed chains are actually populated: a two-week
    option lists +/-8% strikes, a one-year option lists +/-40%. A fixed band
    would manufacture 250%-vol "quotes" at the front that no market maker
    would ever show.
    """
    rng = np.random.default_rng(seed)
    asof = pd.Timestamp(asof or pd.Timestamp.today().normalize())
    rows = []

    for dte in expiry_days:
        T = dte / CALENDAR_DAYS
        F = spot * np.exp((r - q) * T)
        a, b, rho, m, sig = _svi_slice_params(T, base_vol)
        atm_vol = float(np.sqrt(max(_svi_total_variance(0.0, a, b, rho, m, sig), 1e-12) / T))

        width = max(atm_vol * np.sqrt(T), 0.03)
        k_grid = np.linspace(-2.5 * width, 1.8 * width, n_strikes)
        tick = 0.5 if spot > 100 else 0.1
        strikes = np.unique(np.round(F * np.exp(k_grid) / tick) * tick)
        k = np.log(strikes / F)
        iv = np.sqrt(np.maximum(_svi_total_variance(k, a, b, rho, m, sig), 1e-10) / T)

        for K, sigma_true in zip(strikes, iv):
            for is_call in (True, False):
                mid = float(black76_price(F, K, T, sigma_true, r=r, is_call=is_call))
                if mid < 0.02:                     # exchanges do not quote these
                    continue
                # Spreads widen in the wings and at the very front of the curve.
                wing = abs(np.log(K / F))
                rel_spread = 0.010 + 0.25 * wing + 0.02 / np.sqrt(max(dte, 1))
                half = max(mid * rel_spread, 0.01)
                mid_noisy = mid * (1 + rng.normal(0, 0.0020))
                rows.append({
                    "expiry": asof + pd.Timedelta(days=int(dte)),
                    "strike": float(K),
                    "option_type": "call" if is_call else "put",
                    "bid": max(round(mid_noisy - half, 2), 0.01),
                    "ask": round(mid_noisy + half, 2),
                    "last_price": round(float(mid_noisy), 2),
                    "volume": int(rng.lognormal(6.0 - 12 * wing, 1.0)),
                    "open_interest": int(rng.lognormal(7.5 - 10 * wing, 1.0)),
                    "provider_iv": float(sigma_true),   # ground truth for tests
                    "spot": float(spot),
                    "asof": asof,
                    "synthetic": True,
                })

    df = pd.DataFrame(rows)
    LOG.warning("Using SYNTHETIC option chain (%d quotes across %d expiries).",
                len(df), len(expiry_days))
    return df
