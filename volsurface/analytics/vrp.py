"""
Volatility risk premium: what the market charges for volatility versus what
the model says volatility will actually be.

    VRP(T) = ATM implied vol(T) - GARCH forecast vol(T)

The premium is positive on average — historically ~2-4 vol points on equity
indices — because selling options means selling insurance against exactly the
states of the world in which investors most need it. That persistent wedge is
the P&L engine behind short-vol strategies, and its *variation* is the signal:
a VRP two standard deviations above its own history says options are expensive
relative to the model, and vice versa.

Two things this module is careful about, because they are the usual mistakes:

1. **Horizon matching.** A 30-day implied vol must be compared with a GARCH
   forecast of average variance over the next 30 days — not with a one-day
   conditional vol, and not with trailing realised vol.
2. **Realised vol is the ex-post truth, implied is the ex-ante price.** The
   historical VRP series therefore aligns implied at t with realised over
   (t, t+h]; the last h observations are necessarily unresolved and are
   reported as NaN rather than quietly dropped.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import CALENDAR_DAYS, TRADING_DAYS
from ..utils import get_logger

LOG = get_logger("volsurface.vrp")


def current_vrp(surface, garch_term_structure: pd.DataFrame,
                horizons_days: tuple[int, ...] | list[int] = (21, 63)
                ) -> pd.DataFrame:
    """
    Snapshot VRP across the term structure, as of the surface's `asof` date.

    `garch_term_structure` is the output of
    `models.garch.forecast_term_structure` — columns
    [horizon_days, horizon_years, garch_vol_ann], in **trading** days. Implied
    vols live on a **calendar**-day clock, so the two are matched by converting
    the GARCH horizon into calendar time (h_trading / 252 years) before
    querying the surface. Mixing the two clocks is a ~20% error in T and shows
    up as a spurious term-structure signal.
    """
    if garch_term_structure.empty:
        raise ValueError("garch_term_structure is empty")

    gts = garch_term_structure.copy()
    gts["T"] = gts["horizon_days"] / TRADING_DAYS          # year fraction
    rows = []
    for h in sorted({int(x) for x in horizons_days}):
        if h not in set(gts["horizon_days"]):
            # Interpolate the GARCH term structure in variance space.
            t = h / TRADING_DAYS
            var = np.interp(t, gts["T"], gts["garch_vol_ann"] ** 2)
            garch_vol = float(np.sqrt(max(var, 0.0)))
        else:
            garch_vol = float(gts.loc[gts["horizon_days"] == h, "garch_vol_ann"].iloc[0])

        T = h / TRADING_DAYS
        atm_iv = float(surface.atm_vol(T))
        rows.append({
            "horizon_days": h,
            "calendar_days": int(round(T * CALENDAR_DAYS)),
            "T": T,
            "atm_iv": atm_iv,
            "garch_vol": garch_vol,
            "vrp_vol_points": (atm_iv - garch_vol) * 100,
            "vrp_ratio": atm_iv / garch_vol if garch_vol > 0 else np.nan,
            # The variance premium is what a variance swap actually pays.
            "variance_premium": atm_iv ** 2 - garch_vol ** 2,
            "signal": _vrp_signal(atm_iv, garch_vol),
        })
    out = pd.DataFrame(rows)
    LOG.info("VRP snapshot: %s",
             ", ".join(f"{r.horizon_days}d {r.vrp_vol_points:+.2f}vp"
                       for r in out.itertuples()))
    return out


def _vrp_signal(atm_iv: float, garch_vol: float, rich: float = 1.15,
                cheap: float = 0.95) -> str:
    """
    Map the implied/forecast ratio to a verdict **code**.

    Returns one of "rich", "cheap", "in_line", "na". A stable code rather than
    a sentence, for two reasons: it survives translation (the display layer
    renders it in the user's language), and it is what a downstream script
    wants to filter on. `vrp_signal_text` turns it into prose.

    Thresholds are deliberately asymmetric: a positive VRP is the *normal*
    state, so it takes a large premium to call options genuinely rich, but only
    a small discount to call them cheap.
    """
    if not (np.isfinite(atm_iv) and np.isfinite(garch_vol) and garch_vol > 0):
        return "na"
    ratio = atm_iv / garch_vol
    if ratio >= rich:
        return "rich"
    if ratio <= cheap:
        return "cheap"
    return "in_line"


def vrp_signal_text(code: str) -> str:
    """Render a signal code as prose in the active display language."""
    from ..i18n import t

    return t(f"vrp.signal.{code}")


def historical_vrp(iv_proxy: pd.Series, returns: pd.Series,
                   horizon_days: int = 21) -> pd.DataFrame:
    """
    Historical VRP time series from an implied-vol index (e.g. VIX for SPY/SPX,
    VXN for QQQ) against subsequently realised volatility.

    Parameters
    ----------
    iv_proxy : implied vol in **decimal** (divide VIX by 100 before calling).
    returns  : daily log returns of the underlying.
    horizon_days : trading days over which realised vol is measured; use 21 for
        a VIX-style 30-calendar-day index.

    Returns a frame with the realised leg aligned *forward* from each date, so
    `vrp` at time t is the premium an option seller actually earned over the
    following window (positive = the seller won).
    """
    iv = pd.Series(iv_proxy).dropna().astype(float)
    r = pd.Series(returns).dropna().astype(float)
    idx = iv.index.intersection(r.index)
    if len(idx) < horizon_days * 3:
        raise ValueError(
            f"Only {len(idx)} overlapping observations — need at least "
            f"{horizon_days * 3} for a meaningful VRP history."
        )
    iv, r = iv.loc[idx], r.loc[idx]

    r2 = r ** 2
    fwd_var = r2.shift(-1)[::-1].rolling(horizon_days).mean()[::-1] * TRADING_DAYS
    rv_fwd = np.sqrt(fwd_var)

    df = pd.DataFrame({
        "implied_vol": iv,
        "realized_vol_fwd": rv_fwd,
        "vrp": iv - rv_fwd,
        "variance_premium": iv ** 2 - rv_fwd ** 2,
    })
    df["vrp_zscore"] = ((df["vrp"] - df["vrp"].rolling(TRADING_DAYS).mean())
                        / df["vrp"].rolling(TRADING_DAYS).std())
    n_unresolved = int(df["realized_vol_fwd"].isna().sum())
    if n_unresolved:
        LOG.info("%d most recent dates have no completed %dd realised window yet",
                 n_unresolved, horizon_days)
    return df


def vrp_summary(hist: pd.DataFrame) -> dict[str, float]:
    """
    Summary statistics of a historical VRP series, including the t-statistic
    for "the premium is non-zero".

    The t-stat uses a Newey-West correction with `h`-lag Bartlett weights:
    overlapping windows make the raw series strongly autocorrelated, and the
    naive t-stat on 2,000 overlapping observations is inflated by roughly
    sqrt(h) — enough to turn noise into a "highly significant" result.
    """
    v = hist["vrp"].dropna().to_numpy()
    n = v.size
    if n < 30:
        return {"n": n}

    mean = float(v.mean())
    demeaned = v - mean
    lags = max(int(np.floor(4 * (n / 100) ** (2 / 9))), 1)     # Newey-West rule
    lrv = float(np.mean(demeaned ** 2))
    for L in range(1, lags + 1):
        cov = float(np.mean(demeaned[L:] * demeaned[:-L]))
        lrv += 2 * (1 - L / (lags + 1)) * cov
    se = float(np.sqrt(max(lrv, 1e-16) / n))

    return {
        "n": n,
        "mean_vrp_vol_points": mean * 100,
        "median_vrp_vol_points": float(np.median(v)) * 100,
        "std_vol_points": float(v.std(ddof=1)) * 100,
        "pct_positive": float((v > 0).mean() * 100),
        "newey_west_t_stat": mean / se if se > 0 else np.nan,
        "nw_lags": lags,
        "worst_vol_points": float(v.min()) * 100,
        "best_vol_points": float(v.max()) * 100,
    }
