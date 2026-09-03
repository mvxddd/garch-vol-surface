"""
Skew, term-structure and relative-value screening.

The surface is only the map. This module reads it: it measures the standard
desk quotes (risk reversal, butterfly, ATM skew, forward vol), then looks for
places where the market's own quotes are internally inconsistent or where one
tenor is out of line with its neighbours.

A word on what "anomaly" means here. Almost every flag this module raises has
a boring explanation — a stale quote, a dividend, an earnings date inside one
expiry but not the next. That is precisely why the output includes the
*evidence* (bid/ask width in vol points, quote counts, depth of the violation)
rather than just a signal. The screen's job is to hand a human a short,
ranked list to look at, not to place trades.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import CALENDAR_DAYS, AnalyticsConfig
from ..utils import get_logger, zscore

LOG = get_logger("volsurface.skew")


# --------------------------------------------------------------------------- #
# Measurement
# --------------------------------------------------------------------------- #
def skew_metrics(surface, delta: float = 0.25) -> pd.DataFrame:
    """
    Standard smile quotes at every calibrated tenor.

    risk_reversal : IV(delta put) - IV(delta call). Positive = puts bid over
        calls = the equity crash skew. Quoted in vol points.
    butterfly     : mean(wing IVs) - ATM IV. Measures how fat the wings are,
        i.e. how much the market pays for convexity beyond the ATM.
    atm_skew      : dIV/dk at the money — the local slope, the number that
        drives a delta-hedged position's exposure to spot/vol correlation.
    """
    rows = []
    for T in surface.maturities:
        atm = float(surface.atm_vol(T))
        iv_put = surface.iv_for_delta(delta, float(T), is_call=False)
        iv_call = surface.iv_for_delta(delta, float(T), is_call=True)
        # Central difference in log-moneyness for the ATM slope.
        eps = 0.01
        slope = float((surface.iv(eps, T) - surface.iv(-eps, T)) / (2 * eps))
        curv = float((surface.iv(eps, T) - 2 * atm + surface.iv(-eps, T)) / eps ** 2)

        rows.append({
            "days": int(round(float(T) * CALENDAR_DAYS)),
            "T": float(T),
            "forward": float(surface.forward(T)),
            "atm_iv": atm,
            f"iv_{int(delta*100)}d_put": iv_put,
            f"iv_{int(delta*100)}d_call": iv_call,
            "risk_reversal": (iv_put - iv_call) * 100,
            "butterfly": (0.5 * (iv_put + iv_call) - atm) * 100,
            "atm_skew": slope,
            "atm_curvature": curv,
        })
    return pd.DataFrame(rows)


def term_structure_metrics(surface) -> pd.DataFrame:
    """
    ATM term structure with the **forward volatility** between consecutive
    tenors — the quantity a calendar spread actually trades.

        w(T) = IV(T)^2 T   (total variance)
        fwd_var(T1,T2) = [w(T2) - w(T1)] / (T2 - T1)

    A negative forward variance is a hard calendar arbitrage; a very low one is
    the softer version of the same signal.
    """
    ts = surface.atm_term_structure().copy()
    ts["total_variance"] = ts["atm_iv"] ** 2 * ts["T"]
    ts["fwd_variance"] = np.nan
    ts["fwd_vol"] = np.nan

    dw = ts["total_variance"].diff()
    dt = ts["T"].diff()
    fwd_var = dw / dt
    ts["fwd_variance"] = fwd_var
    ts["fwd_vol"] = np.sqrt(fwd_var.clip(lower=0))
    ts["slope_vs_prev_vp"] = ts["atm_iv"].diff() * 100
    ts["shape"] = np.where(ts["atm_iv"].diff() > 0, "contango",
                           np.where(ts["atm_iv"].diff() < 0, "backwardation", "flat"))
    return ts


def smile_residuals(surface) -> pd.DataFrame:
    """
    Per-quote residual of the market IV against the calibrated smile, expressed
    in units of that quote's own bid/ask width in vol.

    `resid_in_spreads` > 1 means the market's mid is further from the fitted
    smile than half its own bid/ask — the only residual worth a second look,
    because anything smaller is inside the noise you would pay to cross.
    """
    if surface.quotes is None or surface.quotes.empty:
        return pd.DataFrame()
    q = surface.quotes.copy()
    q["iv_model"] = surface.iv(q["k"].to_numpy(), q["T"].to_numpy())
    q["resid_vol_points"] = (q["iv"] - q["iv_model"]) * 100
    half_spread = (q.get("iv_spread", pd.Series(np.nan, index=q.index)).abs() / 2)
    q["half_spread_vol_points"] = half_spread * 100
    q["resid_in_spreads"] = q["resid_vol_points"] / q["half_spread_vol_points"].replace(0, np.nan)
    cols = ["expiry", "days", "strike", "option_type", "k", "delta", "iv",
            "iv_model", "resid_vol_points", "half_spread_vol_points",
            "resid_in_spreads", "vega", "open_interest"]
    q["days"] = (q["T"] * CALENDAR_DAYS).round().astype(int)
    return q[[c for c in cols if c in q.columns]].sort_values(
        "resid_in_spreads", key=lambda s: s.abs(), ascending=False
    ).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Screening
# --------------------------------------------------------------------------- #
def _severity(z: float) -> str:
    a = abs(z)
    if a >= 4:
        return "high"
    if a >= 3:
        return "medium"
    return "low"


def detect_anomalies(surface, cfg: AnalyticsConfig | None = None,
                     vrp: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Screen the surface for relative-value candidates and hard inconsistencies.

    Five families of check, ordered from "this is arbitrage" to "this is a
    statistical outlier that deserves a look":

    1. **Calendar arbitrage** — total variance decreasing in maturity.
    2. **Butterfly arbitrage** — a smile implying a negative density.
    3. **Term-structure kinks** — one tenor's ATM vol far from the smooth curve
       fitted through its neighbours (in total-variance space).
    4. **Skew kinks** — a tenor's 25-delta risk reversal or butterfly out of
       line with the cross-sectional decay in maturity.
    5. **Individual quote outliers** — a single strike far from its own
       calibrated smile *relative to its bid/ask*.

    Returns a ranked frame; empty means the surface is internally clean, which
    is the normal outcome on a liquid name and is itself a useful result.
    """
    cfg = cfg or AnalyticsConfig()
    z_thr = cfg.anomaly_z_threshold
    findings: list[dict] = []

    # -- 1. calendar ------------------------------------------------------- #
    cal = surface.calendar_arbitrage()
    for v in cal.get("violations", []):
        findings.append({
            "category": "calendar_arbitrage",
            "tenor_days": int(round(v["T_far"] * CALENDAR_DAYS)),
            "metric": "total variance decreasing in T",
            "value": -v["max_depth_total_var"],
            "benchmark": 0.0,
            "z_score": np.nan,
            "severity": "high",
            "detail": (f"w({v['T_far']:.3f}) < w({v['T_near']:.3f}) at "
                       f"{v['n_k_violating']} log-moneyness points"),
            "suggested_action": ("Buy the far-dated / sell the near-dated option "
                                 "at the affected strikes — but first verify both "
                                 "quotes are live and check for a dividend or "
                                 "special event between the two expiries."),
        })

    # -- 2. butterfly ------------------------------------------------------ #
    for s in surface.slices:
        if getattr(s, "butterfly_free", True):
            continue
        findings.append({
            "category": "butterfly_arbitrage",
            "tenor_days": int(round(s.T * CALENDAR_DAYS)),
            "metric": "min Durrleman g(k)",
            "value": float(getattr(s, "min_durrleman_g", np.nan)),
            "benchmark": 0.0,
            "z_score": np.nan,
            "severity": "high",
            "detail": "Calibrated smile implies a negative risk-neutral density",
            "suggested_action": ("Inspect the quotes in the affected wing; a genuine "
                                 "violation is a butterfly spread, but a stale wing "
                                 "quote is the far more likely cause."),
        })

    # -- 3. term-structure kinks ------------------------------------------- #
    ts = term_structure_metrics(surface)
    if len(ts) >= 4:
        # Fit log(total variance) ~ a + b log(T): the smooth benchmark shape.
        x = np.log(ts["T"].to_numpy())
        y = np.log(np.maximum(ts["total_variance"].to_numpy(), 1e-12))
        coef = np.polyfit(x, y, 1)
        resid = y - np.polyval(coef, x)
        z = zscore(resid)
        for i, row in ts.iterrows():
            if abs(z[i]) >= z_thr:
                findings.append({
                    "category": "term_structure_kink",
                    "tenor_days": int(row["days"]),
                    "metric": "ATM total variance vs fitted log-linear term structure",
                    "value": float(row["atm_iv"]),
                    "benchmark": float(np.sqrt(np.exp(np.polyval(coef, x[i])) / row["T"])),
                    "z_score": float(z[i]),
                    "severity": _severity(z[i]),
                    "detail": (f"ATM {row['atm_iv']*100:.2f}% sits {z[i]:+.1f} sd from the "
                               f"smooth term structure through the other tenors"),
                    "suggested_action": ("Calendar spread against the neighbouring "
                                         "tenors. Check the earnings calendar first — "
                                         "an event inside one expiry explains most kinks."),
                })
    # Negative forward variance is a softer restatement of a calendar arb.
    for _, row in ts.iterrows():
        if np.isfinite(row["fwd_variance"]) and row["fwd_variance"] < 0:
            findings.append({
                "category": "negative_forward_variance",
                "tenor_days": int(row["days"]),
                "metric": "forward variance vs previous tenor",
                "value": float(row["fwd_variance"]),
                "benchmark": 0.0,
                "z_score": np.nan,
                "severity": "high",
                "detail": "Implied forward variance between listed tenors is negative",
                "suggested_action": "Calendar spread; verify quotes before acting.",
            })

    # -- 4. skew kinks ----------------------------------------------------- #
    sk = skew_metrics(surface)
    if len(sk) >= 4:
        for col, label in (("risk_reversal", "25d risk reversal"),
                           ("butterfly", "25d butterfly")):
            vals = sk[col].to_numpy()
            ok = np.isfinite(vals)
            if ok.sum() < 4:
                continue
            # Skew decays smoothly in 1/sqrt(T); regress on that basis.
            basis = 1.0 / np.sqrt(sk["T"].to_numpy())
            coef = np.polyfit(basis[ok], vals[ok], 1)
            resid = vals - np.polyval(coef, basis)
            z = zscore(np.where(ok, resid, np.nan))
            for i in range(len(sk)):
                if ok[i] and abs(z[i]) >= z_thr:
                    findings.append({
                        "category": "skew_kink",
                        "tenor_days": int(sk["days"].iloc[i]),
                        "metric": label,
                        "value": float(vals[i]),
                        "benchmark": float(np.polyval(coef, basis[i])),
                        "z_score": float(z[i]),
                        "severity": _severity(z[i]),
                        "detail": (f"{label} of {vals[i]:+.2f} vol points is {z[i]:+.1f} sd "
                                   f"from the 1/sqrt(T) decay fitted across tenors"),
                        "suggested_action": (
                            "Risk reversal (or butterfly) against the adjacent tenor, "
                            "delta-hedged. Confirm both wings have two-sided markets."),
                    })

    # -- 5. individual quote outliers -------------------------------------- #
    resid_df = smile_residuals(surface)
    if not resid_df.empty and "resid_in_spreads" in resid_df:
        worst = resid_df[resid_df["resid_in_spreads"].abs() > 1.5].head(10)
        for _, row in worst.iterrows():
            findings.append({
                "category": "quote_outlier",
                "tenor_days": int(row["days"]),
                "metric": f"{row['option_type']} K={row['strike']:.1f} vs fitted smile",
                "value": float(row["iv"]),
                "benchmark": float(row["iv_model"]),
                "z_score": float(row["resid_in_spreads"]),
                "severity": _severity(row["resid_in_spreads"]),
                "detail": (f"Market IV is {row['resid_vol_points']:+.2f} vol points off "
                           f"the smile, {abs(row['resid_in_spreads']):.1f}x its own "
                           f"half-spread ({row['half_spread_vol_points']:.2f} vp)"),
                "suggested_action": ("Single-strike relative value against the smile. "
                                     "Check open interest and last-trade time — an "
                                     "untraded strike quoted wide is not an opportunity."),
            })

    # -- 6. VRP extremes (optional) ---------------------------------------- #
    if vrp is not None and not vrp.empty and "vrp_zscore" in vrp.columns:
        latest = vrp.dropna(subset=["vrp_zscore"]).tail(1)
        for _, row in latest.iterrows():
            if abs(row["vrp_zscore"]) >= z_thr:
                findings.append({
                    "category": "vrp_extreme",
                    "tenor_days": np.nan,
                    "metric": "volatility risk premium z-score",
                    "value": float(row.get("vrp", np.nan)),
                    "benchmark": 0.0,
                    "z_score": float(row["vrp_zscore"]),
                    "severity": _severity(row["vrp_zscore"]),
                    "detail": "VRP is far from its own trailing distribution",
                    "suggested_action": ("Directional volatility exposure (variance swap "
                                         "or delta-hedged straddle) in the direction of "
                                         "mean reversion."),
                })

    if not findings:
        LOG.info("Anomaly screen: surface is internally consistent — no flags.")
        return pd.DataFrame(columns=[
            "category", "tenor_days", "metric", "value", "benchmark", "z_score",
            "severity", "detail", "suggested_action"])

    out = pd.DataFrame(findings)
    order = {"high": 0, "medium": 1, "low": 2}
    out["_rank"] = out["severity"].map(order).fillna(3)
    out = (out.sort_values(["_rank", "z_score"],
                           key=lambda s: s.abs() if s.name == "z_score" else s,
                           ascending=[True, False])
             .drop(columns="_rank").reset_index(drop=True))
    LOG.info("Anomaly screen: %d flag(s) — %s", len(out),
             ", ".join(f"{k}={v}" for k, v in out["category"].value_counts().items()))
    return out
