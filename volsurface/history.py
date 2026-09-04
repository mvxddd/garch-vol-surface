"""
A store of daily surface snapshots, and the z-scores it makes possible.

Why this exists
---------------
Without history, the anomaly screen can only compare tenors **against each
other**: "the 60-day risk reversal is off the curve through its neighbours".
That is a weak signal, because the whole curve can be dislocated at once and
nothing looks unusual.

With a few months of snapshots, every quantity gets compared against **its own
past**: "30-day skew is steeper than on 97% of days in the last year". That is
the form a vol signal actually takes on a desk, and it is also the raw material
a backtest needs.

Design
------
One row per (date, ticker), one column per metric — a narrow, append-only
table in Parquet. Deliberately not a database: the whole point is that it stays
a file you can copy, diff, inspect in pandas, and commit if you want to.

Re-running the same day overwrites that day's row rather than duplicating it,
so a cron job that fires twice is harmless.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import CALENDAR_DAYS
from .utils import get_logger

LOG = get_logger("volsurface.history")

# Tenors, in calendar days, at which the term structure is sampled. Fixed
# points matter: listed expiries roll, so "the 30-day point" is the only thing
# comparable across dates.
SNAPSHOT_TENORS: tuple[int, ...] = (7, 30, 60, 90, 180, 365)

# Metrics that get a z-score. Everything else in a snapshot is context.
ZSCORE_METRICS: tuple[str, ...] = tuple(
    [f"atm_{d}d" for d in SNAPSHOT_TENORS]
    + [f"rr25_{d}d" for d in SNAPSHOT_TENORS]
    + [f"bf25_{d}d" for d in SNAPSHOT_TENORS]
    + ["term_slope_30_180", "atm_skew_30d", "vrp_vol_points"]
)


def snapshot(surface, ticker: str, vrp: pd.DataFrame | None = None,
             garch_vol_30d: float | None = None,
             asof: pd.Timestamp | None = None) -> pd.Series:
    """
    Reduce a calibrated surface to one comparable row.

    Interpolating onto **fixed** tenors is the whole trick: listed expiries
    move every week, so raw per-expiry numbers cannot be compared across dates.
    """
    asof = pd.Timestamp(asof or surface.asof).normalize()
    row: dict[str, object] = {"date": asof, "ticker": ticker,
                              "spot": float(surface.spot),
                              "n_expiries": len(surface.slices)}

    for days in SNAPSHOT_TENORS:
        T = days / CALENDAR_DAYS
        atm = float(surface.atm_vol(T))
        put = surface.iv_for_delta(0.25, T, is_call=False)
        call = surface.iv_for_delta(0.25, T, is_call=True)
        eps = 0.01
        row[f"atm_{days}d"] = atm
        row[f"rr25_{days}d"] = (put - call) * 100 if np.isfinite(put + call) else np.nan
        row[f"bf25_{days}d"] = ((0.5 * (put + call) - atm) * 100
                                if np.isfinite(put + call) else np.nan)
        row[f"skew_{days}d"] = float((surface.iv(eps, T) - surface.iv(-eps, T))
                                     / (2 * eps))

    row["term_slope_30_180"] = (row["atm_180d"] - row["atm_30d"]) * 100
    row["atm_skew_30d"] = row["skew_30d"]

    if vrp is not None and not vrp.empty and "vrp_vol_points" in vrp:
        near = vrp.iloc[(vrp["calendar_days"] - 30).abs().argsort()].iloc[0]
        row["vrp_vol_points"] = float(near["vrp_vol_points"])
        row["garch_vol"] = float(near["garch_vol"])
    elif garch_vol_30d is not None:
        row["garch_vol"] = float(garch_vol_30d)
        row["vrp_vol_points"] = (row["atm_30d"] - float(garch_vol_30d)) * 100
    else:
        row["vrp_vol_points"] = np.nan
        row["garch_vol"] = np.nan

    return pd.Series(row)


class SurfaceHistory:
    """Append-only Parquet store of surface snapshots, one file per ticker."""

    def __init__(self, directory: str | Path = "outputs/history"):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def path_for(self, ticker: str) -> Path:
        return self.directory / f"{ticker.upper().replace('^', '_')}_surface.parquet"

    def load(self, ticker: str) -> pd.DataFrame:
        """All snapshots for a ticker, oldest first. Empty frame if none yet."""
        path = self.path_for(ticker)
        if not path.exists():
            return pd.DataFrame()
        try:
            df = pd.read_parquet(path)
        except Exception as exc:
            LOG.error("Could not read %s (%s) — treating history as empty", path, exc)
            return pd.DataFrame()
        return df.sort_values("date").reset_index(drop=True)

    def append(self, row: pd.Series, ticker: str | None = None) -> Path:
        """
        Add (or replace) one day's snapshot.

        Replacing rather than appending on a repeat run is deliberate: a
        scheduled job that fires twice, or a manual re-run after a data fix,
        must not create two rows for one date and quietly halve every z-score's
        effective sample.
        """
        ticker = ticker or str(row.get("ticker"))
        path = self.path_for(ticker)
        new = pd.DataFrame([row])
        new["date"] = pd.to_datetime(new["date"]).dt.normalize()

        existing = self.load(ticker)
        if not existing.empty:
            existing["date"] = pd.to_datetime(existing["date"]).dt.normalize()
            replaced = int((existing["date"] == new["date"].iloc[0]).sum())
            if replaced:
                LOG.info("Replacing existing snapshot for %s on %s",
                         ticker, new["date"].iloc[0].date())
                existing = existing[existing["date"] != new["date"].iloc[0]]
            combined = pd.concat([existing, new], ignore_index=True)
        else:
            combined = new

        combined = combined.sort_values("date").reset_index(drop=True)
        try:
            combined.to_parquet(path, index=False)
        except Exception as exc:
            LOG.error("Could not write history to %s: %s", path, exc)
            return path
        LOG.info("History for %s: %d snapshot(s), %s → %s", ticker, len(combined),
                 combined["date"].iloc[0].date(), combined["date"].iloc[-1].date())
        return path

    # -- the payoff ---------------------------------------------------------- #
    def zscores(self, ticker: str, lookback: int = 252,
                min_observations: int = 30) -> pd.DataFrame:
        """
        z-score of the latest snapshot against its own trailing history.

        Returns one row per metric with the current level, the trailing mean and
        standard deviation, the z-score, and the percentile — empty if there is
        not yet enough history, which is the honest answer for a young store.
        """
        hist = self.load(ticker)
        if len(hist) < min_observations:
            LOG.info("Only %d snapshot(s) for %s — need %d before z-scores mean "
                     "anything.", len(hist), ticker, min_observations)
            return pd.DataFrame()

        window = hist.tail(lookback)
        latest = window.iloc[-1]
        prior = window.iloc[:-1]

        rows = []
        for metric in ZSCORE_METRICS:
            if metric not in window.columns:
                continue
            series = pd.to_numeric(prior[metric], errors="coerce").dropna()
            value = pd.to_numeric(pd.Series([latest.get(metric)]),
                                  errors="coerce").iloc[0]
            if series.size < min_observations - 1 or not np.isfinite(value):
                continue
            mu, sd = float(series.mean()), float(series.std(ddof=1))
            z = (value - mu) / sd if sd > 0 else 0.0
            rows.append({
                "metric": metric, "value": float(value), "mean": mu, "std": sd,
                "z_score": float(z),
                "percentile": float((series < value).mean() * 100),
                "n_observations": int(series.size),
            })

        out = pd.DataFrame(rows)
        if not out.empty:
            out = out.reindex(out["z_score"].abs().sort_values(ascending=False).index)
        return out.reset_index(drop=True)

    def series(self, ticker: str, metric: str) -> pd.Series:
        """One metric through time, indexed by date — for plotting."""
        hist = self.load(ticker)
        if hist.empty or metric not in hist.columns:
            return pd.Series(dtype=float)
        return (hist.set_index(pd.to_datetime(hist["date"]))[metric]
                    .astype(float).rename(metric))

    def summary(self, ticker: str) -> dict[str, object]:
        hist = self.load(ticker)
        if hist.empty:
            return {"ticker": ticker, "n_snapshots": 0}
        return {
            "ticker": ticker,
            "n_snapshots": len(hist),
            "first": str(pd.Timestamp(hist["date"].iloc[0]).date()),
            "last": str(pd.Timestamp(hist["date"].iloc[-1]).date()),
            "metrics": int(len([c for c in hist.columns
                                if c not in {"date", "ticker"}])),
        }


def historical_anomalies(zscores: pd.DataFrame, threshold: float = 2.0
                         ) -> pd.DataFrame:
    """
    Turn a z-score table into screen findings, in the same shape the
    cross-sectional screen produces so the two can be concatenated.
    """
    if zscores.empty:
        return pd.DataFrame(columns=["category", "tenor_days", "metric", "value",
                                     "benchmark", "z_score", "severity", "detail",
                                     "suggested_action"])
    hits = zscores[zscores["z_score"].abs() >= threshold]
    rows = []
    for _, r in hits.iterrows():
        tenor = np.nan
        for part in str(r["metric"]).split("_"):
            if part.endswith("d") and part[:-1].isdigit():
                tenor = int(part[:-1])
        rich = r["z_score"] > 0
        rows.append({
            "category": "historical_extreme",
            "tenor_days": tenor,
            "metric": f"{r['metric']} vs its own {r['n_observations']}-day history",
            "value": r["value"],
            "benchmark": r["mean"],
            "z_score": r["z_score"],
            "severity": ("high" if abs(r["z_score"]) >= 3 else "medium"),
            "detail": (f"{r['metric']} at {r['value']:.3f} is {r['z_score']:+.1f} sd "
                       f"from its trailing mean ({r['percentile']:.0f}th percentile)"),
            "suggested_action": (
                "Mean-reversion candidate: fade the move if nothing fundamental "
                "changed. Check the event calendar first — a scheduled catalyst "
                "explains most historical extremes."
                if rich else
                "Level is unusually low versus its own history: a candidate to own "
                "rather than sell. Confirm the quotes are live before acting."),
        })
    return pd.DataFrame(rows)
