"""
Precompute a universe of underlyings.

The shape of the change this makes
----------------------------------
Today the pipeline calculates **on request**: ask for SPY and wait ~15 seconds
while it downloads a chain and fits a dozen non-linear calibrations. That is
right for research and wrong for anything serving more than one person at once.
Ten simultaneous requests would be ten simultaneous chain downloads, which
exhausts a vendor's rate limit and takes the service down with it.

A service calculates **ahead**: one scheduled pass over the whole universe, the
results persisted, and a request becomes a read. This module is that pass.

What it guarantees
------------------
* **One bad ticker cannot take down the run.** Each underlying is isolated;
  failures are recorded with their reason and the pass continues. A universe
  where three names delisted should still produce results for the other forty.
* **Progress survives a crash.** With `resume=True`, names whose snapshot is
  already stored for today are skipped, so a pass killed half-way costs only
  the remainder.
* **Rate limits are respected.** A configurable pause between names, because
  the fastest way to lose a data feed is to hammer it.

Everything it produces goes through `SurfaceHistory`, so the same store the
anomaly screen z-scores against is the one a service would read from.
"""
from __future__ import annotations

import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import pandas as pd

from .config import Config
from .history import SurfaceHistory, snapshot
from .pipeline import run_pipeline
from .utils import get_logger

LOG = get_logger("volsurface.batch")

# A reasonable starting universe: liquid index ETFs, sector ETFs, and the
# single names with the deepest option markets.
DEFAULT_UNIVERSE: tuple[str, ...] = (
    "SPY", "QQQ", "IWM", "DIA",
    "AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "META", "GOOGL",
    "GLD", "TLT", "XLF", "XLE",
)


@dataclass
class BatchResult:
    """Per-ticker outcome of one pass."""

    rows: list[dict] = field(default_factory=list)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)

    @property
    def succeeded(self) -> list[str]:
        return [r["ticker"] for r in self.rows if r["status"] == "ok"]

    @property
    def failed(self) -> list[str]:
        return [r["ticker"] for r in self.rows if r["status"] == "failed"]

    def summary(self) -> dict[str, object]:
        n = len(self.rows)
        ok = len(self.succeeded)
        return {
            "tickers": n,
            "ok": ok,
            "failed": len(self.failed),
            "skipped": sum(1 for r in self.rows if r["status"] == "skipped"),
            "success_rate_pct": round(ok / n * 100, 1) if n else 0.0,
            "total_seconds": round(sum(r.get("seconds", 0.0) for r in self.rows), 1),
        }


def _already_done(store: SurfaceHistory, ticker: str,
                  asof: pd.Timestamp) -> bool:
    hist = store.load(ticker)
    if hist.empty:
        return False
    dates = pd.to_datetime(hist["date"]).dt.normalize()
    return bool((dates == asof.normalize()).any())


def run_universe(
    tickers: Sequence[str] | None = None,
    base_config: Config | None = None,
    pause_seconds: float = 1.0,
    resume: bool = True,
    run_walk_forward: bool = False,
    make_figures: bool = False,
) -> BatchResult:
    """
    Run the pipeline across a universe, persisting a snapshot for each.

    Parameters
    ----------
    base_config : a template. `ticker` is overwritten per name; everything else
        — provider, filters, language — is shared. Pass `Config.for_production`
        to get the licence check and no synthetic fallback.
    pause_seconds : delay between names, to stay inside vendor rate limits.
    resume : skip names already snapshotted today.

    Walk-forward validation is off by default: it re-estimates the MLE dozens
    of times per name, which is minutes each, and it answers a research
    question that does not change between daily passes.
    """
    tickers = list(tickers or DEFAULT_UNIVERSE)
    template = base_config or Config()
    store = SurfaceHistory(template.analytics.history_dir)
    asof = pd.Timestamp.today().normalize()
    result = BatchResult()

    LOG.info("Universe pass over %d ticker(s), provider=%s, resume=%s",
             len(tickers), template.data.provider, resume)
    if (template.data.allow_synthetic_fallback
            and template.data.provider != "synthetic"):
        LOG.warning("Synthetic fallback is enabled. Names whose feed fails will "
                    "be rejected rather than stored, but Config.for_production "
                    "turns the fallback off entirely.")

    for i, ticker in enumerate(tickers, 1):
        ticker = ticker.strip().upper()
        if resume and _already_done(store, ticker, asof):
            LOG.info("[%d/%d] %s already snapshotted today — skipping",
                     i, len(tickers), ticker)
            result.rows.append({"ticker": ticker, "status": "skipped",
                                "seconds": 0.0})
            continue

        started = time.perf_counter()
        try:
            cfg = _clone_for(template, ticker)
            res = run_pipeline(cfg, make_figures=make_figures,
                               run_walk_forward=run_walk_forward)
            if res.surface is None:
                raise RuntimeError(
                    res.errors.get("build_surface", "no surface was built"))

            # A universe pass writes into the store the anomaly screen later
            # z-scores against, so a simulated row there is worse than a
            # missing day — it is a fabricated one that never washes out.
            # The research default falls back to the synthetic generator when a
            # feed fails, which is right for a notebook and catastrophic here:
            # a delisted or mistyped ticker came back "ok" with plausible
            # numbers that were entirely invented.
            if res.headline().get("synthetic_data") and \
                    template.data.provider != "synthetic":
                raise RuntimeError(
                    "feed returned nothing and the run fell back to synthetic "
                    "data — refusing to store it as market history")

            # Persist even when the pipeline did not, so a caller who forgot
            # save_snapshot still gets a usable store out of a universe pass.
            if not template.analytics.save_snapshot:
                store.append(snapshot(res.surface, ticker, vrp=res.vrp), ticker)

            head = res.headline()
            result.rows.append({
                "ticker": ticker, "status": "ok",
                "seconds": round(time.perf_counter() - started, 1),
                "n_quotes": head.get("n_quotes"),
                "n_expiries": head.get("n_expiries"),
                "atm_30d_iv": head.get("atm_30d_iv"),
                "vrp_vol_points": head.get("vrp_vol_points"),
                "n_anomalies": head.get("n_anomalies"),
                "calendar_arbitrage": head.get("calendar_arbitrage"),
                "error": None,
            })
            LOG.info("[%d/%d] %s ok in %.1fs — %s quotes, ATM30 %.2f%%",
                     i, len(tickers), ticker, result.rows[-1]["seconds"],
                     head.get("n_quotes"), (head.get("atm_30d_iv") or 0) * 100)
        except Exception as exc:
            # One delisting, one halted name, one vendor hiccup must not end
            # the pass — record why and move on.
            result.rows.append({
                "ticker": ticker, "status": "failed",
                "seconds": round(time.perf_counter() - started, 1),
                "error": f"{type(exc).__name__}: {exc}",
            })
            LOG.error("[%d/%d] %s FAILED: %s", i, len(tickers), ticker, exc)

        if pause_seconds and i < len(tickers):
            time.sleep(pause_seconds)

    LOG.info("Universe pass finished: %s", result.summary())
    return result


def _clone_for(template: Config, ticker: str) -> Config:
    """A copy of the template pointed at one ticker."""
    import copy

    cfg = copy.deepcopy(template)
    cfg.data.ticker = ticker
    cfg.analytics.save_snapshot = True
    return cfg


def cross_section(store: SurfaceHistory, tickers: Iterable[str],
                  asof: pd.Timestamp | None = None) -> pd.DataFrame:
    """
    Latest snapshot for every ticker, side by side.

    This is the view a universe pass exists to produce, and it is the one a
    single-name pipeline cannot: which underlying has the steepest skew today,
    where the premium is richest, which term structures are inverted.
    """
    rows = []
    for ticker in tickers:
        hist = store.load(str(ticker).upper())
        if hist.empty:
            continue
        if asof is not None:
            dates = pd.to_datetime(hist["date"]).dt.normalize()
            hist = hist[dates <= pd.Timestamp(asof).normalize()]
            if hist.empty:
                continue
        rows.append(hist.iloc[-1])
    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows).reset_index(drop=True)
    keep = ["ticker", "date", "spot", "atm_30d", "atm_90d", "rr25_30d",
            "bf25_30d", "term_slope_30_180", "vrp_vol_points"]
    return out[[c for c in keep if c in out.columns]]
