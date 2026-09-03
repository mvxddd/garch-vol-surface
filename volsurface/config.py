"""
Central configuration for the GARCH / implied-volatility-surface engine.

Everything that a user might reasonably want to tune lives here as a typed
dataclass field, so the notebook and the CLI share one source of truth and no
magic numbers are buried in the analytics code.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Literal, Sequence

# --------------------------------------------------------------------------- #
# Constants used across the whole engine
# --------------------------------------------------------------------------- #
TRADING_DAYS: int = 252          # annualisation factor for daily data
CALENDAR_DAYS: int = 365         # used for time-to-expiry in year fractions
SECONDS_PER_DAY: int = 86_400


@dataclass
class DataConfig:
    """Where market data comes from and how far back we look."""

    ticker: str = "SPY"
    # History for the GARCH fit. 8y ≈ 2000 observations: enough for stable MLE
    # without letting a 2008-style regime dominate a 2026 forecast.
    start: str = "2016-01-01"
    end: str | None = None                      # None -> today
    price_field: Literal["Close", "Adj Close"] = "Close"

    provider: Literal["yfinance", "polygon", "synthetic"] = "yfinance"
    polygon_api_key: str | None = field(
        default_factory=lambda: os.environ.get("POLYGON_API_KEY")
    )

    cache_dir: Path = Path("outputs/data")
    use_cache: bool = True
    cache_ttl_hours: float = 12.0               # options chains go stale fast
    max_retries: int = 3
    retry_backoff_sec: float = 1.5

    # If the network / provider fails, fall back to a calibrated simulator so
    # the notebook is always runnable (demos, CI, planes, interviews).
    allow_synthetic_fallback: bool = True
    synthetic_seed: int = 42


@dataclass
class GarchConfig:
    """GARCH-family estimation and out-of-sample validation settings."""

    # Models to fit and compare. (name, vol_model, p, o, q, dist)
    #   o > 0 introduces the asymmetry / leverage term.
    specs: Sequence[tuple[str, str, int, int, int, str]] = (
        ("GARCH(1,1)-Normal", "Garch", 1, 0, 1, "normal"),
        ("GARCH(1,1)-t", "Garch", 1, 0, 1, "t"),
        ("GJR-GARCH(1,1,1)-t", "Garch", 1, 1, 1, "t"),
        ("EGARCH(1,1,1)-skewt", "EGARCH", 1, 1, 1, "skewt"),
    )
    mean_model: Literal["Constant", "Zero", "AR"] = "Constant"
    # `arch` is numerically much better behaved on returns scaled to percent.
    return_scale: float = 100.0

    # Out-of-sample walk-forward validation
    oos_fraction: float = 0.30       # last 30% of the sample is held out
    forecast_horizons: Sequence[int] = (1, 5, 21, 63)   # 1d, 1w, 1m, 1q
    refit_every: int = 21            # re-estimate MLE monthly (speed vs drift)
    window: Literal["expanding", "rolling"] = "expanding"
    rolling_window_size: int = 1000


@dataclass
class OptionsConfig:
    """Options-chain retrieval and the quality filters applied to quotes."""

    max_expiries: int = 12
    min_days_to_expiry: int = 7        # sub-week options are microstructure noise
    max_days_to_expiry: int = 400

    # Quote-quality filters. Bad quotes produce a lumpy surface far faster than
    # any interpolation choice, so filtering is where most of the value is.
    min_bid: float = 0.05
    max_rel_spread: float = 0.35       # (ask-bid)/mid
    min_open_interest: int = 10
    min_volume: int = 0
    moneyness_bounds: tuple[float, float] = (0.70, 1.30)   # K / F
    max_iv: float = 3.00               # 300% vol -> almost surely a stale quote
    min_iv: float = 0.01
    min_quotes_per_expiry: int = 6     # need enough points to fit a smile

    risk_free_rate: float = 0.042      # flat fallback if no curve is supplied
    dividend_yield: float | None = None  # None -> implied from put-call parity
    use_parity_forward: bool = True    # derive F from C-P regression (preferred)


@dataclass
class SurfaceConfig:
    """Smile / surface construction."""

    # 'svi'   -> arbitrage-aware parametric fit per expiry (institutional default)
    # 'spline'-> weighted smoothing spline in log-moneyness
    # 'rbf'   -> global thin-plate interpolation over (k, T)
    method: Literal["svi", "spline", "rbf"] = "svi"
    svi_max_iter: int = 4000
    svi_n_starts: int = 8              # multi-start least squares (non-convex)
    vega_weighting: bool = True        # weight the fit by vega, not by price

    k_grid_points: int = 81            # log-moneyness nodes on the output grid
    t_grid_points: int = 41            # maturity nodes on the output grid
    k_grid_bounds: tuple[float, float] = (-0.35, 0.25)

    # Skew / risk-reversal measurement points
    skew_deltas: tuple[float, float] = (0.25, 0.25)   # (put delta, call delta)
    atm_definition: Literal["forward", "delta50"] = "forward"


@dataclass
class AnalyticsConfig:
    """Vol risk premium and relative-value screening."""

    vrp_horizons_days: Sequence[int] = (5, 21, 63, 126)
    # A term-structure / skew reading this many z-scores from its own history
    # (or from the cross-sectional fit) is flagged as a candidate anomaly.
    anomaly_z_threshold: float = 2.0
    vrp_lookback_days: int = 252


@dataclass
class Config:
    """Top-level container passed through the entire pipeline."""

    data: DataConfig = field(default_factory=DataConfig)
    garch: GarchConfig = field(default_factory=GarchConfig)
    options: OptionsConfig = field(default_factory=OptionsConfig)
    surface: SurfaceConfig = field(default_factory=SurfaceConfig)
    analytics: AnalyticsConfig = field(default_factory=AnalyticsConfig)

    output_dir: Path = Path("outputs")
    figure_dir: Path = Path("outputs/figures")
    report_dir: Path = Path("outputs/reports")
    verbose: bool = True

    def with_ticker(self, ticker: str) -> "Config":
        """Convenience for notebooks: same config, different underlying."""
        self.data.ticker = ticker
        return self

    def ensure_dirs(self) -> None:
        """Create every output directory the pipeline writes into."""
        for d in (self.output_dir, self.figure_dir, self.report_dir,
                  self.data.cache_dir):
            Path(d).mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict:
        """JSON-serialisable snapshot, embedded in every saved report."""
        def _coerce(obj):
            if isinstance(obj, Path):
                return str(obj)
            if isinstance(obj, tuple):
                return list(obj)
            return obj

        raw = asdict(self)

        def _walk(node):
            if isinstance(node, dict):
                return {k: _walk(v) for k, v in node.items()}
            if isinstance(node, (list, tuple)):
                return [_walk(v) for v in node]
            return _coerce(node)

        out = _walk(raw)
        # Never persist secrets into a report artefact.
        out["data"]["polygon_api_key"] = "***" if self.data.polygon_api_key else None
        return out
