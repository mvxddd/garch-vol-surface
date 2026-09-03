"""Historical price loading (yfinance / Polygon / synthetic) with caching."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import DataConfig
from ..utils import cache_path, get_logger, read_cache, retry, write_cache
from .synthetic import synthetic_prices

LOG = get_logger("volsurface.prices")

_OHLCV = ["Open", "High", "Low", "Close", "Volume"]


def _flatten_columns(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    yfinance returns a MultiIndex (field, ticker) for some call signatures and
    flat columns for others, and it has changed twice in recent releases.
    Normalise both shapes to flat OHLCV so downstream code never cares.
    """
    if isinstance(df.columns, pd.MultiIndex):
        levels = df.columns.get_level_values
        if ticker in set(levels(-1)):
            df = df.xs(ticker, axis=1, level=-1)
        elif ticker in set(levels(0)):
            df = df.xs(ticker, axis=1, level=0)
        else:
            df.columns = [c[0] for c in df.columns]
    df.columns = [str(c).title().replace("Adj Close", "Adj Close") for c in df.columns]
    return df


@retry(attempts=3, backoff=1.5, logger=LOG)
def _download_yfinance(ticker: str, start: str, end: str | None) -> pd.DataFrame:
    import yfinance as yf

    hist = yf.Ticker(ticker).history(
        start=start, end=end, interval="1d", auto_adjust=True, actions=False,
    )
    if hist is None or hist.empty:
        raise ValueError(f"yfinance returned no rows for {ticker}")
    hist = _flatten_columns(hist, ticker)
    hist.index = pd.to_datetime(hist.index).tz_localize(None)
    return hist


@retry(attempts=3, backoff=1.5, logger=LOG)
def _download_polygon(ticker: str, start: str, end: str | None,
                      api_key: str) -> pd.DataFrame:
    """Polygon aggregates endpoint — the recommended production upgrade."""
    import requests

    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    url = (f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/"
           f"{start}/{end}")
    resp = requests.get(
        url, params={"adjusted": "true", "sort": "asc", "limit": 50_000,
                     "apiKey": api_key},
        timeout=30,
    )
    resp.raise_for_status()
    payload = resp.json()
    results = payload.get("results") or []
    if not results:
        raise ValueError(f"Polygon returned no rows for {ticker}: "
                         f"{payload.get('status')}")
    df = pd.DataFrame(results)
    df["Date"] = pd.to_datetime(df["t"], unit="ms")
    df = (df.rename(columns={"o": "Open", "h": "High", "l": "Low",
                             "c": "Close", "v": "Volume"})
            .set_index("Date")[_OHLCV])
    return df


def load_prices(cfg: DataConfig) -> pd.DataFrame:
    """
    Load adjusted daily OHLCV for `cfg.ticker`.

    Resolution order: on-disk cache -> configured provider -> synthetic
    fallback (only if `cfg.allow_synthetic_fallback`). Any failure is logged
    loudly; the caller can always check `df.attrs['synthetic']`.
    """
    key = cache_path(cfg.cache_dir, "prices", ticker=cfg.ticker,
                     start=cfg.start, end=cfg.end, provider=cfg.provider)
    if cfg.use_cache:
        cached = read_cache(key, ttl_hours=max(cfg.cache_ttl_hours, 12.0))
        if cached is not None and not cached.empty:
            df = cached.set_index("Date")
            df.attrs["synthetic"] = bool(df.attrs.get("synthetic", False))
            LOG.info("Loaded %d cached price rows for %s", len(df), cfg.ticker)
            return df

    df: pd.DataFrame | None = None
    if cfg.provider == "synthetic":
        df = synthetic_prices(cfg.start, cfg.end, seed=cfg.synthetic_seed)
    else:
        try:
            if cfg.provider == "polygon":
                if not cfg.polygon_api_key:
                    raise ValueError("POLYGON_API_KEY is not set")
                df = _download_polygon(cfg.ticker, cfg.start, cfg.end,
                                       cfg.polygon_api_key)
            else:
                df = _download_yfinance(cfg.ticker, cfg.start, cfg.end)
            LOG.info("Downloaded %d price rows for %s from %s",
                     len(df), cfg.ticker, cfg.provider)
        except Exception as exc:
            LOG.error("Price download failed for %s (%s): %s",
                      cfg.ticker, cfg.provider, exc)
            if not cfg.allow_synthetic_fallback:
                raise
            df = synthetic_prices(cfg.start, cfg.end, seed=cfg.synthetic_seed)

    keep = [c for c in _OHLCV if c in df.columns]
    if "Close" not in keep:
        raise ValueError(f"Price frame for {cfg.ticker} has no Close column: "
                         f"{list(df.columns)}")
    df = df[keep].copy()

    # Data hygiene: drop non-positive prices, de-duplicate, sort, forward-fill
    # a *single* missing session (exchange glitch) but never a longer gap.
    df = df[~df.index.duplicated(keep="last")].sort_index()
    bad = (df["Close"] <= 0) | df["Close"].isna()
    if bad.any():
        LOG.warning("Dropping %d rows with non-positive/NaN close", int(bad.sum()))
        df = df[~bad]
    df[keep] = df[keep].ffill(limit=1)
    df.attrs.setdefault("synthetic", False)
    df.attrs["ticker"] = cfg.ticker

    if cfg.use_cache:
        out = df.reset_index().rename(columns={df.index.name or "index": "Date"})
        write_cache(out, key)
    return df


def compute_returns(prices: pd.DataFrame, field: str = "Close",
                    winsor: float | None = 0.0005) -> pd.Series:
    """
    Daily **log** returns.

    Log returns are used because GARCH is a model of additive shocks and
    because h-day aggregation is then a simple sum. Optional light winsorising
    (0.05% each tail by default) stops a single bad print — a split not yet
    adjusted, a 1987-style outlier — from dominating the MLE. Set
    `winsor=None` to disable.
    """
    if field not in prices.columns:
        field = "Close"
    px = prices[field].astype(float)
    ret = np.log(px / px.shift(1)).dropna()
    ret.name = "log_return"

    if winsor:
        lo, hi = ret.quantile(winsor), ret.quantile(1 - winsor)
        n_clipped = int(((ret < lo) | (ret > hi)).sum())
        if n_clipped:
            LOG.info("Winsorising %d extreme returns at [%.3f%%, %.3f%%]",
                     n_clipped, lo * 100, hi * 100)
        ret = ret.clip(lo, hi)

    if ret.empty:
        raise ValueError("Return series is empty after cleaning.")
    return ret


# --------------------------------------------------------------------------- #
# Implied-volatility index (for the historical volatility-risk-premium study)
# --------------------------------------------------------------------------- #
VOL_INDEX_MAP = {
    "SPY": "^VIX", "SPX": "^VIX", "^GSPC": "^VIX", "ES=F": "^VIX", "VOO": "^VIX",
    "QQQ": "^VXN", "^NDX": "^VXN", "NQ=F": "^VXN",
    "IWM": "^RVX", "DIA": "^VXD", "GLD": "^GVZ", "USO": "^OVX", "TLT": "^VXTLT",
}


def load_vol_index(cfg: DataConfig, index_ticker: str | None = None
                   ) -> pd.Series | None:
    """
    Daily history of the listed implied-vol index for this underlying (VIX for
    SPY/SPX, VXN for QQQ, RVX for IWM, ...), returned in **decimal** vol.

    This is the only practical way to get a long implied-volatility history
    without a paid options archive: the index is a model-free 30-day implied
    vol computed from the whole option chain, published daily since 1990. It
    stands in for "ATM implied vol" in the historical VRP study.

    Returns None (with a warning) when no index maps to the ticker or the
    download fails — the VRP *snapshot* never depends on this, only the
    historical time series does.
    """
    idx = index_ticker or VOL_INDEX_MAP.get(cfg.ticker.upper())
    if idx is None:
        LOG.info("No listed vol index maps to %s — skipping the historical "
                 "VRP study (the current-snapshot VRP is unaffected).", cfg.ticker)
        return None

    if cfg.provider == "synthetic":
        return None
    try:
        hist = _download_yfinance(idx, cfg.start, cfg.end)
        s = (hist["Close"].astype(float) / 100.0).rename(f"{idx}_iv")
        LOG.info("Loaded %d observations of %s (mean %.1f%%)",
                 len(s), idx, s.mean() * 100)
        return s
    except Exception as exc:
        LOG.warning("Could not load vol index %s: %s", idx, exc)
        return None


def synthetic_vol_index(returns: pd.Series, premium: float = 1.28,
                        seed: int = 42) -> pd.Series:
    """
    A VIX-like implied-vol series for the synthetic/offline path.

    Built as an **ex-ante** quantity — an EWMA of past squared returns, scaled
    up by a constant premium and jittered — so the resulting VRP study is not
    circular. It is a stand-in for a real index, not a substitute: it is
    labelled synthetic wherever it surfaces.
    """
    rng = np.random.default_rng(seed)
    r = pd.Series(returns).dropna().astype(float)
    ewma_var = r.pow(2).ewm(alpha=1 - 0.94, adjust=False).mean()
    iv = np.sqrt(ewma_var * 252) * premium
    iv = iv * np.exp(rng.normal(0, 0.05, len(iv)))
    out = iv.rename("synthetic_iv")
    out.attrs["synthetic"] = True
    return out
