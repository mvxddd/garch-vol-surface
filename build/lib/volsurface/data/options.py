"""
Option-chain retrieval and normalisation.

The output of `load_option_chain` is a single tidy frame with one row per
quoted contract and a fixed schema, whatever the provider:

    expiry | strike | option_type | bid | ask | last_price | volume |
    open_interest | provider_iv | spot | asof | synthetic

Nothing downstream touches a provider-specific field, which is what makes the
Polygon upgrade a one-line config change rather than a rewrite.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import DataConfig, OptionsConfig
from ..utils import cache_path, get_logger, read_cache, write_cache
from .providers import get_provider
from .synthetic import synthetic_option_chain

LOG = get_logger("volsurface.options")

SCHEMA = ["expiry", "strike", "option_type", "bid", "ask", "last_price",
          "volume", "open_interest", "provider_iv", "last_trade", "spot",
          "asof", "synthetic"]


def latest_spot(ticker: str, prices: pd.DataFrame | None = None) -> float:
    """
    Best available spot: live quote if reachable, else the last close.

    Using a stale spot with a live chain tilts the entire smile, so this is
    worth getting right — but a wrong-by-a-tick spot is far better than a crash,
    hence the layered fallback.
    """
    try:
        import yfinance as yf

        info = yf.Ticker(ticker).fast_info
        for field in ("last_price", "lastPrice", "regular_market_price"):
            px = getattr(info, field, None) if not isinstance(info, dict) else info.get(field)
            if px and np.isfinite(float(px)) and float(px) > 0:
                return float(px)
    except Exception as exc:
        LOG.warning("Live spot unavailable for %s (%s) — using last close", ticker, exc)
    if prices is not None and not prices.empty:
        return float(prices["Close"].iloc[-1])
    raise ValueError(f"Could not determine spot for {ticker}")


def load_option_chain(
    data_cfg: DataConfig,
    opt_cfg: OptionsConfig,
    spot: float | None = None,
    prices: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Load and normalise a multi-expiry option chain, with caching and a
    synthetic fallback. Returns a frame conforming to `SCHEMA`.
    """
    asof = pd.Timestamp.today().normalize()
    if spot is None:
        spot = (float(prices["Close"].iloc[-1]) if prices is not None
                and not prices.empty else latest_spot(data_cfg.ticker, prices))

    key = cache_path(data_cfg.cache_dir, "chain", ticker=data_cfg.ticker,
                     provider=data_cfg.provider, asof=str(asof.date()),
                     max_exp=opt_cfg.max_expiries)
    if data_cfg.use_cache:
        cached = read_cache(key, ttl_hours=data_cfg.cache_ttl_hours)
        if cached is not None and not cached.empty:
            LOG.info("Loaded %d cached option quotes for %s",
                     len(cached), data_cfg.ticker)
            return cached

    try:
        raw = get_provider(data_cfg).option_chain(data_cfg.ticker, opt_cfg,
                                                  spot, asof)
    except Exception as exc:
        LOG.error("Option chain download failed for %s: %s", data_cfg.ticker, exc)
        if not data_cfg.allow_synthetic_fallback:
            raise
        raw = synthetic_option_chain(spot, asof=asof, r=opt_cfg.risk_free_rate,
                                     seed=data_cfg.synthetic_seed)

    for col in SCHEMA:
        if col not in raw.columns:
            raw[col] = np.nan
    raw = raw[SCHEMA].copy()
    raw["option_type"] = raw["option_type"].astype(str).str.lower().str[0].map(
        {"c": "call", "p": "put"}
    )
    raw["expiry"] = pd.to_datetime(raw["expiry"])
    raw = raw.dropna(subset=["expiry", "strike", "option_type"])

    if data_cfg.use_cache:
        write_cache(raw, key)
    return raw.reset_index(drop=True)
