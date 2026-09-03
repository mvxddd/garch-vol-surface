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
from ..utils import cache_path, get_logger, read_cache, retry, write_cache
from .synthetic import synthetic_option_chain

LOG = get_logger("volsurface.options")

SCHEMA = ["expiry", "strike", "option_type", "bid", "ask", "last_price",
          "volume", "open_interest", "provider_iv", "spot", "asof", "synthetic"]


@retry(attempts=3, backoff=2.0, logger=LOG)
def _yf_ticker(ticker: str):
    import yfinance as yf

    tk = yf.Ticker(ticker)
    if not tk.options:
        raise ValueError(f"No listed expiries returned for {ticker}")
    return tk


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


def _load_yfinance_chain(ticker: str, opt_cfg: OptionsConfig,
                         spot: float, asof: pd.Timestamp) -> pd.DataFrame:
    tk = _yf_ticker(ticker)
    expiries = list(tk.options)

    # Keep expiries inside the configured tenor window, spread across the term
    # structure rather than clustering on the front month.
    valid = []
    for e in expiries:
        try:
            ts = pd.Timestamp(e)
        except Exception:
            continue
        dte = (ts - asof).days
        if opt_cfg.min_days_to_expiry <= dte <= opt_cfg.max_days_to_expiry:
            valid.append((ts, dte))
    if not valid:
        raise ValueError(
            f"No expiries for {ticker} within "
            f"[{opt_cfg.min_days_to_expiry}, {opt_cfg.max_days_to_expiry}] days"
        )
    if len(valid) > opt_cfg.max_expiries:
        # Log-spaced selection: dense at the front, sparse at the back — the
        # shape of the term structure you actually want to see.
        targets = np.geomspace(valid[0][1], valid[-1][1], opt_cfg.max_expiries)
        chosen, used = [], set()
        for t in targets:
            i = int(np.argmin([abs(d - t) for _, d in valid]))
            if i not in used:
                used.add(i)
                chosen.append(valid[i])
        valid = sorted(chosen, key=lambda x: x[1])

    frames = []
    for ts, dte in valid:
        try:
            chain = tk.option_chain(ts.strftime("%Y-%m-%d"))
        except Exception as exc:
            LOG.warning("Chain fetch failed for %s %s: %s", ticker, ts.date(), exc)
            continue
        for side, df in (("call", chain.calls), ("put", chain.puts)):
            if df is None or df.empty:
                continue
            part = pd.DataFrame({
                "expiry": ts,
                "strike": pd.to_numeric(df["strike"], errors="coerce"),
                "option_type": side,
                "bid": pd.to_numeric(df.get("bid"), errors="coerce"),
                "ask": pd.to_numeric(df.get("ask"), errors="coerce"),
                "last_price": pd.to_numeric(df.get("lastPrice"), errors="coerce"),
                "volume": pd.to_numeric(df.get("volume"), errors="coerce").fillna(0),
                "open_interest": pd.to_numeric(df.get("openInterest"),
                                               errors="coerce").fillna(0),
                "provider_iv": pd.to_numeric(df.get("impliedVolatility"),
                                             errors="coerce"),
            })
            frames.append(part)

    if not frames:
        raise ValueError(f"Every expiry fetch failed for {ticker}")
    out = pd.concat(frames, ignore_index=True)
    out["spot"] = spot
    out["asof"] = asof
    out["synthetic"] = False
    LOG.info("Fetched %d raw quotes across %d expiries for %s",
             len(out), out["expiry"].nunique(), ticker)
    return out


@retry(attempts=3, backoff=2.0, logger=LOG)
def _load_polygon_chain(ticker: str, opt_cfg: OptionsConfig, api_key: str,
                        spot: float, asof: pd.Timestamp) -> pd.DataFrame:
    """
    Polygon options snapshot (paid tier). Higher quality than Yahoo in three
    ways that matter: real NBBO bid/ask, reliable open interest, and no silent
    truncation of the chain.
    """
    import requests

    url = f"https://api.polygon.io/v3/snapshot/options/{ticker}"
    params = {
        "limit": 250,
        "expiration_date.gte": (asof + pd.Timedelta(days=opt_cfg.min_days_to_expiry)
                                ).strftime("%Y-%m-%d"),
        "expiration_date.lte": (asof + pd.Timedelta(days=opt_cfg.max_days_to_expiry)
                                ).strftime("%Y-%m-%d"),
        "apiKey": api_key,
    }
    rows, pages = [], 0
    while url and pages < 40:                      # hard page cap = safety valve
        resp = requests.get(url, params=params if pages == 0 else {"apiKey": api_key},
                            timeout=30)
        resp.raise_for_status()
        payload = resp.json()
        for item in payload.get("results", []):
            det = item.get("details", {})
            quote = item.get("last_quote", {}) or {}
            day = item.get("day", {}) or {}
            rows.append({
                "expiry": pd.Timestamp(det.get("expiration_date")),
                "strike": det.get("strike_price"),
                "option_type": det.get("contract_type"),
                "bid": quote.get("bid"),
                "ask": quote.get("ask"),
                "last_price": day.get("close"),
                "volume": day.get("volume", 0),
                "open_interest": item.get("open_interest", 0),
                "provider_iv": item.get("implied_volatility"),
            })
        url = payload.get("next_url")
        pages += 1
    if not rows:
        raise ValueError(f"Polygon returned no contracts for {ticker}")

    out = pd.DataFrame(rows)
    out["spot"] = spot
    out["asof"] = asof
    out["synthetic"] = False
    LOG.info("Polygon: %d contracts over %d pages", len(out), pages)
    return out


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
        if data_cfg.provider == "synthetic":
            raw = synthetic_option_chain(spot, asof=asof,
                                         r=opt_cfg.risk_free_rate,
                                         seed=data_cfg.synthetic_seed)
        elif data_cfg.provider == "polygon":
            if not data_cfg.polygon_api_key:
                raise ValueError("POLYGON_API_KEY is not set")
            raw = _load_polygon_chain(data_cfg.ticker, opt_cfg,
                                      data_cfg.polygon_api_key, spot, asof)
        else:
            raw = _load_yfinance_chain(data_cfg.ticker, opt_cfg, spot, asof)
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
