"""
The seam between this project and whoever sells it market data.

Everything above `data/` — cleaning, calibration, analytics, charts — is written
against a single tidy schema and has no idea where the numbers came from.
This module is the one place that knows, so swapping Yahoo for a licensed
vendor is a new class and a registration, not a rewrite.

Why that seam matters more than it looks
----------------------------------------
The default provider is `yfinance`, which scrapes an undocumented Yahoo
endpoint. That is fine for research and completely unsuitable for a product:
Yahoo's terms do not permit commercial redistribution, there is no SLA, the
response shape has changed twice in recent releases, and the rate limits are
undocumented and enforced by IP ban. Any serious deployment replaces it, and
this interface is what makes that a contained change.

What a provider is responsible for
----------------------------------
Fetching, and normalising to the schema. That is all. Retries, caching, the
synthetic fallback and every quality filter live in the loaders above, so a new
vendor does not have to re-implement — or accidentally skip — any of it.

Adding one
----------
    class MyVendor:
        name = "myvendor"
        def prices(self, ticker, start, end): -> DataFrame[Open..Close, Volume]
        def option_chain(self, ticker, cfg, spot, asof): -> DataFrame[SCHEMA]
        def vol_index(self, ticker, start, end): -> Series | None
        def is_licensed_for_redistribution(self): -> bool

    register_provider("myvendor", MyVendor)
"""
from __future__ import annotations

from typing import Callable, Protocol, runtime_checkable

import numpy as np
import pandas as pd

from ..config import DataConfig, OptionsConfig
from ..utils import get_logger, retry

LOG = get_logger("volsurface.providers")

_OHLCV = ["Open", "High", "Low", "Close", "Volume"]

# Listed implied-vol indices, by underlying.
VOL_INDEX_MAP = {
    "SPY": "^VIX", "SPX": "^VIX", "^GSPC": "^VIX", "ES=F": "^VIX", "VOO": "^VIX",
    "QQQ": "^VXN", "^NDX": "^VXN", "NQ=F": "^VXN",
    "IWM": "^RVX", "DIA": "^VXD", "GLD": "^GVZ", "USO": "^OVX", "TLT": "^VXTLT",
}


@runtime_checkable
class MarketDataProvider(Protocol):
    """What the layers above require of a data source."""

    name: str

    def prices(self, ticker: str, start: str, end: str | None) -> pd.DataFrame:
        """Daily OHLCV indexed by naive Timestamp, adjusted for splits."""
        ...

    def option_chain(self, ticker: str, cfg: OptionsConfig, spot: float,
                     asof: pd.Timestamp) -> pd.DataFrame:
        """Raw quotes, normalised to `data.options.SCHEMA`."""
        ...

    def vol_index(self, ticker: str, start: str,
                  end: str | None) -> pd.Series | None:
        """Listed implied-vol index in decimal, or None if there is none."""
        ...

    def is_licensed_for_redistribution(self) -> bool:
        """
        Whether this source may legally be resold or served to third parties.

        Deliberately part of the interface rather than a comment somewhere:
        `Config.for_production` refuses to start on a provider that answers
        False, so shipping a service on scraped data takes a conscious override
        rather than an oversight.
        """
        ...


# --------------------------------------------------------------------------- #
# Yahoo (research default)
# --------------------------------------------------------------------------- #
def _flatten_columns(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    yfinance returns a MultiIndex (field, ticker) for some call signatures and
    flat columns for others, and it has changed twice in recent releases.
    Normalise both shapes so nothing downstream cares.
    """
    if isinstance(df.columns, pd.MultiIndex):
        levels = df.columns.get_level_values
        if ticker in set(levels(-1)):
            df = df.xs(ticker, axis=1, level=-1)
        elif ticker in set(levels(0)):
            df = df.xs(ticker, axis=1, level=0)
        else:
            df.columns = [c[0] for c in df.columns]
    df.columns = [str(c).title() for c in df.columns]
    return df


class YFinanceProvider:
    """
    Yahoo Finance via `yfinance`.

    Research and demo use only. Yahoo's terms do not permit commercial
    redistribution, and this reads an undocumented endpoint with no stability
    guarantee — see the module docstring.
    """

    name = "yfinance"

    def is_licensed_for_redistribution(self) -> bool:
        return False

    @retry(attempts=3, backoff=1.5, logger=LOG)
    def prices(self, ticker: str, start: str, end: str | None) -> pd.DataFrame:
        import yfinance as yf

        hist = yf.Ticker(ticker).history(
            start=start, end=end, interval="1d", auto_adjust=True, actions=False)
        if hist is None or hist.empty:
            raise ValueError(f"yfinance returned no rows for {ticker}")
        hist = _flatten_columns(hist, ticker)
        hist.index = pd.to_datetime(hist.index).tz_localize(None)
        return hist

    def vol_index(self, ticker: str, start: str,
                  end: str | None) -> pd.Series | None:
        idx = VOL_INDEX_MAP.get(ticker.upper())
        if idx is None:
            LOG.info("No listed vol index maps to %s — the historical VRP study "
                     "is skipped (the snapshot VRP is unaffected).", ticker)
            return None
        try:
            hist = self.prices(idx, start, end)
            return (hist["Close"].astype(float) / 100.0).rename(f"{idx}_iv")
        except Exception as exc:
            LOG.warning("Could not load vol index %s: %s", idx, exc)
            return None

    @retry(attempts=3, backoff=2.0, logger=LOG)
    def _ticker(self, ticker: str):
        import yfinance as yf

        tk = yf.Ticker(ticker)
        if not tk.options:
            raise ValueError(f"No listed expiries returned for {ticker}")
        return tk

    def option_chain(self, ticker: str, cfg: OptionsConfig, spot: float,
                     asof: pd.Timestamp) -> pd.DataFrame:
        tk = self._ticker(ticker)

        valid = []
        for e in tk.options:
            try:
                ts = pd.Timestamp(e)
            except Exception:
                continue
            dte = (ts - asof).days
            if cfg.min_days_to_expiry <= dte <= cfg.max_days_to_expiry:
                valid.append((ts, dte))
        if not valid:
            raise ValueError(
                f"No expiries for {ticker} within [{cfg.min_days_to_expiry}, "
                f"{cfg.max_days_to_expiry}] days")

        if len(valid) > cfg.max_expiries:
            # Log-spaced: dense at the front where the term structure moves,
            # sparse at the back — the shape you actually want to see.
            targets = np.geomspace(valid[0][1], valid[-1][1], cfg.max_expiries)
            chosen, used = [], set()
            for t in targets:
                i = int(np.argmin([abs(d - t) for _, d in valid]))
                if i not in used:
                    used.add(i)
                    chosen.append(valid[i])
            valid = sorted(chosen, key=lambda x: x[1])

        frames = []
        for ts, _dte in valid:
            try:
                chain = tk.option_chain(ts.strftime("%Y-%m-%d"))
            except Exception as exc:
                LOG.warning("Chain fetch failed for %s %s: %s", ticker, ts.date(), exc)
                continue
            for side, df in (("call", chain.calls), ("put", chain.puts)):
                if df is None or df.empty:
                    continue
                frames.append(pd.DataFrame({
                    "expiry": ts,
                    "strike": pd.to_numeric(df["strike"], errors="coerce"),
                    "option_type": side,
                    "bid": pd.to_numeric(df.get("bid"), errors="coerce"),
                    "ask": pd.to_numeric(df.get("ask"), errors="coerce"),
                    "last_price": pd.to_numeric(df.get("lastPrice"), errors="coerce"),
                    "volume": pd.to_numeric(df.get("volume"),
                                            errors="coerce").fillna(0),
                    "open_interest": pd.to_numeric(df.get("openInterest"),
                                                   errors="coerce").fillna(0),
                    "provider_iv": pd.to_numeric(df.get("impliedVolatility"),
                                                 errors="coerce"),
                    "last_trade": pd.to_datetime(df.get("lastTradeDate"),
                                                 errors="coerce", utc=True),
                }))

        if not frames:
            raise ValueError(f"Every expiry fetch failed for {ticker}")
        out = pd.concat(frames, ignore_index=True)
        out["spot"], out["asof"], out["synthetic"] = spot, asof, False
        LOG.info("Fetched %d raw quotes across %d expiries for %s",
                 len(out), out["expiry"].nunique(), ticker)
        return out


# --------------------------------------------------------------------------- #
# Polygon (the production upgrade)
# --------------------------------------------------------------------------- #
class PolygonProvider:
    """
    Polygon.io REST.

    Licensed, with real NBBO quotes, dependable open interest and no silent
    truncation of the chain. Note that a licence for *internal* use is not a
    licence to serve the data onward: OPRA charges separately for
    redistribution to end users, and that fee usually decides the economics of
    an options product. `is_licensed_for_redistribution` therefore keys off an
    explicit flag rather than assuming a paid key is enough.
    """

    name = "polygon"

    def __init__(self, api_key: str | None, redistribution_licensed: bool = False):
        if not api_key:
            raise ValueError("POLYGON_API_KEY is not set")
        self.api_key = api_key
        self._redistribution = bool(redistribution_licensed)

    def is_licensed_for_redistribution(self) -> bool:
        return self._redistribution

    @retry(attempts=3, backoff=1.5, logger=LOG)
    def prices(self, ticker: str, start: str, end: str | None) -> pd.DataFrame:
        import requests

        end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
        resp = requests.get(
            f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start}/{end}",
            params={"adjusted": "true", "sort": "asc", "limit": 50_000,
                    "apiKey": self.api_key}, timeout=30)
        resp.raise_for_status()
        results = resp.json().get("results") or []
        if not results:
            raise ValueError(f"Polygon returned no rows for {ticker}")
        df = pd.DataFrame(results)
        df["Date"] = pd.to_datetime(df["t"], unit="ms")
        return (df.rename(columns={"o": "Open", "h": "High", "l": "Low",
                                   "c": "Close", "v": "Volume"})
                  .set_index("Date")[_OHLCV])

    def vol_index(self, ticker: str, start: str,
                  end: str | None) -> pd.Series | None:
        idx = VOL_INDEX_MAP.get(ticker.upper())
        if idx is None:
            return None
        try:
            hist = self.prices(f"I:{idx.lstrip('^')}", start, end)
            return (hist["Close"].astype(float) / 100.0).rename(f"{idx}_iv")
        except Exception as exc:
            LOG.warning("Polygon vol index %s unavailable: %s", idx, exc)
            return None

    @retry(attempts=3, backoff=2.0, logger=LOG)
    def option_chain(self, ticker: str, cfg: OptionsConfig, spot: float,
                     asof: pd.Timestamp) -> pd.DataFrame:
        import requests

        url = f"https://api.polygon.io/v3/snapshot/options/{ticker}"
        params = {
            "limit": 250,
            "expiration_date.gte": (asof + pd.Timedelta(days=cfg.min_days_to_expiry)
                                    ).strftime("%Y-%m-%d"),
            "expiration_date.lte": (asof + pd.Timedelta(days=cfg.max_days_to_expiry)
                                    ).strftime("%Y-%m-%d"),
            "apiKey": self.api_key,
        }
        rows, pages = [], 0
        while url and pages < 40:               # hard page cap = safety valve
            resp = requests.get(url, params=params if pages == 0
                                else {"apiKey": self.api_key}, timeout=30)
            resp.raise_for_status()
            payload = resp.json()
            for item in payload.get("results", []):
                det = item.get("details", {}) or {}
                quote = item.get("last_quote", {}) or {}
                day = item.get("day", {}) or {}
                rows.append({
                    "expiry": pd.Timestamp(det.get("expiration_date")),
                    "strike": det.get("strike_price"),
                    "option_type": det.get("contract_type"),
                    "bid": quote.get("bid"), "ask": quote.get("ask"),
                    "last_price": day.get("close"),
                    "volume": day.get("volume", 0),
                    "open_interest": item.get("open_interest", 0),
                    "provider_iv": item.get("implied_volatility"),
                    "last_trade": pd.to_datetime(
                        day.get("last_updated") or quote.get("last_updated"),
                        unit="ns", errors="coerce", utc=True),
                })
            url = payload.get("next_url")
            pages += 1
        if not rows:
            raise ValueError(f"Polygon returned no contracts for {ticker}")
        out = pd.DataFrame(rows)
        out["spot"], out["asof"], out["synthetic"] = spot, asof, False
        LOG.info("Polygon: %d contracts over %d pages", len(out), pages)
        return out


# --------------------------------------------------------------------------- #
# Synthetic (offline)
# --------------------------------------------------------------------------- #
class SyntheticProvider:
    """
    The offline generator. Never licensed for redistribution — not for legal
    reasons but for a more important one: it is not market data at all, and
    serving it to someone who thinks it is would be the worst failure this
    project could have.
    """

    name = "synthetic"

    def __init__(self, seed: int = 42):
        self.seed = seed

    def is_licensed_for_redistribution(self) -> bool:
        return False

    def prices(self, ticker: str, start: str, end: str | None) -> pd.DataFrame:
        from .synthetic import synthetic_prices

        return synthetic_prices(start, end, seed=self.seed)

    def vol_index(self, ticker: str, start: str,
                  end: str | None) -> pd.Series | None:
        return None         # built from returns by the caller when needed

    def option_chain(self, ticker: str, cfg: OptionsConfig, spot: float,
                     asof: pd.Timestamp) -> pd.DataFrame:
        from .synthetic import synthetic_option_chain

        return synthetic_option_chain(spot, asof=asof, r=cfg.risk_free_rate,
                                      seed=self.seed)


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
_REGISTRY: dict[str, Callable[[DataConfig], MarketDataProvider]] = {
    "yfinance": lambda cfg: YFinanceProvider(),
    "polygon": lambda cfg: PolygonProvider(
        cfg.polygon_api_key,
        redistribution_licensed=getattr(cfg, "redistribution_licensed", False)),
    "synthetic": lambda cfg: SyntheticProvider(seed=cfg.synthetic_seed),
}


def register_provider(name: str,
                      factory: Callable[[DataConfig], MarketDataProvider]) -> None:
    """Add a vendor. One call, and every layer above can use it."""
    _REGISTRY[name.lower()] = factory
    LOG.info("Registered market data provider %r", name)


def available_providers() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def get_provider(cfg: DataConfig) -> MarketDataProvider:
    """Build the provider named by the config."""
    key = str(cfg.provider).lower()
    if key not in _REGISTRY:
        raise ValueError(f"Unknown provider {cfg.provider!r}; "
                         f"available: {', '.join(available_providers())}")
    return _REGISTRY[key](cfg)
