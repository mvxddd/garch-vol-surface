"""Cross-cutting helpers: logging, retries, on-disk caching, small numerics."""
from __future__ import annotations

import functools
import hashlib
import json
import logging
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterable, TypeVar

import numpy as np
import pandas as pd

T = TypeVar("T")

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-22s | %(message)s"


def get_logger(name: str = "volsurface", verbose: bool = True) -> logging.Logger:
    """Idempotent logger factory (safe to call repeatedly from notebook cells)."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, "%H:%M:%S"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO if verbose else logging.WARNING)
    logger.propagate = False
    return logger


LOG = get_logger()


@contextmanager
def timer(label: str, logger: logging.Logger | None = None):
    """Context manager that logs wall-clock duration of a pipeline stage."""
    log = logger or LOG
    t0 = time.perf_counter()
    log.info("▶ %s ...", label)
    try:
        yield
    finally:
        log.info("✔ %s finished in %.2fs", label, time.perf_counter() - t0)


def retry(
    attempts: int = 3,
    backoff: float = 1.5,
    exceptions: tuple[type[BaseException], ...] = (Exception,),
    logger: logging.Logger | None = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Exponential-backoff retry decorator.

    Market-data endpoints (Yahoo especially) fail transiently and often; a bare
    call without retries is the single most common cause of a broken notebook.
    """
    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs) -> T:
            log = logger or LOG
            last: BaseException | None = None
            for i in range(1, attempts + 1):
                try:
                    return fn(*args, **kwargs)
                except exceptions as exc:            # noqa: PERF203
                    last = exc
                    if i == attempts:
                        break
                    sleep_for = backoff ** i
                    log.warning(
                        "%s failed (attempt %d/%d): %s — retrying in %.1fs",
                        fn.__name__, i, attempts, exc, sleep_for,
                    )
                    time.sleep(sleep_for)
            raise RuntimeError(
                f"{fn.__name__} failed after {attempts} attempts"
            ) from last
        return wrapper
    return decorator


# --------------------------------------------------------------------------- #
# Lightweight parquet/csv cache
# --------------------------------------------------------------------------- #
def _cache_key(namespace: str, **parts: Any) -> str:
    blob = json.dumps({"ns": namespace, **parts}, sort_keys=True, default=str)
    digest = hashlib.sha1(blob.encode()).hexdigest()[:12]
    return f"{namespace}_{digest}"


def cache_path(cache_dir: Path, namespace: str, **parts: Any) -> Path:
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    return Path(cache_dir) / f"{_cache_key(namespace, **parts)}.parquet"


def read_cache(path: Path, ttl_hours: float) -> pd.DataFrame | None:
    """Return a cached frame if it exists and is younger than `ttl_hours`."""
    p = Path(path)
    if not p.exists():
        return None
    age_h = (time.time() - p.stat().st_mtime) / 3600.0
    if age_h > ttl_hours:
        LOG.info("Cache %s is %.1fh old (ttl %.1fh) — refetching", p.name, age_h, ttl_hours)
        return None
    try:
        return pd.read_parquet(p)
    except Exception as exc:                       # corrupt/partial write
        LOG.warning("Could not read cache %s (%s) — refetching", p.name, exc)
        return None


def write_cache(df: pd.DataFrame, path: Path) -> None:
    """Best-effort cache write; a cache failure must never break the run."""
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
    except Exception as exc:
        LOG.warning("Cache write to %s failed: %s (continuing)", path, exc)


# --------------------------------------------------------------------------- #
# Small numerical helpers
# --------------------------------------------------------------------------- #
def safe_div(num: np.ndarray | float, den: np.ndarray | float,
             fill: float = np.nan) -> np.ndarray:
    """Elementwise division that returns `fill` instead of raising/inf."""
    num_a, den_a = np.asarray(num, dtype=float), np.asarray(den, dtype=float)
    out = np.full(np.broadcast(num_a, den_a).shape, fill, dtype=float)
    ok = np.isfinite(den_a) & (den_a != 0) & np.isfinite(num_a)
    np.divide(num_a, den_a, out=out, where=ok)
    return out


def winsorize(s: pd.Series, lower: float = 0.001, upper: float = 0.999) -> pd.Series:
    """Clip a series at empirical quantiles (used on returns before MLE)."""
    if s.dropna().empty:
        return s
    lo, hi = s.quantile(lower), s.quantile(upper)
    return s.clip(lo, hi)


def zscore(s: pd.Series | np.ndarray, ddof: int = 1) -> np.ndarray:
    arr = np.asarray(s, dtype=float)
    mu, sd = np.nanmean(arr), np.nanstd(arr, ddof=ddof)
    if not np.isfinite(sd) or sd == 0:
        return np.zeros_like(arr)
    return (arr - mu) / sd


def chunked(seq: Iterable[T], size: int) -> Iterable[list[T]]:
    """Yield lists of at most `size` items (used to batch API calls)."""
    buf: list[T] = []
    for item in seq:
        buf.append(item)
        if len(buf) >= size:
            yield buf
            buf = []
    if buf:
        yield buf
