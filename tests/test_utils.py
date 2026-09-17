"""
Retries, caching and the small numeric helpers.

These are the least glamorous code in the project and among the most load-
bearing: every market-data call goes through the retry decorator, and every
second run of the pipeline goes through the cache. They were also the least
tested — 66% — which is the wrong way round, because a cache that silently
returns stale data or a retry that gives up early fails quietly and looks like
a data problem.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from volsurface.utils import (
    cache_path,
    chunked,
    get_logger,
    read_cache,
    retry,
    safe_div,
    timer,
    winsorize,
    write_cache,
    zscore,
)


# --------------------------------------------------------------------------- #
# Retry
# --------------------------------------------------------------------------- #
def test_retry_returns_on_first_success():
    calls = []

    @retry(attempts=3, backoff=1.0)
    def works():
        calls.append(1)
        return "ok"

    assert works() == "ok"
    assert len(calls) == 1, "a successful call must not be repeated"


def test_retry_recovers_after_transient_failures():
    """The case this exists for: a feed that fails twice and then works."""
    calls = []

    @retry(attempts=3, backoff=1.0)
    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise ConnectionError("transient")
        return "recovered"

    assert flaky() == "recovered"
    assert len(calls) == 3


def test_retry_gives_up_and_keeps_the_cause():
    """The original error must survive, or debugging a feed outage is guesswork."""
    @retry(attempts=2, backoff=1.0)
    def always_fails():
        raise ValueError("underlying reason")

    with pytest.raises(RuntimeError) as excinfo:
        always_fails()
    assert "always_fails" in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, ValueError)
    assert "underlying reason" in str(excinfo.value.__cause__)


def test_retry_only_catches_what_it_was_told_to():
    @retry(attempts=3, backoff=1.0, exceptions=(ConnectionError,))
    def wrong_error():
        raise KeyError("not a network problem")

    with pytest.raises(KeyError):
        wrong_error()


def test_retry_backs_off_between_attempts():
    started = time.perf_counter()
    calls = []

    @retry(attempts=3, backoff=1.2)
    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise ConnectionError("transient")
        return True

    assert flaky()
    # 1.2**1 + 1.2**2 ≈ 2.6s of sleeping; allow slack but require *some* pause,
    # because hammering a rate-limited feed is how you lose access to it.
    assert time.perf_counter() - started > 1.0


def test_retry_preserves_the_function_identity():
    @retry(attempts=1)
    def documented():
        """A docstring worth keeping."""

    assert documented.__name__ == "documented"
    assert "worth keeping" in documented.__doc__


# --------------------------------------------------------------------------- #
# Cache
# --------------------------------------------------------------------------- #
def test_cache_key_depends_on_every_parameter(tmp_path):
    a = cache_path(tmp_path, "chain", ticker="SPY", start="2020-01-01")
    b = cache_path(tmp_path, "chain", ticker="QQQ", start="2020-01-01")
    c = cache_path(tmp_path, "chain", ticker="SPY", start="2021-01-01")
    assert len({a, b, c}) == 3, "different requests must not share a cache file"
    assert a == cache_path(tmp_path, "chain", ticker="SPY", start="2020-01-01")


def test_cache_round_trip(tmp_path):
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    path = cache_path(tmp_path, "prices", ticker="SPY")
    write_cache(df, path)
    pd.testing.assert_frame_equal(read_cache(path, ttl_hours=24), df)


def test_cache_expires(tmp_path):
    """A stale chain served as current is worse than no cache at all."""
    import os

    df = pd.DataFrame({"a": [1]})
    path = cache_path(tmp_path, "chain", ticker="SPY")
    write_cache(df, path)
    old = time.time() - 60 * 60 * 30          # 30 hours ago
    os.utime(path, (old, old))
    assert read_cache(path, ttl_hours=24) is None
    assert read_cache(path, ttl_hours=48) is not None


def test_missing_cache_is_not_an_error(tmp_path):
    assert read_cache(tmp_path / "never-written.parquet", ttl_hours=24) is None


def test_corrupt_cache_refetches_rather_than_raising(tmp_path):
    path = cache_path(tmp_path, "chain", ticker="SPY")
    path.write_bytes(b"this is not parquet")
    assert read_cache(path, ttl_hours=24) is None


def test_cache_write_failure_never_breaks_a_run(tmp_path):
    """Caching is an optimisation; failing to cache must stay silent."""
    unwritable = tmp_path / "no-such-dir" / "\0bad" / "x.parquet"
    write_cache(pd.DataFrame({"a": [1]}), unwritable)   # must not raise


# --------------------------------------------------------------------------- #
# Numerics
# --------------------------------------------------------------------------- #
def test_safe_div_fills_instead_of_raising():
    out = safe_div(np.array([1.0, 2.0, 3.0]), np.array([2.0, 0.0, np.nan]))
    assert out[0] == 0.5
    assert np.isnan(out[1]) and np.isnan(out[2])
    assert not np.isinf(out).any()


def test_safe_div_broadcasts():
    assert np.allclose(safe_div(np.array([2.0, 4.0]), 2.0), [1.0, 2.0])


def test_winsorize_clips_both_tails():
    s = pd.Series([*list(range(100)), 10000, -10000])
    out = winsorize(s, lower=0.01, upper=0.99)
    assert out.max() < 10_000 and out.min() > -10_000
    assert len(out) == len(s), "winsorising clips, it does not drop"


def test_winsorize_on_empty_input():
    empty = pd.Series(dtype=float)
    assert winsorize(empty).empty


def test_zscore_standardises():
    z = zscore(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    assert abs(float(np.mean(z))) < 1e-12
    assert abs(float(np.std(z, ddof=1)) - 1.0) < 1e-12


def test_zscore_on_a_constant_series_is_zero_not_nan():
    """A flat metric has no z-score; returning NaN would poison the screen."""
    assert np.allclose(zscore(np.array([5.0] * 10)), 0.0)


def test_chunked_splits_and_keeps_the_remainder():
    assert list(chunked(range(7), 3)) == [[0, 1, 2], [3, 4, 5], [6]]
    assert list(chunked([], 3)) == []


def test_logger_is_idempotent():
    """Notebook cells call this repeatedly; handlers must not accumulate."""
    first = get_logger("volsurface.test-idempotent")
    before = len(first.handlers)
    for _ in range(5):
        get_logger("volsurface.test-idempotent")
    assert len(first.handlers) == before == 1


def test_timer_logs_the_stage_label():
    """
    Captured through a handler rather than caplog: `get_logger` sets
    propagate=False so notebook output is not duplicated, and caplog only sees
    records that propagate to the root logger.
    """
    import io
    import logging

    log = get_logger("volsurface.test-timer")
    buffer = io.StringIO()
    handler = logging.StreamHandler(buffer)
    log.addHandler(handler)
    try:
        with timer("a labelled stage", log):
            pass
        assert "a labelled stage" in buffer.getvalue()
        assert "finished in" in buffer.getvalue()
    finally:
        log.removeHandler(handler)


def test_timer_logs_the_finish_even_when_the_stage_raises():
    """Otherwise a crash leaves no timing for the stage that caused it."""
    import io
    import logging

    log = get_logger("volsurface.test-timer-fail")
    buffer = io.StringIO()
    handler = logging.StreamHandler(buffer)
    log.addHandler(handler)
    try:
        with pytest.raises(ValueError):
            with timer("failing stage", log):
                raise ValueError("boom")
        assert "failing stage finished" in buffer.getvalue()
    finally:
        log.removeHandler(handler)
