"""Provider seam, production guard, and the universe pass."""
from __future__ import annotations

import pandas as pd
import pytest

from volsurface import Config
from volsurface.batch import BatchResult, cross_section, run_universe
from volsurface.config import DataConfig, OptionsConfig
from volsurface.data import providers as PV
from volsurface.history import SurfaceHistory


# --------------------------------------------------------------------------- #
# The provider seam
# --------------------------------------------------------------------------- #
def test_builtin_providers_satisfy_the_protocol():
    for name in ("yfinance", "synthetic"):
        cfg = DataConfig(provider=name)
        assert isinstance(PV.get_provider(cfg), PV.MarketDataProvider)


def test_unknown_provider_names_itself_and_the_alternatives():
    with pytest.raises(ValueError, match="Unknown provider"):
        PV.get_provider(DataConfig(provider="not-a-vendor"))


def test_a_new_vendor_is_one_registration():
    """The whole point of the seam: adding a source must not touch the loaders."""

    class FakeVendor:
        name = "fakevendor"

        def is_licensed_for_redistribution(self):
            return True

        def prices(self, ticker, start, end):
            idx = pd.bdate_range("2026-01-01", periods=10)
            return pd.DataFrame({"Open": 1.0, "High": 1.0, "Low": 1.0,
                                 "Close": 1.0, "Volume": 1.0}, index=idx)

        def option_chain(self, ticker, cfg, spot, asof):
            return pd.DataFrame()

        def vol_index(self, ticker, start, end):
            return None

    PV.register_provider("fakevendor", lambda cfg: FakeVendor())
    try:
        assert "fakevendor" in PV.available_providers()
        got = PV.get_provider(DataConfig(provider="fakevendor"))
        assert got.name == "fakevendor"
        assert len(got.prices("X", "2026-01-01", None)) == 10
    finally:
        PV._REGISTRY.pop("fakevendor", None)


def test_scraped_and_simulated_sources_are_not_redistributable():
    """
    Both answer False, for different reasons: Yahoo's terms forbid it, and the
    synthetic generator is not market data at all.
    """
    assert PV.YFinanceProvider().is_licensed_for_redistribution() is False
    assert PV.SyntheticProvider().is_licensed_for_redistribution() is False


# --------------------------------------------------------------------------- #
# Production config
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("provider", ["yfinance", "synthetic"])
def test_production_refuses_an_unlicensed_provider(provider):
    with pytest.raises(PermissionError, match="not licensed for redistribution"):
        Config.for_production("SPY", provider=provider)


def test_production_turns_off_the_synthetic_fallback():
    cfg = Config.for_production("SPY", provider="yfinance", allow_unlicensed=True)
    assert cfg.data.allow_synthetic_fallback is False
    assert cfg.data.ticker == "SPY"
    assert cfg.verbose is False


def test_research_defaults_are_unchanged():
    """The offline demo path must keep working — production is opt-in."""
    cfg = Config()
    assert cfg.data.allow_synthetic_fallback is True
    assert cfg.data.provider == "yfinance"


# --------------------------------------------------------------------------- #
# Universe pass
# --------------------------------------------------------------------------- #
@pytest.fixture
def synthetic_config(tmp_path):
    cfg = Config()
    cfg.data.provider = "synthetic"
    cfg.data.use_cache = False
    cfg.data.start = "2022-01-01"
    cfg.garch.specs = (("GARCH(1,1)-t", "Garch", 1, 0, 1, "t"),)
    cfg.verbose = False
    cfg.output_dir = tmp_path
    cfg.figure_dir = tmp_path / "figures"
    cfg.report_dir = tmp_path / "reports"
    cfg.analytics.history_dir = tmp_path / "history"
    cfg.analytics.save_snapshot = True
    return cfg


def test_universe_pass_snapshots_every_name(synthetic_config):
    res = run_universe(["AAA", "BBB"], base_config=synthetic_config,
                       pause_seconds=0)
    assert res.summary()["ok"] == 2
    store = SurfaceHistory(synthetic_config.analytics.history_dir)
    for ticker in ("AAA", "BBB"):
        assert store.summary(ticker)["n_snapshots"] == 1


def test_resume_skips_names_already_done_today(synthetic_config):
    run_universe(["AAA"], base_config=synthetic_config, pause_seconds=0)
    again = run_universe(["AAA"], base_config=synthetic_config, pause_seconds=0,
                         resume=True)
    assert again.summary()["skipped"] == 1
    assert again.summary()["ok"] == 0


def test_one_broken_ticker_does_not_end_the_pass(synthetic_config, monkeypatch):
    """A delisted name in the middle of a universe must cost only that name."""
    import volsurface.batch as B

    real = B.run_pipeline

    def selective(cfg, **kw):
        if cfg.data.ticker == "BOOM":
            raise RuntimeError("simulated feed failure")
        return real(cfg, **kw)

    monkeypatch.setattr(B, "run_pipeline", selective)
    res = run_universe(["AAA", "BOOM", "CCC"], base_config=synthetic_config,
                       pause_seconds=0)
    assert res.succeeded == ["AAA", "CCC"]
    assert res.failed == ["BOOM"]
    assert "simulated feed failure" in res.to_frame().query("ticker=='BOOM'")["error"].iloc[0]


def test_synthetic_fallback_is_never_stored_as_market_history(tmp_path, monkeypatch):
    """
    Regression. A mistyped or delisted ticker used to come back "ok" with
    entirely invented numbers, because the research default silently falls back
    to the generator — and those rows landed in the store every future z-score
    is measured against.
    """
    from volsurface.data import prices as PR

    cfg = Config()
    cfg.data.provider = "yfinance"          # not synthetic: fallback is a failure
    cfg.data.use_cache = False
    cfg.data.start = "2022-01-01"
    cfg.garch.specs = (("GARCH(1,1)-t", "Garch", 1, 0, 1, "t"),)
    cfg.verbose = False
    cfg.output_dir = tmp_path
    cfg.figure_dir = tmp_path / "figures"
    cfg.report_dir = tmp_path / "reports"
    cfg.analytics.history_dir = tmp_path / "history"
    cfg.analytics.save_snapshot = True

    def dead_feed(*a, **k):
        raise RuntimeError("no such ticker")

    monkeypatch.setattr(PR, "get_provider",
                        lambda c: type("P", (), {"prices": staticmethod(dead_feed)})())

    res = run_universe(["NOTREAL"], base_config=cfg, pause_seconds=0)
    assert res.failed == ["NOTREAL"]
    assert "synthetic" in res.to_frame()["error"].iloc[0].lower()
    assert SurfaceHistory(cfg.analytics.history_dir).summary("NOTREAL")["n_snapshots"] == 0


def test_cross_section_lines_up_the_universe(synthetic_config):
    run_universe(["AAA", "BBB"], base_config=synthetic_config, pause_seconds=0)
    table = cross_section(SurfaceHistory(synthetic_config.analytics.history_dir),
                          ["AAA", "BBB", "NEVER_RUN"])
    assert list(table["ticker"]) == ["AAA", "BBB"]
    assert {"atm_30d", "rr25_30d", "vrp_vol_points"} <= set(table.columns)


def test_empty_batch_summary_is_safe():
    assert BatchResult().summary()["tickers"] == 0
