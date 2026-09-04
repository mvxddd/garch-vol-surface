"""Portfolio risk, the snapshot store, and the volatility-premium backtest."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from volsurface import backtest as BT
from volsurface import portfolio as PF
from volsurface.config import OptionsConfig, SurfaceConfig, TRADING_DAYS
from volsurface.data.clean import prepare_quotes
from volsurface.data.synthetic import synthetic_option_chain, synthetic_prices
from volsurface.data.prices import compute_returns
from volsurface.history import SurfaceHistory, historical_anomalies, snapshot
from volsurface.models.surface import build_surface

SPOT, ASOF = 450.0, pd.Timestamp("2026-09-02")


@pytest.fixture(scope="module")
def surface():
    chain = synthetic_option_chain(SPOT, asof=ASOF)
    cfg = OptionsConfig()
    quotes, forwards, _ = prepare_quotes(chain, cfg, spot=SPOT, asof=ASOF)
    return build_surface(quotes, forwards, SPOT, SurfaceConfig(), asof=ASOF,
                         risk_free_rate=cfg.risk_free_rate)


# --------------------------------------------------------------------------- #
# Portfolio
# --------------------------------------------------------------------------- #
def test_position_validation():
    with pytest.raises(ValueError):
        PF.Position("call", 1, strike=None, expiry=ASOF)      # no strike
    with pytest.raises(ValueError):
        PF.Position("call", 1, strike=100, expiry=None)       # no expiry
    with pytest.raises(ValueError):
        PF.Position("banana", 1, strike=100, expiry=ASOF)     # not an instrument
    # The underlying needs neither, and defaults to a multiplier of 1.
    assert PF.Position("underlying", 100).multiplier == 1.0


def test_long_call_greeks_have_the_right_signs(surface):
    T = float(surface.maturities[2])
    expiry = ASOF + pd.Timedelta(days=int(round(T * 365)))
    book = PF.Portfolio([PF.Position("call", 1, SPOT, expiry)])
    row = PF.price_portfolio(book, surface).iloc[0]
    assert row["price"] > 0
    assert 0 < row["delta"] < 100          # one contract, 100 multiplier
    assert row["vega"] > 0 and row["gamma"] > 0
    assert row["theta"] < 0                # long options decay


def test_short_straddle_is_short_vol_and_short_gamma(surface):
    T = float(surface.maturities[1])
    expiry = ASOF + pd.Timedelta(days=int(round(T * 365)))
    book = PF.Portfolio([PF.Position("call", -1, SPOT, expiry),
                         PF.Position("put", -1, SPOT, expiry)])
    totals = PF.aggregate_risk(PF.price_portfolio(book, surface))
    assert totals["vega_per_vol_point"] < 0
    assert totals["theta_per_day"] > 0          # collects decay
    assert totals["delta_change_per_1pct"] < 0  # short gamma


def test_vol_shock_moves_value_in_the_direction_of_vega(surface):
    book = PF.example_portfolio(surface)
    base = PF.price_portfolio(book, surface)
    vega = PF.aggregate_risk(base)["vega_per_vol_point"]
    up = PF.price_portfolio(book, surface, vol_shift=0.01)["value"].sum()
    change = up - base["value"].sum()
    # One vol point up should move the book by roughly its vega, same sign.
    assert np.sign(change) == np.sign(vega)
    assert abs(change - vega) < abs(vega) * 0.5


def test_stress_grid_shape_and_zero_centre(surface):
    book = PF.example_portfolio(surface)
    grid = PF.stress_grid(book, surface, spot_shocks=(-0.05, 0.0, 0.05),
                          vol_shocks=(-0.02, 0.0, 0.02))
    assert grid.shape == (3, 3)
    assert grid.loc[0.0, 0.0] == pytest.approx(0.0, abs=1e-6)


def test_sticky_conventions_differ_on_a_large_move(surface):
    """If the two conventions agreed, one of them would not be implemented."""
    book = PF.example_portfolio(surface)
    a = PF.stress_grid(book, surface, spot_shocks=(-0.10,), vol_shocks=(0.0,),
                       sticky="moneyness").iloc[0, 0]
    b = PF.stress_grid(book, surface, spot_shocks=(-0.10,), vol_shocks=(0.0,),
                       sticky="strike").iloc[0, 0]
    assert a != pytest.approx(b, abs=1.0)


def test_vega_ladder_sums_to_total_vega(surface):
    priced = PF.price_portfolio(PF.example_portfolio(surface), surface)
    options_vega = priced[priced["instrument"] != "underlying"]["vega"].sum()
    for by in ("tenor", "strike"):
        assert PF.vega_ladder(priced, by=by)["vega"].sum() == pytest.approx(
            options_vega, rel=1e-9)


def test_portfolio_from_csv(tmp_path, surface):
    expiry = (ASOF + pd.Timedelta(days=30)).date()
    csv = tmp_path / "book.csv"
    csv.write_text("instrument,quantity,strike,expiry\n"
                   f"call,-2,460,{expiry}\n"
                   f"put,1,430,{expiry}\n"
                   "underlying,100,,\n")
    book = PF.Portfolio.from_csv(csv)
    assert len(book) == 3
    assert not PF.price_portfolio(book, surface).empty


# --------------------------------------------------------------------------- #
# History
# --------------------------------------------------------------------------- #
def test_snapshot_has_fixed_tenors(surface):
    row = snapshot(surface, "TEST", garch_vol_30d=0.14)
    for days in (7, 30, 90, 365):
        assert np.isfinite(row[f"atm_{days}d"])
        assert np.isfinite(row[f"rr25_{days}d"])
    assert row["vrp_vol_points"] == pytest.approx((row["atm_30d"] - 0.14) * 100)


def test_history_replaces_rather_than_duplicates(tmp_path, surface):
    store = SurfaceHistory(tmp_path)
    row = snapshot(surface, "TEST", garch_vol_30d=0.14)
    store.append(row, "TEST")
    store.append(row, "TEST")            # same day again
    assert len(store.load("TEST")) == 1


def test_zscores_refuse_a_young_store(tmp_path, surface):
    store = SurfaceHistory(tmp_path)
    store.append(snapshot(surface, "TEST", garch_vol_30d=0.14), "TEST")
    assert store.zscores("TEST").empty


def test_zscores_flag_an_injected_shock(tmp_path, surface):
    """
    Fifty quiet days then one dislocated day: the store must notice. The shock
    is injected into the stored row itself, so the test is about the z-score
    machinery, not about the surface generator.
    """
    store = SurfaceHistory(tmp_path)
    base = snapshot(surface, "TEST", garch_vol_30d=0.14)
    rng = np.random.default_rng(0)
    for i, date in enumerate(pd.bdate_range("2026-01-01", periods=50)):
        row = base.copy()
        row["date"] = date
        row["atm_30d"] = float(base["atm_30d"]) + rng.normal(0, 0.002)
        row["rr25_30d"] = float(base["rr25_30d"]) + rng.normal(0, 0.05)
        store.append(row, "TEST")

    shocked = base.copy()
    shocked["date"] = pd.Timestamp("2026-03-16")
    shocked["rr25_30d"] = float(base["rr25_30d"]) + 1.5       # a huge skew move
    store.append(shocked, "TEST")

    z = store.zscores("TEST", min_observations=30)
    assert not z.empty
    hit = z[z["metric"] == "rr25_30d"].iloc[0]
    assert hit["z_score"] > 5
    assert not historical_anomalies(z, threshold=2.0).empty


def test_history_survives_a_missing_file(tmp_path):
    assert SurfaceHistory(tmp_path).load("NOPE").empty
    assert SurfaceHistory(tmp_path).summary("NOPE")["n_snapshots"] == 0


# --------------------------------------------------------------------------- #
# Backtest
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def vol_and_returns():
    """Implied vol that is systematically above realised — a premium by design."""
    prices = synthetic_prices("2016-01-01", "2026-01-01", seed=11)
    returns = compute_returns(prices)
    realised = np.sqrt(returns.pow(2).rolling(21).mean() * TRADING_DAYS)
    implied = (realised.shift(1) * 1.25).bfill().clip(lower=0.05)
    return implied, returns


def test_selling_a_premium_makes_money(vol_and_returns):
    implied, returns = vol_and_returns
    res = BT.short_straddle_backtest(implied, returns, direction=-1,
                                     costs=BT.CostModel(0.0, 0.0))
    assert res.stats["total_pnl"] > 0
    assert res.stats["hit_rate_pct"] > 55


def test_the_two_directions_are_mirror_images(vol_and_returns):
    implied, returns = vol_and_returns
    free = BT.CostModel(0.0, 0.0)
    short = BT.short_straddle_backtest(implied, returns, direction=-1, costs=free)
    long = BT.short_straddle_backtest(implied, returns, direction=+1, costs=free)
    assert short.stats["total_pnl"] == pytest.approx(-long.stats["total_pnl"],
                                                     rel=1e-9)


def test_costs_reduce_pnl(vol_and_returns):
    implied, returns = vol_and_returns
    free = BT.short_straddle_backtest(implied, returns, costs=BT.CostModel(0.0, 0.0))
    paid = BT.short_straddle_backtest(implied, returns, costs=BT.CostModel(1.0, 5.0))
    assert paid.stats["total_pnl"] < free.stats["total_pnl"]
    assert paid.stats["total_costs"] > 0


def test_non_overlapping_trades_do_not_share_dates(vol_and_returns):
    implied, returns = vol_and_returns
    res = BT.short_straddle_backtest(implied, returns, holding_days=21,
                                     overlap=False)
    gaps = res.trades["entry_date"].diff().dropna().dt.days
    assert (gaps >= 21).all()


def test_backtest_refuses_a_short_sample():
    idx = pd.bdate_range("2026-01-01", periods=30)
    with pytest.raises(ValueError, match="need at least"):
        BT.short_straddle_backtest(pd.Series(0.2, index=idx),
                                   pd.Series(0.001, index=idx), holding_days=21)


def test_signal_backtest_refuses_a_young_history(vol_and_returns):
    _, returns = vol_and_returns
    tiny = pd.DataFrame({"date": pd.bdate_range("2026-01-01", periods=5),
                         "vrp_vol_points": [1.0] * 5, "atm_30d": [0.2] * 5})
    with pytest.raises(ValueError, match="need at least"):
        BT.signal_backtest(tiny, returns)


def test_compare_strategies_returns_both_directions(vol_and_returns):
    implied, returns = vol_and_returns
    table = BT.compare_strategies(implied, returns)
    assert {"always short", "always long"} <= set(table["strategy"])
    assert table["sharpe_annualised"].notna().any()
