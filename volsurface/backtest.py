"""
Backtesting the volatility risk premium.

Read this before trusting a number out of here
----------------------------------------------
There are two honest ways to backtest options with the data this project has,
and one dishonest one.

**Honest, and implemented here:** harvest the premium at the *index* level. Sell
a 30-day at-the-money straddle, delta-hedge it daily, and hold to expiry. Under
the standard result that a continuously delta-hedged option's P&L is driven by
the gap between implied and realised variance, the payoff is

    P&L(long vol) ≈ vega_notional × (RV_realised² − IV_entry²) / (2 × IV_entry)

and the short side is its negative. That needs only an implied-vol index (VIX
and friends) and the underlying's returns — both of which we already load.
Transaction costs are charged explicitly. This is a real, decades-long
backtest.

Getting that sign right is not a detail: with it flipped, selling volatility
appears to *lose* money on 85% of trades, which flatly contradicts the premium
the same data set measures. If a backtest here ever disagrees with the VRP
study, suspect the sign before believing the result.

**Honest, and supported once you have data:** replay accumulated surface
snapshots and trade the screen's own signals. `signal_backtest` does this, but
it needs a history store with months of snapshots. On a young store it refuses
to produce a number rather than producing a meaningless one.

**Dishonest, and deliberately not implemented:** backtesting individual strikes
against today's single option chain. There is no history of option prices in the
free data, so any such "backtest" would be re-pricing the past with today's
surface — a look-ahead so severe it guarantees a beautiful equity curve. If you
want per-strike backtests, you need Polygon's historical chains; the honest
answer until then is that it cannot be done.

Costs are charged on the option (a spread crossed at entry and at expiry) and
on every delta hedge. Defaults are deliberately pessimistic for a retail
account; an institutional book would pay far less.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import TRADING_DAYS
from .utils import get_logger

LOG = get_logger("volsurface.backtest")


@dataclass
class CostModel:
    """
    Transaction costs, in the units they are actually incurred.

    option_spread_vol_points : half the bid/ask of the straddle, expressed in
        vol points — the natural unit, since that is how options are quoted
        between dealers. 0.5 means the round trip costs 1 vol point.
    hedge_cost_bps : cost of each delta hedge, in basis points of the notional
        traded (spread plus commission on the underlying).
    """

    option_spread_vol_points: float = 0.5
    hedge_cost_bps: float = 1.0

    def option_cost(self, vega_notional: float) -> float:
        """Round-trip option cost in currency, for a given vega notional."""
        return abs(vega_notional) * self.option_spread_vol_points * 2.0

    def hedge_cost(self, traded_notional: float) -> float:
        return abs(traded_notional) * self.hedge_cost_bps / 10_000.0


@dataclass
class BacktestResult:
    """Trades, the equity curve, and the statistics that describe them."""

    trades: pd.DataFrame = field(default_factory=pd.DataFrame)
    equity: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    stats: dict[str, float] = field(default_factory=dict)
    spec: dict[str, object] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [f"{k:<26} {v:>12,.3f}" if isinstance(v, float)
                 else f"{k:<26} {v:>12}" for k, v in self.stats.items()]
        return "\n".join(lines)


def _performance_stats(pnl: pd.Series, equity: pd.Series,
                       periods_per_year: float) -> dict[str, float]:
    """Standard performance panel. Sharpe is on the trade series, not daily."""
    pnl = pnl.dropna()
    if pnl.empty:
        return {}
    mean, sd = float(pnl.mean()), float(pnl.std(ddof=1))
    dd = equity - equity.cummax()
    wins = pnl[pnl > 0]
    losses = pnl[pnl <= 0]
    return {
        "n_trades": int(pnl.size),
        "total_pnl": float(pnl.sum()),
        "mean_pnl_per_trade": mean,
        "std_pnl_per_trade": sd,
        "sharpe_annualised": (mean / sd * np.sqrt(periods_per_year)
                              if sd > 0 else np.nan),
        "hit_rate_pct": float((pnl > 0).mean() * 100),
        "best_trade": float(pnl.max()),
        "worst_trade": float(pnl.min()),
        # The number that matters most for a short-vol strategy: how much of the
        # average win a single bad loss wipes out.
        "avg_win": float(wins.mean()) if wins.size else 0.0,
        "avg_loss": float(losses.mean()) if losses.size else 0.0,
        "worst_over_avg_win": (float(abs(pnl.min()) / wins.mean())
                               if wins.size and wins.mean() > 0 else np.nan),
        "max_drawdown": float(dd.min()),
        "final_equity": float(equity.iloc[-1]) if equity.size else 0.0,
    }


def short_straddle_backtest(
    implied_vol: pd.Series,
    returns: pd.Series,
    holding_days: int = 21,
    vega_notional: float = 10_000.0,
    costs: CostModel | None = None,
    overlap: bool = False,
    direction: int = -1,
    signal: pd.Series | None = None,
    signal_threshold: float = 0.0,
) -> BacktestResult:
    """
    Delta-hedged straddle, entered every `holding_days` and held to expiry.

    Parameters
    ----------
    implied_vol : ATM implied vol at entry, decimal (VIX / 100).
    returns     : daily log returns of the underlying.
    direction   : −1 to sell volatility (the usual premium harvest), +1 to buy.
    vega_notional : currency P&L per vol point of the implied/realised gap.
    overlap     : if False (default) trades are non-overlapping, so each is an
        independent observation and the Sharpe is not inflated by the ~√h
        autocorrelation that overlapping windows create.
    signal      : optional series (e.g. a VRP z-score). When given, a trade is
        only opened when `direction * signal >= signal_threshold` — this is how
        you test whether *timing* the premium beats always being short.

    The P&L uses the variance-swap approximation of a delta-hedged straddle,
    which is exact in the continuous-hedging limit and a good approximation for
    daily hedging on an index.
    """
    costs = costs or CostModel()
    iv = pd.Series(implied_vol).dropna().astype(float)
    r = pd.Series(returns).dropna().astype(float)
    idx = iv.index.intersection(r.index)
    if len(idx) < holding_days * 4:
        raise ValueError(
            f"Only {len(idx)} overlapping observations — need at least "
            f"{holding_days * 4} to backtest a {holding_days}-day holding period."
        )
    iv, r = iv.loc[idx].sort_index(), r.loc[idx].sort_index()
    arr, dates = r.to_numpy(), r.index
    n = arr.size

    step = 1 if overlap else holding_days
    rows = []
    for i in range(0, n - holding_days, step):
        entry_iv = float(iv.iloc[i])
        if not np.isfinite(entry_iv) or entry_iv <= 0:
            continue
        if signal is not None:
            sig = signal.reindex([dates[i]]).iloc[0]
            if not np.isfinite(sig) or direction * float(sig) < signal_threshold:
                continue

        window = arr[i + 1: i + 1 + holding_days]
        realised = float(np.sqrt(np.mean(window ** 2) * TRADING_DAYS))

        # Variance-swap approximation of a delta-hedged straddle's P&L.
        # `direction` is the volatility exposure: +1 long vol makes money when
        # realised exceeds implied, and the short side is its mirror image.
        gross = (direction * vega_notional
                 * (realised ** 2 - entry_iv ** 2) / (2 * entry_iv)) * 100

        # Costs: cross the option spread twice, plus a daily delta hedge whose
        # traded notional scales with the underlying's daily move.
        hedge_notional = float(np.abs(window).sum()) * vega_notional * 100
        cost = costs.option_cost(vega_notional) + costs.hedge_cost(hedge_notional)

        rows.append({
            "entry_date": dates[i],
            "exit_date": dates[min(i + holding_days, n - 1)],
            "implied_vol": entry_iv,
            "realized_vol": realised,
            "vrp_vol_points": (entry_iv - realised) * 100,
            "gross_pnl": gross,
            "costs": cost,
            "net_pnl": gross - cost,
        })

    if not rows:
        raise ValueError("No trades were opened — check the signal threshold.")

    trades = pd.DataFrame(rows)
    equity = trades.set_index("exit_date")["net_pnl"].cumsum()
    per_year = TRADING_DAYS / (1 if overlap else holding_days)
    stats = _performance_stats(trades["net_pnl"], equity, per_year)
    stats["gross_pnl"] = float(trades["gross_pnl"].sum())
    stats["total_costs"] = float(trades["costs"].sum())
    stats["cost_share_of_gross_pct"] = (
        abs(stats["total_costs"] / stats["gross_pnl"]) * 100
        if stats["gross_pnl"] else np.nan)

    LOG.info("Straddle backtest: %d trades, net %.0f, Sharpe %.2f, hit rate %.0f%%, "
             "costs ate %.0f%% of gross",
             stats["n_trades"], stats["total_pnl"], stats.get("sharpe_annualised", 0),
             stats["hit_rate_pct"], stats.get("cost_share_of_gross_pct", 0))

    return BacktestResult(trades=trades, equity=equity, stats=stats, spec={
        "strategy": "short straddle" if direction < 0 else "long straddle",
        "holding_days": holding_days, "vega_notional": vega_notional,
        "overlap": overlap, "timed": signal is not None,
        "option_spread_vol_points": costs.option_spread_vol_points,
        "hedge_cost_bps": costs.hedge_cost_bps,
    })


def compare_strategies(implied_vol: pd.Series, returns: pd.Series,
                       vrp_zscore: pd.Series | None = None,
                       holding_days: int = 21, **kwargs) -> pd.DataFrame:
    """
    Always-short against a timed version, so the question "does timing the
    premium add anything over simply always being short?" gets an answer
    rather than an opinion.
    """
    runs = {"always short": dict(direction=-1),
            "always long": dict(direction=+1)}
    if vrp_zscore is not None:
        runs["short when VRP rich (z>0.5)"] = dict(
            direction=-1, signal=vrp_zscore, signal_threshold=0.5)
        runs["short when VRP very rich (z>1)"] = dict(
            direction=-1, signal=vrp_zscore, signal_threshold=1.0)

    rows = []
    for name, spec in runs.items():
        try:
            res = short_straddle_backtest(implied_vol, returns,
                                          holding_days=holding_days,
                                          **{**kwargs, **spec})
            rows.append({"strategy": name, **res.stats})
        except Exception as exc:
            LOG.warning("Strategy %r failed: %s", name, exc)
    out = pd.DataFrame(rows)
    cols = ["strategy", "n_trades", "total_pnl", "sharpe_annualised",
            "hit_rate_pct", "max_drawdown", "worst_trade", "total_costs"]
    return out[[c for c in cols if c in out.columns]]


def signal_backtest(history: pd.DataFrame, returns: pd.Series,
                    metric: str = "vrp_vol_points", threshold: float = 1.0,
                    holding_days: int = 21, vega_notional: float = 10_000.0,
                    costs: CostModel | None = None,
                    min_observations: int = 60) -> BacktestResult:
    """
    Trade a surface metric from the snapshot store: go short volatility when the
    metric is more than `threshold` standard deviations rich, long when cheap.

    Refuses to run on a store with fewer than `min_observations` snapshots.
    A backtest on twenty days of history is not a weak result — it is a
    meaningless one, and returning a number would invite it to be quoted.
    """
    if history is None or len(history) < min_observations:
        raise ValueError(
            f"Snapshot history has {0 if history is None else len(history)} rows; "
            f"need at least {min_observations}. Run the pipeline daily with "
            f"--snapshot to accumulate it."
        )
    if metric not in history.columns:
        raise ValueError(f"{metric!r} is not in the snapshot store")

    hist = history.copy()
    hist["date"] = pd.to_datetime(hist["date"])
    hist = hist.sort_values("date").set_index("date")
    values = pd.to_numeric(hist[metric], errors="coerce")

    # Expanding z-score: only information available at the time is used.
    mu = values.expanding(min_periods=min_observations).mean()
    sd = values.expanding(min_periods=min_observations).std(ddof=1)
    z = ((values - mu) / sd).dropna()
    if z.empty:
        raise ValueError("Not enough history to form an expanding z-score.")

    iv = pd.to_numeric(hist.get("atm_30d"), errors="coerce").reindex(z.index)
    signal = z.where(z.abs() >= threshold, np.nan).dropna()
    if signal.empty:
        raise ValueError(f"No day reached |z| >= {threshold}.")

    # Direction: rich => sell, cheap => buy.
    direction_series = -np.sign(signal)
    costs = costs or CostModel()
    r = pd.Series(returns).dropna().astype(float)
    arr, dates = r.to_numpy(), r.index

    rows = []
    for date, side in direction_series.items():
        if date not in r.index:
            continue
        i = int(r.index.get_loc(date))
        if i + holding_days >= arr.size:
            continue
        entry_iv = float(iv.get(date, np.nan))
        if not np.isfinite(entry_iv) or entry_iv <= 0:
            continue
        window = arr[i + 1: i + 1 + holding_days]
        realised = float(np.sqrt(np.mean(window ** 2) * TRADING_DAYS))
        # `side` is the volatility exposure: −1 short when the metric is rich.
        gross = (side * vega_notional
                 * (realised ** 2 - entry_iv ** 2) / (2 * entry_iv)) * 100
        cost = (costs.option_cost(vega_notional)
                + costs.hedge_cost(float(np.abs(window).sum())
                                   * vega_notional * 100))
        rows.append({"entry_date": date, "side": int(side),
                     "z_score": float(signal[date]), "implied_vol": entry_iv,
                     "realized_vol": realised, "gross_pnl": gross,
                     "costs": cost, "net_pnl": gross - cost})

    if not rows:
        raise ValueError("Signal produced no tradeable dates.")
    trades = pd.DataFrame(rows)
    equity = trades.set_index("entry_date")["net_pnl"].cumsum()
    stats = _performance_stats(trades["net_pnl"], equity,
                               TRADING_DAYS / holding_days)
    return BacktestResult(trades=trades, equity=equity, stats=stats,
                          spec={"strategy": f"{metric} z>{threshold}",
                                "holding_days": holding_days,
                                "vega_notional": vega_notional})
