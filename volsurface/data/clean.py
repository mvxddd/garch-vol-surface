"""
Quote cleaning, forward extraction and implied-volatility construction.

This is the least glamorous and most valuable module in the project. A vol
surface is only as good as the quotes underneath it, and a raw Yahoo chain is
full of things that will wreck it: zero bids, 40%-wide spreads, options that
have not traded in a week, strikes listed but never quoted, and deep-ITM
contracts whose price carries no volatility information at all.

The pipeline is a funnel, and every stage is counted and reported:

    raw quotes
      -> structural validity      (finite prices, sane strikes, live expiry)
      -> liquidity filters        (bid, spread, open interest, volume)
      -> per-expiry forward       (put-call parity regression)
      -> OTM selection            (calls above F, puts below F)
      -> IV inversion             (Black-76, vega-identifiable roots only)
      -> smile-level sanity       (IV bounds, min quotes per expiry)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import CALENDAR_DAYS, OptionsConfig
from ..models.black_scholes import (greeks, implied_forward_from_parity,
                                    implied_vol)
from ..utils import get_logger

LOG = get_logger("volsurface.clean")


class QuoteFunnel:
    """Records how many quotes survive each filter — a data-quality audit."""

    def __init__(self) -> None:
        self.stages: list[tuple[str, int]] = []

    def record(self, name: str, n: int) -> None:
        self.stages.append((name, int(n)))

    def to_frame(self) -> pd.DataFrame:
        df = pd.DataFrame(self.stages, columns=["stage", "n_quotes"])
        df["dropped"] = df["n_quotes"].shift(1).sub(df["n_quotes"]).fillna(0).astype(int)
        start = df["n_quotes"].iloc[0] if len(df) else 0
        df["pct_of_raw"] = (df["n_quotes"] / start * 100).round(1) if start else np.nan
        return df

    def log(self) -> None:
        for stage, n in self.stages:
            LOG.info("  %-28s %6d quotes", stage, n)


def _mid_price(df: pd.DataFrame) -> pd.Series:
    """
    Mid where a two-sided market exists, last trade otherwise.

    Never average a zero bid into the mid: a 0 x 0.05 market has a "mid" of
    0.025 that no one will trade, and it produces an IV that is pure fiction.
    """
    bid, ask, last = df["bid"], df["ask"], df["last_price"]
    two_sided = (bid > 0) & (ask > 0) & (ask >= bid)
    mid = np.where(two_sided, (bid + ask) / 2.0, np.nan)
    return pd.Series(np.where(np.isfinite(mid), mid, last), index=df.index)


def clean_option_chain(chain: pd.DataFrame, cfg: OptionsConfig,
                       asof: pd.Timestamp | None = None,
                       funnel: QuoteFunnel | None = None) -> pd.DataFrame:
    """Structural + liquidity filtering. Returns a frame with mid/T/dte added."""
    if chain.empty:
        raise ValueError("Option chain is empty — nothing to clean.")
    funnel = funnel or QuoteFunnel()
    df = chain.copy()
    funnel.record("raw quotes", len(df))

    asof = pd.Timestamp(asof or df["asof"].iloc[0]).normalize()
    df["expiry"] = pd.to_datetime(df["expiry"]).dt.normalize()
    df["dte"] = (df["expiry"] - asof).dt.days
    # Year fraction on a calendar basis: an option decays over weekends too,
    # and every listed expiry convention is calendar-dated.
    df["T"] = df["dte"] / CALENDAR_DAYS

    for col in ("bid", "ask", "last_price", "strike", "volume", "open_interest"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[(df["dte"] >= cfg.min_days_to_expiry) & (df["dte"] <= cfg.max_days_to_expiry)]
    funnel.record("tenor window", len(df))

    df = df[np.isfinite(df["strike"]) & (df["strike"] > 0)]
    df["mid"] = _mid_price(df)
    df = df[np.isfinite(df["mid"]) & (df["mid"] > 0)]
    funnel.record("valid price & strike", len(df))

    # De-duplicate: some feeds return the same contract twice (e.g. a mini and
    # a standard listing sharing a strike). Keep the more liquid one.
    df = (df.sort_values("open_interest", ascending=False)
            .drop_duplicates(subset=["expiry", "strike", "option_type"], keep="first"))
    funnel.record("de-duplicated", len(df))

    two_sided = (df["bid"] > 0) & (df["ask"] > 0)
    rel_spread = np.where(two_sided, (df["ask"] - df["bid"]) / df["mid"], np.nan)
    df["rel_spread"] = rel_spread
    df = df[(df["bid"] >= cfg.min_bid) | (~two_sided & (df["last_price"] > cfg.min_bid))]
    funnel.record(f"bid >= {cfg.min_bid}", len(df))

    df = df[~np.isfinite(df["rel_spread"]) | (df["rel_spread"] <= cfg.max_rel_spread)]
    funnel.record(f"rel spread <= {cfg.max_rel_spread:.0%}", len(df))

    df = df[(df["open_interest"].fillna(0) >= cfg.min_open_interest)
            | (df["volume"].fillna(0) > 0)]
    funnel.record(f"OI >= {cfg.min_open_interest} or traded", len(df))

    df = df[df["volume"].fillna(0) >= cfg.min_volume]
    funnel.record(f"volume >= {cfg.min_volume}", len(df))

    if df.empty:
        raise ValueError(
            "Every quote was filtered out. Loosen OptionsConfig (min_bid, "
            "max_rel_spread, min_open_interest) or check the data source."
        )
    df.attrs["funnel"] = funnel
    return df.reset_index(drop=True)


def compute_forwards(clean: pd.DataFrame, cfg: OptionsConfig,
                     spot: float) -> pd.DataFrame:
    """
    Per-expiry implied forward from put-call parity, with a carry fallback.

    Why bother: for a dividend-paying underlying, using S·e^{(r-q)T} with a
    guessed `q` mislocates the ATM point. Because the smile is steep at the
    money, an error of 0.3% in the forward shows up as ~1 vol point of fake
    skew — larger than most of the effects we are trying to measure. Parity
    reads the forward the market is actually using.
    """
    rows = []
    for (expiry, T), grp in clean.groupby(["expiry", "T"]):
        calls = grp[grp["option_type"] == "call"].set_index("strike")["mid"]
        puts = grp[grp["option_type"] == "put"].set_index("strike")["mid"]
        common = calls.index.intersection(puts.index)

        fwd, r2, source = np.nan, 0.0, "carry"
        if cfg.use_parity_forward and len(common) >= 3:
            fwd, r2 = implied_forward_from_parity(
                common.to_numpy(), calls.loc[common].to_numpy(),
                puts.loc[common].to_numpy(), r=cfg.risk_free_rate, T=float(T),
            )
            source = "parity"

        # Sanity gate: a parity forward more than 15% from the carry forward,
        # or from a poorly-conditioned regression, is rejected.
        q = cfg.dividend_yield if cfg.dividend_yield is not None else 0.0
        carry_fwd = spot * np.exp((cfg.risk_free_rate - q) * float(T))
        if (not np.isfinite(fwd)) or r2 < 0.99 or abs(fwd / carry_fwd - 1.0) > 0.15:
            if source == "parity":
                LOG.warning("Parity forward rejected for %s (F=%.2f, R2=%.4f) — "
                            "falling back to carry forward %.2f",
                            pd.Timestamp(expiry).date(), fwd, r2, carry_fwd)
            fwd, source = carry_fwd, "carry"

        rows.append({
            "expiry": expiry, "T": float(T), "dte": int(round(float(T) * CALENDAR_DAYS)),
            "forward": float(fwd), "forward_source": source, "parity_r2": float(r2),
            "n_parity_strikes": int(len(common)),
            # Implied dividend/borrow: what the market is charging to carry.
            "implied_q": float(cfg.risk_free_rate
                               - np.log(max(fwd, 1e-9) / spot) / max(float(T), 1e-9)),
        })

    fwd_df = pd.DataFrame(rows).sort_values("T").reset_index(drop=True)
    n_parity = int((fwd_df["forward_source"] == "parity").sum())
    LOG.info("Forwards: %d/%d expiries from put-call parity", n_parity, len(fwd_df))
    return fwd_df


def build_iv_quotes(clean: pd.DataFrame, forwards: pd.DataFrame,
                    cfg: OptionsConfig,
                    funnel: QuoteFunnel | None = None) -> pd.DataFrame:
    """
    Invert every surviving quote to an implied volatility and engineer the
    features the surface layer consumes.

    Only **out-of-the-money** options are kept (calls above the forward, puts
    below). ITM options carry the same volatility information but far more of
    their price is intrinsic value, so the same bid/ask in dollars translates
    into a much larger error in vol. Every desk builds surfaces from OTM
    quotes for exactly this reason.

    Engineered features
    -------------------
    k              log-moneyness log(K/F) — the natural x-axis of a smile
    total_variance IV^2 * T — the quantity that must be monotone in T
    vega, delta    from Black-76; vega becomes the calibration weight
    iv_bid/iv_ask  vol of the bid and the ask: the width of the vol market,
                   which is what tells you whether a "signal" is tradeable
    """
    funnel = funnel or clean.attrs.get("funnel") or QuoteFunnel()
    df = clean.merge(forwards[["expiry", "forward"]], on="expiry", how="inner")
    df = df[np.isfinite(df["forward"]) & (df["forward"] > 0)]

    df["moneyness"] = df["strike"] / df["forward"]
    lo, hi = cfg.moneyness_bounds
    df = df[(df["moneyness"] >= lo) & (df["moneyness"] <= hi)]
    funnel.record(f"moneyness in [{lo:.2f}, {hi:.2f}]", len(df))

    is_call = df["option_type"].eq("call").to_numpy()
    otm = np.where(is_call, df["strike"].to_numpy() >= df["forward"].to_numpy(),
                   df["strike"].to_numpy() < df["forward"].to_numpy())
    df = df[otm]
    funnel.record("OTM only", len(df))
    if df.empty:
        raise ValueError("No OTM quotes survived filtering.")

    F = df["forward"].to_numpy()
    K = df["strike"].to_numpy()
    T = df["T"].to_numpy()
    call = df["option_type"].eq("call").to_numpy()
    r = cfg.risk_free_rate

    df["iv"] = implied_vol(df["mid"].to_numpy(), F, K, T, r=r, is_call=call)
    # Vol of the bid and the ask — the bid/ask spread expressed in vol points.
    with np.errstate(invalid="ignore"):
        df["iv_bid"] = implied_vol(df["bid"].to_numpy(), F, K, T, r=r, is_call=call)
        df["iv_ask"] = implied_vol(df["ask"].to_numpy(), F, K, T, r=r, is_call=call)

    n_before = len(df)
    df = df[np.isfinite(df["iv"])]
    funnel.record("IV inverted", len(df))
    if n_before and len(df) < n_before:
        LOG.info("Dropped %d quotes with no identifiable IV", n_before - len(df))

    df = df[(df["iv"] >= cfg.min_iv) & (df["iv"] <= cfg.max_iv)]
    funnel.record(f"IV in [{cfg.min_iv:.0%}, {cfg.max_iv:.0%}]", len(df))

    F, K, T = df["forward"].to_numpy(), df["strike"].to_numpy(), df["T"].to_numpy()
    call = df["option_type"].eq("call").to_numpy()
    g = greeks(F, K, T, df["iv"].to_numpy(), r=r, is_call=call)
    df["vega"] = g["vega"]
    df["delta"] = g["delta"]
    df["forward_delta"] = g["forward_delta"]
    df["gamma"] = g["gamma"]
    df["k"] = np.log(K / F)
    df["total_variance"] = df["iv"] ** 2 * df["T"]
    df["iv_spread"] = df["iv_ask"] - df["iv_bid"]

    # Drop expiries too sparse to support a five-parameter smile.
    counts = df.groupby("expiry")["iv"].transform("size")
    thin = counts < cfg.min_quotes_per_expiry
    if thin.any():
        dropped = df.loc[thin, "expiry"].dt.date.unique()
        LOG.warning("Dropping %d thin expiries (< %d quotes): %s",
                    len(dropped), cfg.min_quotes_per_expiry, list(dropped))
    df = df[~thin]
    funnel.record(f">= {cfg.min_quotes_per_expiry} quotes/expiry", len(df))

    if df.empty:
        raise ValueError("No expiry retained enough quotes to build a smile.")

    out = df.sort_values(["T", "k"]).reset_index(drop=True)
    out.attrs["funnel"] = funnel
    LOG.info("Built %d IV quotes across %d expiries (IV range %.1f%%-%.1f%%)",
             len(out), out["expiry"].nunique(), out["iv"].min() * 100,
             out["iv"].max() * 100)
    return out


def prepare_quotes(chain: pd.DataFrame, cfg: OptionsConfig, spot: float,
                   asof: pd.Timestamp | None = None
                   ) -> tuple[pd.DataFrame, pd.DataFrame, QuoteFunnel]:
    """End-to-end: raw chain -> (iv_quotes, forwards, funnel)."""
    funnel = QuoteFunnel()
    clean = clean_option_chain(chain, cfg, asof=asof, funnel=funnel)
    forwards = compute_forwards(clean, cfg, spot=spot)
    quotes = build_iv_quotes(clean, forwards, cfg, funnel=funnel)
    forwards = forwards[forwards["expiry"].isin(quotes["expiry"].unique())]
    funnel.log()
    return quotes, forwards.reset_index(drop=True), funnel
