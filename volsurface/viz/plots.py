"""
The chart library.

Each function returns a matplotlib Figure (or a plotly Figure for the two
interactive ones) and optionally saves it. Charts follow a small set of rules
that are applied consistently rather than re-decided per chart:

* **One axis per panel.** Two measures on different scales become two stacked
  panels, never a second y-axis — a dual-axis chart lets the author choose the
  correlation the reader sees.
* **Ordered things get an ordered ramp; identities get categorical slots.**
  Maturity is ordered, so tenors are shaded light-to-dark. Models are
  identities, so they get fixed hues.
* **Legend plus direct labels for >= 2 series**, because part of the palette sits
  below 3:1 contrast on the light surface and colour must never be the only
  channel carrying meaning.
* **Recessive chrome**: hairline horizontal grid, no top/right spines, muted
  ticks. The data is the only thing with contrast.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from ..config import CALENDAR_DAYS
from ..utils import get_logger
from . import theme as TH

LOG = get_logger("volsurface.viz")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _save(fig, path: str | Path | None):
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        LOG.info("Saved figure -> %s", path)
    return fig


def _titles(ax, title: str, subtitle: str = "", ylabel: str = "", xlabel: str = ""):
    """Left-aligned title with a secondary-ink subtitle — the house pattern."""
    th = TH.active()
    ax.set_title(title, loc="left", color=th.ink, fontsize=12, fontweight="bold", pad=
                 18 if subtitle else 10)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=9.5,
                color=th.ink_secondary, va="bottom", ha="left")
    if ylabel:
        ax.set_ylabel(ylabel, color=th.ink_secondary)
    if xlabel:
        ax.set_xlabel(xlabel, color=th.ink_secondary)


def _label_line_end(ax, x, y, text: str, color: str, dx: float = 0.01,
                    fontsize: float = 9):
    """Direct label at the right end of a line (the second identity channel)."""
    if len(x) == 0:
        return
    ax.annotate(text, xy=(x[-1], y[-1]), xytext=(4, 0), textcoords="offset points",
                color=color, fontsize=fontsize, fontweight="bold",
                va="center", ha="left", clip_on=False)


def _pct(ax, axis: str = "y", decimals: int = 0):
    from matplotlib.ticker import FuncFormatter

    fmt = FuncFormatter(lambda v, _: f"{v*100:.{decimals}f}%")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


# --------------------------------------------------------------------------- #
# 1. The surface itself
# --------------------------------------------------------------------------- #
def plot_surface_3d(surface, n_k: int = 61, n_t: int = 41,
                    k_bounds: tuple[float, float] = (-0.30, 0.20),
                    show_quotes: bool = True, path: str | Path | None = None):
    """
    Interactive 3-D implied-volatility surface (plotly).

    Continuous magnitude (IV) is encoded with the single-hue sequential ramp;
    the actual market quotes are overlaid as points so a reader can see where
    the surface is supported by data and where it is interpolation. That
    overlay is the difference between an honest surface plot and a pretty one.
    """
    import plotly.graph_objects as go

    k_grid, t_grid, iv = surface.grid(k_bounds=k_bounds, n_k=n_k, n_t=n_t)
    days = t_grid * CALENDAR_DAYS
    th = TH.active()

    fig = go.Figure()
    fig.add_trace(go.Surface(
        x=k_grid, y=days, z=iv * 100,
        colorscale=TH.plotly_colorscale(),
        colorbar={"title": {"text": "IV (%)", "side": "right"}, "thickness": 14,
                  "len": 0.62, "outlinewidth": 0},
        contours={"z": {"show": True, "usecolormap": True, "project_z": False,
                        "width": 1}},
        hovertemplate=("log-moneyness %{x:.3f}<br>%{y:.0f} days"
                       "<br><b>IV %{z:.2f}%</b><extra></extra>"),
        name="fitted surface", showscale=True, opacity=0.96,
    ))

    if show_quotes and surface.quotes is not None and not surface.quotes.empty:
        q = surface.quotes
        fig.add_trace(go.Scatter3d(
            x=q["k"], y=q["T"] * CALENDAR_DAYS, z=q["iv"] * 100,
            mode="markers",
            marker={"size": 2.2, "color": th.ink_secondary, "opacity": 0.55},
            name="market quotes",
            hovertemplate=("K %{customdata[0]:.1f} · %{customdata[1]}"
                           "<br>IV %{z:.2f}%<extra></extra>"),
            customdata=np.stack([q["strike"], q["option_type"]], axis=-1),
        ))

    asof = pd.Timestamp(surface.asof).date()
    layout = TH.plotly_layout(
        f"Implied volatility surface · {asof}",
        f"{len(surface.slices)} calibrated expiries ({surface.method.upper()}) · "
        f"forward-ATM at log-moneyness 0 · drag to rotate",
    )
    layout["scene"] = {
        "xaxis": {"title": "log-moneyness  log(K/F)", "gridcolor": th.grid,
                  "backgroundcolor": th.surface, "showbackground": True,
                  "zerolinecolor": th.axis},
        "yaxis": {"title": "days to expiry", "gridcolor": th.grid,
                  "backgroundcolor": th.surface, "showbackground": True,
                  "zerolinecolor": th.axis},
        "zaxis": {"title": "implied vol (%)", "gridcolor": th.grid,
                  "backgroundcolor": th.surface, "showbackground": True,
                  "zerolinecolor": th.axis},
        "camera": {"eye": {"x": 1.65, "y": -1.5, "z": 0.75}},
        "aspectratio": {"x": 1.1, "y": 1.0, "z": 0.65},
    }
    layout["legend"] = {"orientation": "h", "y": -0.02, "x": 0.02}
    fig.update_layout(**layout)

    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        p = str(path)
        (fig.write_html(p, include_plotlyjs="cdn") if p.endswith(".html")
         else fig.write_image(p, scale=2))
        LOG.info("Saved figure -> %s", p)
    return fig


def plot_surface_heatmap(surface, n_k: int = 121, n_t: int = 81,
                         k_bounds: tuple[float, float] = (-0.30, 0.20),
                         path: str | Path | None = None):
    """
    The same surface as a heatmap — usually the *more readable* of the two.

    A 3-D surface is what people expect; a heatmap is what lets you actually
    read a value off the chart and compare two points. Ship both.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    th = TH.active()
    k_grid, t_grid, iv = surface.grid(k_bounds=k_bounds, n_k=n_k, n_t=n_t)
    days = t_grid * CALENDAR_DAYS
    cmap = LinearSegmentedColormap.from_list("seq_blue", list(th.sequential))

    # Determine the strike range actually quoted at each tenor. A short-dated
    # option is simply not listed 30% out of the money, and painting the
    # extrapolated corner in the same colours as fitted data is the single most
    # misleading thing a surface plot can do — so those corners get veiled.
    k_lo = k_hi = None
    if surface.quotes is not None and not surface.quotes.empty:
        span = surface.quotes.groupby("T")["k"].agg(["min", "max"])
        t_nodes = span.index.to_numpy()
        # Clip to the plotted window: a one-year option may be quoted wider
        # than the grid, and an unclipped boundary line would push the axes out.
        k_lo = np.clip(np.interp(t_grid, t_nodes, span["min"].to_numpy()),
                       k_grid.min(), k_grid.max())
        k_hi = np.clip(np.interp(t_grid, t_nodes, span["max"].to_numpy()),
                       k_grid.min(), k_grid.max())

    # Robust colour limits computed on the *supported* region only: one
    # 45%-vol front-month wing must not flatten the contrast everywhere else.
    inside = np.ones_like(iv, dtype=bool) if k_lo is None else (
        (k_grid[None, :] >= k_lo[:, None]) & (k_grid[None, :] <= k_hi[:, None]))
    vmin, vmax = np.nanpercentile(iv[inside] * 100, [1, 99])

    fig, ax = plt.subplots(figsize=(10, 5.6))
    mesh = ax.pcolormesh(k_grid, days, iv * 100, cmap=cmap, shading="gouraud",
                         vmin=vmin, vmax=vmax)
    cs = ax.contour(k_grid, days, iv * 100, levels=10,
                    colors=[th.surface], linewidths=0.7, alpha=0.75)
    ax.clabel(cs, inline=True, fontsize=7.5, fmt="%.0f")

    # Veil (rather than clip) the unsupported corners: smooth edges, and the
    # extrapolated surface stays faintly visible as what it is.
    if k_lo is not None:
        for lo_edge, hi_edge in ((np.full_like(k_lo, k_grid.min()), k_lo),
                                 (k_hi, np.full_like(k_hi, k_grid.max()))):
            ax.fill_betweenx(days, lo_edge, hi_edge, color=th.page, alpha=0.92,
                             lw=0, zorder=3)
        ax.plot(k_lo, days, color=th.axis, lw=1.0, zorder=4)
        ax.plot(k_hi, days, color=th.axis, lw=1.0, zorder=4)

    if surface.quotes is not None and not surface.quotes.empty:
        q = surface.quotes
        ax.scatter(q["k"], q["T"] * CALENDAR_DAYS, s=5, c=th.surface,
                   edgecolors="none", alpha=0.55, zorder=5, label="market quotes")
        ax.legend(loc="upper right", labelcolor=th.ink_secondary)

    ax.axvline(0.0, color=th.surface, lw=1.0, ls="--", alpha=0.8, zorder=6)
    ax.text(0.002, days.max(), " forward ATM", color=th.surface, fontsize=8.5,
            va="top", ha="left", zorder=6)
    ax.set_xlim(k_grid.min(), k_grid.max())
    ax.set_ylim(days.min(), days.max())
    cbar = fig.colorbar(mesh, ax=ax, pad=0.015)
    cbar.set_label("implied vol (%)", color=th.ink_secondary, fontsize=9.5)
    cbar.outline.set_visible(False)
    ax.grid(False)
    _titles(ax, "Implied volatility surface",
            f"{pd.Timestamp(surface.asof).date()} · dots are the quotes the surface "
            f"was fitted to · blank corners are strikes the market does not list "
            f"at that tenor",
            ylabel="days to expiry", xlabel="log-moneyness  log(K/F)")
    return _save(fig, path)


def plot_smile_grid(surface, max_panels: int = 12, path: str | Path | None = None):
    """
    Small multiples: one panel per expiry, market quotes against the fitted
    smile, with the market's own bid/ask band in vol points.

    Small multiples rather than 8 overlaid lines because past ~5 series no
    palette keeps them apart honestly — and because the question a reader
    actually has ("does the fit go through the quotes?") is per-expiry.
    """
    import matplotlib.pyplot as plt

    th = TH.active()
    q = surface.quotes
    if q is None or q.empty:
        raise ValueError("Surface carries no quotes to plot.")

    slices = surface.slices[:max_panels]
    n = len(slices)
    ncols = min(4, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 2.7 * nrows),
                             sharex=False, sharey=False, squeeze=False)

    for i, sl in enumerate(slices):
        ax = axes[i // ncols][i % ncols]
        sub = q[np.isclose(q["T"], sl.T)]
        k_fit = np.linspace(sub["k"].min() - 0.02, sub["k"].max() + 0.02, 200)
        iv_fit = np.asarray(sl.implied_vol(k_fit)) * 100

        if {"iv_bid", "iv_ask"}.issubset(sub.columns):
            lo = np.minimum(sub["iv_bid"], sub["iv_ask"]) * 100
            hi = np.maximum(sub["iv_bid"], sub["iv_ask"]) * 100
            ok = np.isfinite(lo) & np.isfinite(hi)
            ax.vlines(sub["k"][ok], lo[ok], hi[ok], color=th.axis, lw=1.6,
                      alpha=0.9, zorder=1)
        ax.scatter(sub["k"], sub["iv"] * 100, s=11, color=th.ink_secondary,
                   zorder=2, label="market mid")
        ax.plot(k_fit, iv_fit, color=th.categorical[0], lw=2.0, zorder=3,
                label="fitted smile")
        ax.axvline(0, color=th.axis, lw=0.8, ls=":")

        rmse = getattr(sl, "rmse_vol", np.nan)
        ax.set_title(f"{int(round(sl.T * CALENDAR_DAYS))}d   "
                     f"ATM {float(sl.implied_vol(0.0))*100:.1f}%",
                     loc="left", fontsize=10, color=th.ink, fontweight="bold")
        ax.text(0.98, 0.94, f"fit RMSE {rmse*100:.2f} vp\n{len(sub)} quotes",
                transform=ax.transAxes, ha="right", va="top", fontsize=7.6,
                color=th.ink_muted)
        ax.tick_params(labelsize=8)
        if i % ncols == 0:
            ax.set_ylabel("IV (%)", color=th.ink_secondary, fontsize=9)
        if i // ncols == nrows - 1:
            ax.set_xlabel("log(K/F)", color=th.ink_secondary, fontsize=9)

    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles + [plt.Line2D([], [], color=th.axis, lw=1.6)],
               labels + ["bid/ask in vol"], loc="upper right",
               bbox_to_anchor=(0.995, 1.0), ncol=3, frameon=False,
               fontsize=9, labelcolor=th.ink_secondary)
    fig.suptitle("Volatility smile by expiry: market quotes vs calibrated fit",
                 x=0.008, y=1.015, ha="left", fontsize=13, fontweight="bold",
                 color=th.ink)
    fig.tight_layout()
    return _save(fig, path)


def plot_smile_overlay(surface, n_tenors: int = 5, path: str | Path | None = None):
    """
    Selected tenors overlaid, shaded light (short) to dark (long).

    Capped at five lines: that is the number of steps the ordinal ramp keeps
    visibly distinct. Each line is also directly labelled with its tenor.
    """
    import matplotlib.pyplot as plt

    th = TH.active()
    mats = surface.maturities
    pick = (mats if len(mats) <= n_tenors
            else mats[np.linspace(0, len(mats) - 1, n_tenors).round().astype(int)])
    colors = TH.tenor_colors(len(pick))

    fig, ax = plt.subplots(figsize=(9, 5.2))
    k = np.linspace(-0.30, 0.20, 250)
    for T, c in zip(pick, colors):
        iv = np.asarray(surface.iv(k, float(T))) * 100
        ax.plot(k, iv, color=c, lw=2.0)
        _label_line_end(ax, k, iv, f"{int(round(float(T)*CALENDAR_DAYS))}d", c)

    ax.axvline(0, color=th.axis, lw=0.9, ls=":")
    ax.text(0.004, ax.get_ylim()[1], "forward ATM", color=th.ink_muted,
            fontsize=8.5, va="top")
    _titles(ax, "Volatility smile across the term structure",
            "shaded light to dark by maturity · the downward slope to the left "
            "is the equity crash skew",
            ylabel="implied vol (%)", xlabel="log-moneyness  log(K/F)")
    ax.set_xlim(k.min() - 0.01, k.max() + 0.045)        # room for direct labels
    fig.tight_layout()
    return _save(fig, path)


# --------------------------------------------------------------------------- #
# 2. Term structure and skew
# --------------------------------------------------------------------------- #
def plot_term_structure(term_df: pd.DataFrame, garch_ts: pd.DataFrame | None = None,
                        path: str | Path | None = None):
    """
    ATM implied vol, implied forward vol, and (optionally) the GARCH forecast
    term structure — three measures in the same unit, so one axis.

    The gap between the implied and GARCH lines *is* the volatility risk
    premium; drawing them on the same axis is the point of the chart.
    """
    import matplotlib.pyplot as plt

    th = TH.active()
    fig, ax = plt.subplots(figsize=(9, 5.2))
    x = term_df["days"].to_numpy()

    ax.plot(x, term_df["atm_iv"] * 100, color=th.categorical[0], lw=2.2,
            marker="o", ms=5, label="ATM implied vol")
    _label_line_end(ax, x, term_df["atm_iv"].to_numpy() * 100, "implied",
                    th.categorical[0])

    if "fwd_vol" in term_df:
        fv = term_df["fwd_vol"].to_numpy() * 100
        ok = np.isfinite(fv)
        ax.plot(x[ok], fv[ok], color=th.categorical[1], lw=1.8, ls="--",
                marker="s", ms=4, label="implied forward vol (between tenors)")
        _label_line_end(ax, x[ok], fv[ok], "forward", th.categorical[1])

    if garch_ts is not None and not garch_ts.empty:
        gx = garch_ts["horizon_days"].to_numpy() / 252 * CALENDAR_DAYS
        gy = garch_ts["garch_vol_ann"].to_numpy() * 100
        ax.plot(gx, gy, color=th.categorical[2], lw=2.0, marker="^", ms=5,
                label="GARCH forecast vol")
        _label_line_end(ax, gx, gy, "GARCH", th.categorical[2])
        # Shade the premium the market is charging over the model.
        common = np.linspace(max(x.min(), gx.min()), min(x.max(), gx.max()), 100)
        iv_i = np.interp(common, x, term_df["atm_iv"].to_numpy() * 100)
        g_i = np.interp(common, gx, gy)
        ax.fill_between(common, g_i, iv_i, where=iv_i >= g_i, alpha=0.10,
                        color=th.categorical[0], lw=0)
        ax.fill_between(common, g_i, iv_i, where=iv_i < g_i, alpha=0.10,
                        color=th.diverging[0], lw=0)

    ax.set_xscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(v)}d" for v in x], fontsize=9)
    ax.minorticks_off()
    ax.legend(loc="upper left", ncol=1)
    ax.set_xlim(right=ax.get_xlim()[1] * 1.28)          # room for direct labels
    _titles(ax, "Volatility term structure: market vs model",
            "shaded band = volatility risk premium (implied above model)",
            ylabel="annualised vol (%)", xlabel="tenor")
    fig.tight_layout()
    return _save(fig, path)


def plot_skew_term(skew_df: pd.DataFrame, path: str | Path | None = None):
    """
    25-delta risk reversal and butterfly by tenor — both in vol points, one axis.

    Risk reversal is the price of skew (puts over calls); butterfly is the
    price of convexity. Together they are how a smile is quoted on a desk.
    """
    import matplotlib.pyplot as plt

    th = TH.active()
    fig, axes = plt.subplots(2, 1, figsize=(9, 6.4), sharex=True,
                             gridspec_kw={"height_ratios": [1.35, 1]})
    x = skew_df["days"].to_numpy()

    ax = axes[0]
    ax.plot(x, skew_df["risk_reversal"], color=th.categorical[0], lw=2.2,
            marker="o", ms=5, label="25Δ risk reversal (put − call)")
    _label_line_end(ax, x, skew_df["risk_reversal"].to_numpy(), "risk reversal",
                    th.categorical[0])
    ax.plot(x, skew_df["butterfly"], color=th.categorical[1], lw=2.0, ls="--",
            marker="s", ms=4.5, label="25Δ butterfly (wings − ATM)")
    _label_line_end(ax, x, skew_df["butterfly"].to_numpy(), "butterfly",
                    th.categorical[1])
    ax.axhline(0, color=th.axis, lw=0.9)
    ax.legend(loc="center left", ncol=1)
    _titles(ax, "Skew and convexity by tenor",
            "positive risk reversal = puts bid over calls, the equity crash skew",
            ylabel="vol points")

    ax = axes[1]
    ax.plot(x, skew_df["atm_skew"], color=th.categorical[2], lw=2.2, marker="D",
            ms=4.5)
    _label_line_end(ax, x, skew_df["atm_skew"].to_numpy(), "ATM slope",
                    th.categorical[2])
    ax.axhline(0, color=th.axis, lw=0.9)
    _titles(ax, "ATM skew  ∂IV/∂log(K/F)",
            "flattens with maturity — the standard 1/√T decay",
            ylabel="per unit log-moneyness", xlabel="days to expiry")
    for a in axes:
        a.set_xscale("log")
        a.set_xticks(x)
        a.set_xticklabels([f"{int(v)}d" for v in x], fontsize=9)
        a.minorticks_off()
        a.set_xlim(x.min() * 0.88, x.max() * 1.30)      # room for direct labels
    fig.tight_layout()
    return _save(fig, path)


def plot_risk_neutral_density(surface, n_tenors: int = 5,
                              path: str | Path | None = None):
    """
    Risk-neutral densities implied by the calibrated smiles.

    The single fastest visual check that a surface is usable: any dip below
    zero is a butterfly arbitrage. Also the clearest picture of what the skew
    *means* — a left tail far fatter than lognormal.
    """
    import matplotlib.pyplot as plt

    th = TH.active()
    slices = [s for s in surface.slices if hasattr(s, "risk_neutral_density")]
    if not slices:
        raise ValueError("Density plot requires SVI slices.")
    pick = (slices if len(slices) <= n_tenors else
            [slices[i] for i in np.linspace(0, len(slices) - 1, n_tenors)
             .round().astype(int)])
    colors = TH.tenor_colors(len(pick))

    fig, ax = plt.subplots(figsize=(9, 5.0))
    k = np.linspace(-0.55, 0.40, 500)
    for sl, c in zip(pick, colors):
        d = np.asarray(sl.risk_neutral_density(k))
        ax.plot(k, d, color=c, lw=2.0)
        _label_line_end(ax, k, d, f"{int(round(sl.T * CALENDAR_DAYS))}d", c)
    ax.axhline(0, color=th.axis, lw=1.0)
    ax.axvline(0, color=th.axis, lw=0.9, ls=":")
    _titles(ax, "Implied risk-neutral density by tenor",
            "any excursion below zero would be a butterfly arbitrage · "
            "the fat left tail is the skew in probability space",
            ylabel="density", xlabel="log-moneyness  log(K/F)")
    ax.margins(x=0.05)
    fig.tight_layout()
    return _save(fig, path)


# --------------------------------------------------------------------------- #
# 3. GARCH
# --------------------------------------------------------------------------- #
def plot_conditional_vol(returns: pd.Series, fit, realized_window: int = 21,
                         oos_start=None, path: str | Path | None = None):
    """
    Two stacked panels: daily returns, and the model's conditional volatility
    against trailing realised vol.

    Stacked panels rather than a twin axis — returns (%) and volatility
    (annualised %) are different measures, and a dual axis would let the
    drawing imply a fit that was never tested.
    """
    import matplotlib.pyplot as plt

    from ..models.garch import realized_vol

    th = TH.active()
    cond = (pd.Series(np.asarray(fit.result.conditional_volatility),
                      index=pd.Series(returns).dropna().index[-len(
                          fit.result.conditional_volatility):])
            / fit.scale * np.sqrt(252))
    rv = realized_vol(returns, realized_window)

    fig, axes = plt.subplots(2, 1, figsize=(11, 6.6), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1.5]})

    ax = axes[0]
    ax.axhline(0, color=th.axis, lw=0.8)
    ax.plot(returns.index, returns * 100, color=th.ink_muted, lw=0.5, alpha=0.9)
    _titles(ax, "Daily log returns",
            "volatility clustering — calm and violent periods arrive in runs",
            ylabel="return (%)")
    ax.grid(axis="y")

    ax = axes[1]
    ax.plot(rv.index, rv * 100, color=th.ink_muted, lw=1.2,
            label=f"realised vol ({realized_window}d trailing)")
    ax.plot(cond.index, cond * 100, color=th.categorical[0], lw=1.6,
            label=f"{fit.name} conditional vol")
    ax.axhline(fit.long_run_vol * 100, color=th.categorical[1], lw=1.4, ls="--",
               label=f"long-run vol {fit.long_run_vol*100:.1f}%")
    if oos_start is not None:
        ax.axvline(oos_start, color=th.axis, lw=1.2, ls=":")
        ax.text(oos_start, ax.get_ylim()[1], "  out-of-sample from here",
                fontsize=8.5, color=th.ink_muted, va="top", ha="left")
    ax.legend(loc="upper left", ncol=3)
    _titles(ax, "Conditional volatility vs realised",
            f"persistence {fit.persistence:.3f} — shocks decay with a half-life of "
            f"{np.log(0.5)/np.log(max(fit.persistence, 1e-9)):.0f} days",
            ylabel="annualised vol (%)")
    fig.tight_layout()
    return _save(fig, path)


def plot_forecast_vs_realized(walk_forward: pd.DataFrame, model: str | None = None,
                              path: str | Path | None = None):
    """
    Out-of-sample forecast against subsequently realised vol, one panel per
    horizon — the honest picture of what the model can and cannot do.
    """
    import matplotlib.pyplot as plt

    th = TH.active()
    df = walk_forward.copy()
    if model:
        df = df[df["model"] == model]
    if df.empty:
        raise ValueError("No walk-forward rows to plot.")
    model = model or df["model"].iloc[0]
    horizons = sorted(df["horizon"].unique())

    fig, axes = plt.subplots(len(horizons), 1, figsize=(11, 2.5 * len(horizons)),
                             sharex=True, squeeze=False)
    for i, h in enumerate(horizons):
        ax = axes[i][0]
        sub = df[df["horizon"] == h].sort_values("date")
        ax.plot(sub["date"], sub["realized_vol"] * 100, color=th.ink_muted,
                lw=1.3, label="realised (next %dd)" % h)
        ax.plot(sub["date"], sub["forecast_vol"] * 100, color=th.categorical[0],
                lw=1.7, label="forecast")
        corr = sub[["forecast_vol", "realized_vol"]].corr().iloc[0, 1]
        ax.text(0.995, 0.93, f"corr {corr:.2f}", transform=ax.transAxes,
                ha="right", va="top", fontsize=9, color=th.ink_muted)
        _titles(ax, f"{h}-day horizon", ylabel="ann. vol (%)")
        if i == 0:
            ax.legend(loc="upper left", ncol=2)
    fig.suptitle(f"Out-of-sample volatility forecast vs realised · {model}",
                 x=0.008, y=1.005, ha="left", fontsize=13, fontweight="bold",
                 color=th.ink)
    fig.tight_layout()
    return _save(fig, path)


def plot_model_scorecard(eval_df: pd.DataFrame, metric: str = "qlike",
                         path: str | Path | None = None):
    """
    Model ranking per horizon, as a **gap to the best model** rather than the
    raw level.

    QLIKE levels are large, negative and nearly identical across models
    (-3.35 vs -3.31): a bar chart of the level from a zero baseline shows six
    indistinguishable stubs and hides the entire result. What a reader wants is
    "how much worse is each model than the winner", so that is what is plotted;
    the raw value is printed on each bar so nothing is lost.

    Model names sit on the axis, so identity never depends on colour — the
    colours are consistent across panels purely to help the eye track a model
    from one horizon to the next.
    """
    import matplotlib.pyplot as plt

    th = TH.active()
    lower_better = metric in {"qlike", "rmse", "mae", "mape"}
    piv = eval_df.pivot_table(index="model", columns="horizon", values=metric)
    horizons = list(piv.columns)
    # Fixed colour per model, assigned once by overall rank so a model keeps its
    # hue in every panel.
    rank = piv.mean(axis=1).sort_values(ascending=lower_better).index
    color_of = {m: th.categorical[i % len(th.categorical)] for i, m in enumerate(rank)}

    fig, axes = plt.subplots(1, len(horizons), figsize=(3.5 * len(horizons), 4.4),
                             squeeze=False)
    for j, h in enumerate(horizons):
        ax = axes[0][j]
        col = piv[h].dropna()
        best = col.min() if lower_better else col.max()
        gap = (col - best) if lower_better else (best - col)
        gap = gap.sort_values(ascending=False)          # worst at top, best at bottom

        vals = gap.to_numpy()
        bars = ax.barh(range(len(gap)), vals, height=0.62,
                       color=[color_of[m] for m in gap.index],
                       edgecolor=th.surface, linewidth=1.2)
        ax.bar_label(bars, labels=[f" {col[m]:.3f}" for m in gap.index],
                     padding=2, fontsize=8, color=th.ink_secondary)

        # QLIKE gaps span orders of magnitude when a benchmark badly
        # under-forecasts (the loss contains RV^2/sigma^2, so a handful of days
        # dominate its mean). A linear axis would then render five of six bars
        # as invisible stubs, so switch to symlog and say so on the axis.
        pos = vals[vals > 0]
        log_scale = bool(pos.size and pos.max() / max(pos.min(), 1e-12) > 100)
        if log_scale:
            ax.set_xscale("symlog", linthresh=max(pos.min(), 1e-4))
        ax.set_xlabel("gap to best" + (" (log scale)" if log_scale else ""),
                      fontsize=8.5, color=th.ink_muted)
        ax.set_yticks(range(len(gap)))
        ax.set_yticklabels(gap.index, fontsize=8.5)
        ax.grid(axis="x")
        ax.set_axisbelow(True)
        ax.margins(x=0.30)
        ax.set_title(f"{int(h)}-day horizon", loc="left", fontsize=10.5,
                     color=th.ink, fontweight="bold")

    fig.suptitle(f"Out-of-sample {metric.upper()}: gap to the best model",
                 x=0.008, y=1.105, ha="left", fontsize=13, fontweight="bold",
                 color=th.ink)
    fig.text(0.008, 1.045, "shorter bar = better · raw " + metric.upper() +
             " printed on each bar · naive benchmarks included so the GARCH "
             "models have to earn their place", ha="left", fontsize=9.5,
             color=th.ink_secondary)
    fig.tight_layout()
    return _save(fig, path)


# --------------------------------------------------------------------------- #
# 4. Volatility risk premium
# --------------------------------------------------------------------------- #
def plot_vrp_term(vrp_df: pd.DataFrame, path: str | Path | None = None):
    """
    VRP by horizon as diverging bars: blue where implied exceeds the model
    (options rich, the usual state), red where it does not.
    """
    import matplotlib.pyplot as plt

    th = TH.active()
    neg, mid, pos = th.diverging
    fig, ax = plt.subplots(figsize=(9, 4.6))
    x = np.arange(len(vrp_df))
    vals = vrp_df["vrp_vol_points"].to_numpy()
    colors = [pos if v >= 0 else neg for v in vals]

    bars = ax.bar(x, vals, 0.55, color=colors, edgecolor=th.surface, linewidth=1.2)
    ax.bar_label(bars, fmt="%+.2f", fontsize=9, padding=3, color=th.ink_secondary)
    ax.axhline(0, color=th.axis, lw=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(d)}d\nimplied {iv*100:.1f}% / model {g*100:.1f}%"
                        for d, iv, g in zip(vrp_df["horizon_days"],
                                            vrp_df["atm_iv"], vrp_df["garch_vol"])],
                       fontsize=8.5)
    ax.grid(axis="y")
    _titles(ax, "Volatility risk premium: implied minus GARCH forecast",
            "positive = the market charges more for volatility than the model "
            "forecasts (the normal state)", ylabel="vol points")
    fig.tight_layout()
    return _save(fig, path)


def plot_vrp_history(hist: pd.DataFrame, horizon_days: int = 21,
                     path: str | Path | None = None):
    """
    Historical implied vs subsequently realised vol, and the premium between
    them — two stacked panels, because "two vol levels" and "the spread
    between them" are different questions.
    """
    import matplotlib.pyplot as plt

    th = TH.active()
    neg, mid, pos = th.diverging
    df = hist.dropna(subset=["realized_vol_fwd"])
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.6), sharex=True,
                             gridspec_kw={"height_ratios": [1.4, 1]})

    ax = axes[0]
    ax.plot(df.index, df["implied_vol"] * 100, color=th.categorical[0], lw=1.5,
            label="implied vol (option market, ex ante)")
    ax.plot(df.index, df["realized_vol_fwd"] * 100, color=th.categorical[1],
            lw=1.3, label=f"realised vol over the next {horizon_days}d (ex post)")
    ax.legend(loc="upper left", ncol=2)
    _titles(ax, "Implied volatility vs what actually happened",
            "the wedge between the two lines is the premium option sellers collect",
            ylabel="annualised vol (%)")

    ax = axes[1]
    v = df["vrp"] * 100
    ax.fill_between(df.index, 0, v.where(v >= 0), color=pos, alpha=0.75, lw=0)
    ax.fill_between(df.index, 0, v.where(v < 0), color=neg, alpha=0.75, lw=0)
    ax.axhline(0, color=th.axis, lw=1.0)
    ax.axhline(v.mean(), color=th.ink_secondary, lw=1.2, ls="--")
    ax.annotate(f"mean {v.mean():+.2f} vp", xy=(df.index[-1], v.mean()),
                xytext=(-4, 6), textcoords="offset points", ha="right",
                fontsize=9, color=th.ink_secondary, fontweight="bold")
    _titles(ax, "Volatility risk premium",
            f"positive {float((v > 0).mean())*100:.0f}% of the time — sellers win "
            f"often and lose big, which is why the premium exists",
            ylabel="vol points")
    fig.tight_layout()
    return _save(fig, path)


# --------------------------------------------------------------------------- #
# 5. Data quality & screening
# --------------------------------------------------------------------------- #
def plot_quote_funnel(funnel_df: pd.DataFrame, path: str | Path | None = None):
    """
    How many quotes survived each filter. A single ordered series, so a single
    hue — the bar length is the encoding, not the colour.
    """
    import matplotlib.pyplot as plt

    th = TH.active()
    df = funnel_df.iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(df) + 2.0))
    bars = ax.barh(df["stage"], df["n_quotes"], height=0.62,
                   color=th.categorical[0], edgecolor=th.surface, linewidth=1.2)
    ax.bar_label(bars, labels=[f"{int(n):,}  ({p:.0f}%)" for n, p in
                               zip(df["n_quotes"], df["pct_of_raw"])],
                 padding=4, fontsize=8.6, color=th.ink_secondary)
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    ax.margins(x=0.16)
    _titles(ax, "Quote-quality funnel",
            "how many raw option quotes survive each filter on the way to the "
            "surface", xlabel="quotes")
    fig.tight_layout()
    return _save(fig, path)


def plot_fit_residuals(surface, path: str | Path | None = None):
    """
    Calibration residuals in vol points against log-moneyness, coloured by the
    diverging ramp (sign is the point), with the market's half-spread as a
    reference band: residuals inside the band are not tradeable.
    """
    import matplotlib.pyplot as plt

    th = TH.active()
    q = surface.quotes.copy()
    q["iv_model"] = surface.iv(q["k"].to_numpy(), q["T"].to_numpy())
    q["resid"] = (q["iv"] - q["iv_model"]) * 100
    half = (q.get("iv_spread", pd.Series(np.nan, index=q.index)).abs() / 2 * 100)

    fig, ax = plt.subplots(figsize=(10, 4.8))
    if np.isfinite(half).any():
        order = np.argsort(q["k"].to_numpy())
        ks = q["k"].to_numpy()[order]
        hs = half.to_numpy()[order]
        ax.fill_between(ks, -hs, hs, color=th.grid, alpha=0.85, lw=0,
                        label="market half-spread (untradeable zone)")
    pos = q["resid"] >= 0
    ax.scatter(q["k"][pos], q["resid"][pos], s=13, color=th.diverging[2],
               alpha=0.8, label="market above model")
    ax.scatter(q["k"][~pos], q["resid"][~pos], s=13, color=th.diverging[0],
               alpha=0.8, label="market below model")
    ax.axhline(0, color=th.axis, lw=1.0)
    ax.legend(loc="upper right", ncol=3)
    rmse = float(np.sqrt(np.mean(q["resid"] ** 2)))
    _titles(ax, "Calibration residuals: market minus model",
            f"RMSE {rmse:.2f} vol points · residuals inside the grey band are "
            f"smaller than the cost of crossing the spread",
            ylabel="vol points", xlabel="log-moneyness  log(K/F)")
    fig.tight_layout()
    return _save(fig, path)


def plot_anomalies(anomalies: pd.DataFrame, path: str | Path | None = None):
    """
    Ranked screening results. Severity uses the reserved status palette and is
    always printed as text next to the bar — status colour never carries the
    meaning by itself.
    """
    import matplotlib.pyplot as plt

    th = TH.active()
    if anomalies is None or anomalies.empty:
        fig, ax = plt.subplots(figsize=(9, 2.4))
        ax.axis("off")
        ax.text(0.0, 0.6, "No anomalies flagged", fontsize=15, fontweight="bold",
                color=th.ink, transform=ax.transAxes)
        ax.text(0.0, 0.25, "The surface is internally consistent: no calendar or "
                "butterfly violations, no tenor out of line with its neighbours.",
                fontsize=10, color=th.ink_secondary, transform=ax.transAxes)
        return _save(fig, path)

    df = anomalies.head(12).iloc[::-1].copy()
    labels = [f"{r.category.replace('_', ' ')}"
              + (f" · {int(r.tenor_days)}d" if np.isfinite(r.tenor_days) else "")
              for r in df.itertuples()]
    mag = df["z_score"].abs().fillna(df["z_score"].abs().max() or 1.0).to_numpy()

    fig, ax = plt.subplots(figsize=(10, 0.46 * len(df) + 2.2))
    bars = ax.barh(labels, mag, height=0.6,
                   color=[TH.severity_color(s) for s in df["severity"]],
                   edgecolor=th.surface, linewidth=1.2)
    ax.bar_label(bars, labels=[f"  {s.upper()}  ·  z = {z:+.1f}" if np.isfinite(z)
                               else f"  {s.upper()}  ·  hard violation"
                               for s, z in zip(df["severity"], df["z_score"])],
                 padding=3, fontsize=8.4, color=th.ink_secondary)
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    ax.margins(x=0.28)
    _titles(ax, "Relative-value screen",
            "ranked by severity then magnitude · every flag needs a human check "
            "for a stale quote or a scheduled event before it is a trade",
            xlabel="|z-score| vs the cross-sectional benchmark")
    fig.tight_layout()
    return _save(fig, path)
