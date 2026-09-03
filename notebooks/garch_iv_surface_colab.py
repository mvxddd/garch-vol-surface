# %% [markdown]
# # GARCH Volatility Forecasting & the Implied Volatility Surface
#
# **What this notebook does.** Two halves of the same question, joined at the end.
#
# 1. **The model's view.** Fit GARCH-family models to daily returns, pick one on
#    information criteria, and validate it *out of sample* against realised
#    volatility — including against benchmarks it has to beat to be worth using.
# 2. **The market's view.** Pull a full option chain, invert every quote to an
#    implied volatility, and calibrate an arbitrage-free surface across strike
#    and maturity.
# 3. **The difference between them** is the volatility risk premium, and the
#    places where the surface disagrees with itself are relative-value candidates.
#
# Every cell below is a thin wrapper around the `volsurface` package, so the
# notebook stays readable and the logic stays tested.
#
# ---
# **Runtime:** ≈2 minutes on Colab (≈30s without the walk-forward validation).

# %% [markdown]
# ## Cell 1 — Environment setup
#
# `arch` is the only dependency Colab does not ship with. The cell then locates
# the `volsurface` package: it searches the usual places, and if you opened this
# straight from the README badge — where nothing is on the machine yet — it
# clones the repository for you. Running from your own clone, it just adds the
# project to `sys.path` without installing anything.

# %%
import subprocess
import sys
from pathlib import Path

REQUIRED = ["arch", "yfinance", "plotly", "statsmodels"]
missing = []
for pkg in REQUIRED:
    try:
        __import__(pkg)
    except ImportError:
        missing.append(pkg)
if missing:
    print(f"Installing: {', '.join(missing)}")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *missing], check=True)

REPO_URL = "https://github.com/mvxddd/garch-vol-surface"

def locate_package() -> Path | None:
    """Find a directory containing the `volsurface` package, or None."""
    for candidate in (Path.cwd(), Path.cwd().parent, Path("/content"),
                      Path("/content/garch-vol-surface"),
                      Path("/content/drive/MyDrive")):
        if (candidate / "volsurface").is_dir():
            return candidate
    return None

# Make the repository importable whether it was cloned, uploaded, installed, or
# not present at all. On Colab (opened straight from the README badge) nothing
# is present, so clone it — that is what makes the badge genuinely one-click.
root = locate_package()
if root is None:
    IN_COLAB = "google.colab" in sys.modules or Path("/content").is_dir()
    if IN_COLAB:
        target = Path("/content/garch-vol-surface")
        print(f"Cloning {REPO_URL} → {target}")
        subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(target)],
                       check=True)
        root = target
    else:
        try:
            import volsurface  # noqa: F401  (installed, e.g. pip install -e .)
        except ImportError:
            raise SystemExit(
                "Could not find the `volsurface` package.\n\n"
                f"Clone it:  git clone {REPO_URL}\n"
                "then run this notebook from inside the project folder."
            )
if root is not None and str(root) not in sys.path:
    sys.path.insert(0, str(root))

import numpy as np
import pandas as pd

import volsurface
from volsurface import Config
from volsurface.viz import use_theme

pd.set_option("display.width", 140, "display.max_columns", 40,
              "display.float_format", lambda v: f"{v:,.4f}")
use_theme("light")          # "dark" also available; both palettes are CVD-validated
print(f"volsurface {volsurface.__version__} ready")

# %% [markdown]
# ## Cell 2 — Configuration
#
# Everything tunable lives in one place. The defaults target a liquid US index
# ETF, which is the right starting point: single names have sparser chains, and
# their skew is contaminated by earnings dates.
#
# Set `provider = "synthetic"` to run the whole notebook with no network at all —
# the generator produces returns with volatility clustering and an
# arbitrage-free SVI surface, so every downstream number stays meaningful.

# %%
cfg = Config()
cfg.data.ticker = "SPY"
cfg.data.start = "2016-01-01"
cfg.data.provider = "yfinance"        # "polygon" (needs POLYGON_API_KEY) | "synthetic"

cfg.options.max_expiries = 12
cfg.options.risk_free_rate = 0.042    # flat discount curve; forwards come from parity

cfg.garch.oos_fraction = 0.30         # last 30% of history is held out
cfg.garch.forecast_horizons = (1, 5, 21, 63)

cfg.ensure_dirs()
print(f"{cfg.data.ticker} | {cfg.data.start} → today | provider={cfg.data.provider}")

# %% [markdown]
# ## Cell 3 — Load prices and test for the effect we are about to model
#
# Before fitting a volatility model, check that there is volatility clustering to
# model. Engle's ARCH-LM test regresses squared returns on their own lags: a tiny
# p-value means today's shock size predicts tomorrow's, which is the entire
# premise of GARCH. If this test came back insignificant, the right move would be
# to stop and use a constant-volatility model.

# %%
from volsurface.data import load_prices
from volsurface.data.prices import compute_returns
from volsurface.models import garch as G

prices = load_prices(cfg.data)
returns = compute_returns(prices, field=cfg.data.price_field)
spot = float(prices["Close"].iloc[-1])

arch_test = G.arch_effect_test(returns, lags=12)
print(f"{len(prices):,} sessions | spot {spot:,.2f} | "
      f"unconditional vol {returns.std() * np.sqrt(252):.2%}")
print(f"ARCH-LM(12): statistic {arch_test['lm_stat']:.1f}, "
      f"p-value {arch_test['lm_pvalue']:.2e} "
      f"→ {'volatility clustering confirmed' if arch_test['lm_pvalue'] < 0.05 else 'NO clustering — GARCH is not warranted'}")
prices.tail(3)

# %% [markdown]
# ## Cell 4 — Fit the GARCH family and choose a specification
#
# Four specifications, increasing in generality:
#
# | Model | What it adds |
# |---|---|
# | GARCH(1,1)-Normal | the baseline: variance mean-reverts, shocks persist |
# | GARCH(1,1)-t | fat-tailed innovations — equity returns are not Gaussian |
# | GJR-GARCH(1,1,1)-t | **leverage**: down moves raise variance more than up moves |
# | EGARCH(1,1,1)-skewt | leverage in log-variance, plus skewed innovations |
#
# Selection is by **BIC**, which penalises parameters more heavily than AIC —
# appropriate when the extra parameters are there to be *forecast* with, not
# just to fit history.
#
# The number to look at is **persistence** (α + γ/2 + β). At 0.97 a shock has a
# half-life of about three weeks; above 1.0 the model is non-stationary and its
# long-horizon forecasts are meaningless.

# %%
fits = G.fit_model_suite(returns, cfg.garch)
best_fit = G.select_best(fits, criterion="bic")
comparison = G.comparison_table(fits)
comparison_display = comparison.assign(
    long_run_vol=lambda d: (d["long_run_vol"] * 100).round(2).astype(str) + "%")
display(comparison_display.round({"loglik": 1, "aic": 1, "bic": 1, "persistence": 4}))

hl = np.log(0.5) / np.log(best_fit.persistence)
print(f"\nSelected: {best_fit.name}")
print(f"  persistence   {best_fit.persistence:.4f}  (shock half-life ≈ {hl:.0f} trading days)")
print(f"  long-run vol  {best_fit.long_run_vol:.2%} annualised — every forecast decays toward this")

# %% [markdown]
# ## Cell 5 — Residual diagnostics
#
# A correctly specified volatility model leaves standardised residuals with no
# remaining structure. Both Ljung-Box p-values and the ARCH-LM p-value should sit
# comfortably above 0.05; if the squared-residual test fails, the variance
# equation is still missing something.
#
# Excess kurtosis in the standardised residuals is normal and is exactly why the
# Student-t and skew-t specifications usually win on BIC.

# %%
diag = G.residual_diagnostics(best_fit, lags=20)
for key, label in [("ljungbox_z_pvalue", "Ljung-Box on residuals (no autocorrelation)"),
                   ("ljungbox_z2_pvalue", "Ljung-Box on squared residuals (no ARCH left)"),
                   ("arch_lm_pvalue", "ARCH-LM on residuals"),
                   ("skew_z", "residual skewness"),
                   ("kurtosis_z", "residual excess kurtosis")]:
    v = diag.get(key, float("nan"))
    verdict = ""
    if key.endswith("pvalue"):
        verdict = "  ✓ pass" if v > 0.05 else "  ✗ structure remains"
    print(f"  {label:<48} {v:>8.4f}{verdict}")

# %% [markdown]
# ## Cell 6 — Conditional volatility through time
#
# The top panel is the raw material: returns arrive in calm and violent runs.
# The bottom panel is what the model makes of it — conditional volatility
# tracking realised volatility and decaying toward the long-run level.

# %%
from volsurface.viz import plots as P

fig = P.plot_conditional_vol(returns, best_fit, realized_window=21,
                             path=cfg.figure_dir / "conditional_vol.png")
fig.show()

# %% [markdown]
# ## Cell 7 — Out-of-sample walk-forward validation
#
# This is the cell that separates a real study from a curve-fitting exercise.
#
# At every out-of-sample date we forecast using **only** data available up to
# that date, then compare against what actually happened next. Parameters are
# re-estimated monthly and the variance recursion is filtered forward in
# between — exactly how a desk runs it, and ≈20× cheaper than refitting daily.
#
# Two naive benchmarks are included on purpose: a 21-day trailing realised vol
# and RiskMetrics EWMA. A GARCH model that cannot beat EWMA has not earned its
# complexity, and reporting that honestly is the point.
#
# *This is the slow cell (≈1 minute). Set `RUN_WALK_FORWARD = False` to skip it.*

# %%
from volsurface.analytics import metrics as M

RUN_WALK_FORWARD = True

if RUN_WALK_FORWARD:
    frames = []
    for name, vol_model, p, o, q, dist in cfg.garch.specs:
        frames.append(G.walk_forward_forecast(returns, cfg.garch, vol_model=vol_model,
                                              p=p, o=o, q=q, dist=dist, name=name))
    walk_forward = pd.concat(frames, ignore_index=True)
    walk_forward = pd.concat(
        [walk_forward,
         M.naive_benchmarks(returns, cfg.garch.forecast_horizons,
                            walk_forward["date"].unique())],
        ignore_index=True)
    evaluation = M.evaluate_walk_forward(walk_forward)
    display(evaluation.round(4))
else:
    walk_forward = evaluation = None

# %% [markdown]
# ## Cell 8 — Reading the scorecard
#
# * **QLIKE** is the loss to rank on. It is robust to the fact that realised
#   variance is a noisy proxy for the latent quantity, and it punishes
#   *under*-forecasting far harder than over-forecasting — which is the correct
#   asymmetry for anyone who is short options.
# * **Mincer-Zarnowitz β** below 1 is the classic finding that GARCH over-reacts:
#   when it forecasts high volatility, realised volatility comes in lower.
# * **R²** of 5–20% is normal and is not a failure. Volatility is far more
#   predictable than returns, but realised volatility over a short window is
#   still dominated by noise.
# * **Diebold-Mariano** answers whether a ranking difference is real. Overlapping
#   forecasts are autocorrelated, so the test uses a Newey-West long-run variance
#   plus the Harvey small-sample correction.

# %%
if evaluation is not None:
    fig = P.plot_model_scorecard(evaluation, metric="qlike",
                                 path=cfg.figure_dir / "model_scorecard.png")
    fig.show()

    dm = M.compare_models_dm(walk_forward, loss="qlike")
    sig = dm[dm["significant_5pct"]].sort_values("p_value")
    print(f"\n{len(sig)} of {len(dm)} pairwise comparisons are significant at 5%:")
    display(sig.head(10)[["horizon", "model_a", "model_b", "dm_stat", "p_value", "winner"]])

# %% [markdown]
# ## Cell 9 — Forecast versus what actually happened

# %%
if walk_forward is not None:
    champion = evaluation.sort_values("qlike")["model"].iloc[0]
    fig = P.plot_forecast_vs_realized(walk_forward, model=champion,
                                      path=cfg.figure_dir / "forecast_vs_realized.png")
    fig.show()

# %% [markdown]
# ## Cell 10 — The GARCH volatility term structure
#
# Forecasting *forward* from today gives the model's own term structure: the
# average volatility it expects over the next 1, 5, 21 … days. This is the
# object that is directly comparable to implied volatility by tenor, and it is
# what the volatility risk premium is measured against.
#
# Note that variance, not volatility, is averaged across the horizon — an option
# prices off expected integrated variance to expiry.

# %%
garch_ts = G.forecast_term_structure(best_fit, horizons=(1, 5, 21, 42, 63, 126, 252))
garch_ts["garch_vol_ann"] = garch_ts["garch_vol_ann"].round(4)
display(garch_ts)

# %% [markdown]
# ---
# # Part 2 — The market's view
#
# ## Cell 11 — Pull the option chain and clean it
#
# The funnel below is the most under-appreciated part of any surface project.
# Roughly half of a raw chain is unusable: zero bids, stale quotes, spreads wider
# than the signal, and deep-ITM contracts whose price is almost entirely
# intrinsic value. Filtering aggressively and *reporting what was filtered* is
# what makes the resulting surface trustworthy.
#
# Two decisions worth understanding:
#
# * **Forwards come from put-call parity**, not from a guessed dividend yield.
#   Regressing C−P on K recovers the forward the market is actually using. An
#   error of 0.3% in the forward would show up as ≈1 vol point of fake skew.
# * **Only OTM quotes are kept.** ITM options carry the same volatility
#   information but far more intrinsic value, so the same dollar spread becomes a
#   much larger error in vol.

# %%
from volsurface.data import load_option_chain, prepare_quotes

chain = load_option_chain(cfg.data, cfg.options, spot=spot, prices=prices)
quotes, forwards, funnel = prepare_quotes(chain, cfg.options, spot=spot)

display(funnel.to_frame())
display(forwards)
fig = P.plot_quote_funnel(funnel.to_frame(), path=cfg.figure_dir / "quote_funnel.png")
fig.show()

# %% [markdown]
# ## Cell 12 — Calibrate the surface
#
# Each expiry is fitted with **raw SVI**:
#
# $$w(k) = a + b\left[\rho(k-m) + \sqrt{(k-m)^2 + \sigma^2}\right], \qquad w = \text{IV}^2 T$$
#
# Five parameters reproduce essentially every observed equity smile, and — the
# reason it is the market standard — there are explicit conditions on
# $(a,b,\rho,m,\sigma)$ that guarantee the implied density stays non-negative.
# The calibration enforces them, so the surface it produces can actually be used
# to price and hedge.
#
# The fit is **vega-weighted**: the quotes a market maker can trade in size get
# the weight, not the noisy 5-delta wings.
#
# Across maturities the surface interpolates **total variance** linearly in $T$,
# which preserves the absence of calendar arbitrage. Interpolating implied
# volatility directly does not, and routinely manufactures free calendar spreads.

# %%
from volsurface.models.surface import build_surface

surface = build_surface(quotes, forwards, spot, cfg.surface,
                        risk_free_rate=cfg.options.risk_free_rate)

slices = surface.slice_table()
display(slices.round(4))
print("Fit quality by expiry (residuals in vol points):")
display(surface.fit_quality().round(4))

# %% [markdown]
# ## Cell 13 — Does the fit go through the quotes?
#
# Small multiples, one panel per expiry: market mids, the market's own bid/ask
# converted into vol points, and the calibrated smile. A fit RMSE well under the
# half-spread means the model is inside the market — the only standard that
# matters, because a residual smaller than the cost of crossing the spread is not
# tradeable.

# %%
fig = P.plot_smile_grid(surface, path=cfg.figure_dir / "smile_grid.png")
fig.show()

fig = P.plot_fit_residuals(surface, path=cfg.figure_dir / "fit_residuals.png")
fig.show()

# %% [markdown]
# ## Cell 14 — The surface in three dimensions
#
# The interactive version is the one to explore (drag to rotate); the heatmap is
# the one to read numbers off. Both mark where the surface stops being supported
# by quotes — a short-dated option is simply not listed 30% out of the money, and
# colouring that corner like fitted data would be the most misleading thing a
# surface plot can do.

# %%
fig3d = P.plot_surface_3d(surface, path=cfg.figure_dir / "iv_surface_3d.html")
fig3d.show()

fig = P.plot_surface_heatmap(surface, path=cfg.figure_dir / "iv_surface_heatmap.png")
fig.show()

# %% [markdown]
# ## Cell 15 — Skew, term structure, and the implied density
#
# Three views of the same surface, in the language a desk uses:
#
# * **Risk reversal** (25Δ put − 25Δ call): the price of skew. Persistently
#   positive on equities — the crash-insurance premium.
# * **Butterfly** (wings − ATM): the price of convexity.
# * **ATM skew** ∂IV/∂k: flattens roughly like $1/\sqrt{T}$.
#
# The risk-neutral density is the fastest sanity check on the whole surface. Any
# dip below zero would be a butterfly arbitrage; the fat left tail is what the
# skew looks like in probability space.

# %%
from volsurface.analytics import skew as SK

skew_df = SK.skew_metrics(surface)
term_df = SK.term_structure_metrics(surface)
display(skew_df.round(4))

fig = P.plot_skew_term(skew_df, path=cfg.figure_dir / "skew_term.png")
fig.show()
fig = P.plot_risk_neutral_density(surface, path=cfg.figure_dir / "risk_neutral_density.png")
fig.show()

# %% [markdown]
# ---
# # Part 3 — Model versus market
#
# ## Cell 16 — The volatility risk premium
#
# $$\text{VRP}(T) = \text{ATM implied vol}(T) - \text{GARCH forecast vol}(T)$$
#
# One subtlety that is easy to get wrong and expensive to miss: implied vols live
# on a **calendar-day** clock and GARCH horizons on a **trading-day** clock. A
# 21-trading-day forecast must be compared against an option with about 30
# calendar days to run. Mixing the two is a 20% error in $T$ that shows up as
# a spurious signal.
#
# A positive premium is the normal state — selling options means selling
# insurance against exactly the states investors most fear. When the premium goes
# *negative*, it usually means realised volatility has recently collapsed and the
# model is mean-reverting upward faster than the market is pricing.

# %%
from volsurface.analytics import vrp as VRP

vrp_now = VRP.current_vrp(surface, garch_ts, cfg.analytics.vrp_horizons_days)
display(vrp_now.round(4))

fig = P.plot_term_structure(term_df, garch_ts, path=cfg.figure_dir / "term_structure.png")
fig.show()
fig = P.plot_vrp_term(vrp_now, path=cfg.figure_dir / "vrp_term.png")
fig.show()

# %% [markdown]
# ## Cell 17 — Is the premium real? A historical study
#
# A snapshot is an anecdote. Using the listed volatility index (VIX for SPY/SPX,
# VXN for QQQ …) as a long history of ATM implied vol, we can measure the premium
# an option seller actually earned, day by day.
#
# The t-statistic uses a Newey-West correction. Overlapping 21-day windows make
# the series strongly autocorrelated, and a naive t-stat on 2,000 overlapping
# observations is inflated by roughly √21 — enough to turn noise into a "highly
# significant" result.

# %%
from volsurface.data.prices import load_vol_index, synthetic_vol_index

iv_index = load_vol_index(cfg.data)
if iv_index is None:
    iv_index = synthetic_vol_index(returns)      # offline fallback, indicative only

vrp_hist = VRP.historical_vrp(iv_index, returns, horizon_days=21)
stats = VRP.vrp_summary(vrp_hist)
for k, v in stats.items():
    print(f"  {k:<26} {v:,.2f}" if isinstance(v, float) else f"  {k:<26} {v}")

fig = P.plot_vrp_history(vrp_hist, path=cfg.figure_dir / "vrp_history.png")
fig.show()

# %% [markdown]
# ## Cell 18 — Relative-value screen
#
# Six families of check, ordered from "this is arbitrage" to "this is an outlier
# worth a look":
#
# 1. **Calendar arbitrage** — total variance falling with maturity.
# 2. **Butterfly arbitrage** — a smile implying a negative density.
# 3. **Term-structure kinks** — one tenor off the smooth curve through its neighbours.
# 4. **Skew kinks** — a risk reversal out of line with the 1/√T decay.
# 5. **Quote outliers** — a strike far from its own smile *relative to its spread*.
# 6. **VRP extremes** — the premium far from its own trailing distribution.
#
# **Read the output with suspicion.** Almost every flag has a boring explanation:
# a stale quote, a dividend, an earnings date inside one expiry but not the next.
# That is why each row carries its evidence. The screen's job is to hand a human
# a short ranked list, not to place trades.

# %%
anomalies = SK.detect_anomalies(surface, cfg.analytics, vrp=vrp_hist)
if len(anomalies):
    display(anomalies[["category", "tenor_days", "metric", "value", "benchmark",
                       "z_score", "severity"]].round(4))
    print("\nTop finding:", anomalies.iloc[0]["detail"])
    print("Suggested action:", anomalies.iloc[0]["suggested_action"])
else:
    print("No anomalies — the surface is internally consistent, which is the "
          "normal outcome on a liquid underlying.")

fig = P.plot_anomalies(anomalies, path=cfg.figure_dir / "anomalies.png")
fig.show()

# %% [markdown]
# ## Cell 19 — One-line reproduction, and saving everything
#
# Everything above is also available as a single call, with per-stage error
# handling so that a failure in the option feed does not take down the GARCH
# results (and vice versa).

# %%
from volsurface import run_pipeline

result = run_pipeline(cfg, make_figures=True, run_walk_forward=False)
written = result.save()

print("\nHeadline results")
for k, v in result.headline().items():
    print(f"  {k:<22} {v}")
print(f"\nWrote {len(written)} tables + {len(result.figures)} figures under {cfg.output_dir}/")

# %% [markdown]
# ---
# ## What to take away
#
# * GARCH beats naive benchmarks out of sample on QLIKE at every horizon, but its
#   Mincer-Zarnowitz β below 1 says it over-reacts — a known, reportable limitation.
# * The implied surface is calibrated arbitrage-free to within a fraction of the
#   market's own bid/ask, so it can be used to price and hedge, not just to plot.
# * The gap between the two — the volatility risk premium — is persistent,
#   positive on average, and statistically significant once you correct the
#   standard errors for overlapping windows.
#
# ### Where this would need more work before trading it
#
# * **American exercise.** Listed equity options are American; Black-76 treats
#   them as European. For OTM options on a low-dividend index the bias is small,
#   but it is not zero, and it is the first thing to fix for single names.
# * **Snapshot timing.** Quotes across expiries are not perfectly simultaneous,
#   which manufactures small calendar violations. A production system stamps and
#   aligns quote times.
# * **One surface, one day.** Term-structure and skew signals need a *history* of
#   surfaces to z-score against. Persist a daily snapshot and the screen gets far
#   sharper.
