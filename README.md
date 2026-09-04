# GARCH Volatility Forecasting & the Implied Volatility Surface

**English** · **[Русский](README.ru.md)**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mvxddd/garch-vol-surface/blob/main/notebooks/garch_iv_surface_colab.ipynb)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Fit GARCH-family models to forecast realised volatility, build an
**arbitrage-free implied-volatility surface** from live option chains, and
measure the wedge between them — the volatility risk premium — with the
diagnostics needed to decide whether that wedge is real.

```bash
git clone https://github.com/mvxddd/garch-vol-surface && cd garch-vol-surface
pip install -r requirements.txt
python scripts/run_pipeline.py --ticker SPY          # full study, ~2 minutes
python scripts/run_pipeline.py --provider synthetic  # no network required
python scripts/run_pipeline.py --lang ru             # charts and CLI in Russian
python scripts/run_pipeline.py --snapshot            # also save to the history store
streamlit run app.py                                 # interactive web interface
pytest -q                                            # 91 tests, ~12 seconds
```

Or [open the notebook in Colab](https://colab.research.google.com/github/mvxddd/garch-vol-surface/blob/main/notebooks/garch_iv_surface_colab.ipynb) — cell 1 installs the one
dependency Colab lacks and clones the package for you.

Results below are from a live SPY run (2 Sep 2026): **4,417 raw quotes → 1,620
clean implied vols across 12 expiries**, every smile calibrated
arbitrage-free to within **0.04–0.35 vol points** of the market mid.

---

## Beyond a single snapshot

The core study describes one surface on one day. Four modules build on it.

### Portfolio risk — `volsurface/portfolio.py`

Feed it positions (or a CSV) and it prices them on the calibrated surface,
aggregates the Greeks, breaks vega down by tenor and by strike, and runs a
spot/vol stress grid:

```python
from volsurface import portfolio as PF
report = PF.risk_report(PF.Portfolio.from_csv("book.csv"), surface)
```

The stress grid is the point: a delta/vega summary cannot show gamma or vanna,
and a short strangle looks harmless in a Greeks table right up until it isn't.
How the smile moves with spot is an explicit choice — **sticky moneyness**
(default, right for index options) or **sticky strike** — because the two give
materially different P&L on a large move.

### Surface history — `volsurface/history.py`

`--snapshot` stores the day's surface on a fixed tenor grid (7/30/60/90/180/365
days, so numbers stay comparable as listed expiries roll). Once enough days have
accumulated, every metric is z-scored against **its own past**, and those
historical extremes join the cross-sectional screen in one ranked list. Below 30
snapshots it reports nothing rather than a meaningless z-score.

### Backtest — `volsurface/backtest.py`

A delta-hedged straddle harvesting the premium, with explicit transaction costs.
On live SPY since 2015 (139 non-overlapping 21-day trades, 10k vega, 1 vol point
round-trip cost):

| Strategy | Sharpe | Hit rate | Max drawdown |
|---|---:|---:|---:|
| Always short vol | 0.47 | 78% | −976k |
| Always long vol | −1.29 | 15% | −4.37M |
| Short only when VRP is rich (z > 0.5) | **1.55** | 83% | **−155k** |

The always-short curve is the classic short-vol shape: a steady grind up and a
cliff in February 2020, where a single trade lost 15 average wins. Timing on the
premium roughly triples the Sharpe and cuts the drawdown by 84% — which is the
result that makes the VRP study worth doing.

**What is deliberately not implemented:** backtesting individual strikes against
today's chain. There is no history of option prices in the free data, so such a
"backtest" would re-price the past with today's surface — a look-ahead that
guarantees a beautiful equity curve. `signal_backtest` replays real snapshot
history instead, and refuses to run on a store younger than 60 days.

### American exercise — `volsurface/models/american.py`

Every listed equity option is American, and pricing one as European biases its
implied volatility **upward** by the early-exercise value. A vectorised
Cox-Ross-Rubinstein tree prices and inverts the American contract; the whole
chain marches down the tree together in numpy, so 1,553 live SPY quotes invert
in about a second instead of minutes.

What the choice is worth, measured on that chain (mean IV shift, vol points):

| Days | Calls | Puts |
|---:|---:|---:|
| 28 | +0.01 | −0.05 |
| 87 | +0.01 | −0.16 |
| 196 | +0.01 | −0.21 |
| 378 | +0.01 | **−0.64** |

Textbook, and a good check that the implementation is right: **calls barely
move at all** (with a dividend yield below the risk-free rate, exercising a call
early is worth nothing), while the put shift grows monotonically with maturity
and reaches 1.7 vol points on the worst single contract. Of 1,553 quotes, 305
puts shift by more than 0.1 vol point and **zero calls do**.

So European remains the default: it is ~40x faster and the bias is negligible
inside three months, which is where most of the liquidity is. Turn American on
(`--american`, or `cfg.options.exercise_style`) for long-dated strikes, single
names, or whenever rates and dividends are large. Every quote then carries an
`early_exercise_premium` column reporting what the choice was worth to it.

One honest limit: inside about three weeks the early-exercise value is smaller
than the tree's own discretisation error, so the difference there is numerical
noise rather than signal. `convergence_check()` measures that error for your
parameters instead of asking you to trust a step count.

### Web interface — `app.py`

`streamlit run app.py`: pick an underlying, press the button, and get all seven
tabs — surface, forecast, premium, screen, portfolio risk, backtest, data — in
either language. Nothing runs on page load (a surface build hits a rate-limited
feed), and results are cached on the inputs.

---

## Interface language

Charts and terminal output are translated with `--lang` (`en` by default, `ru`
available), or from code:

```python
cfg = Config()
cfg.language = "ru"
```

**What is and is not translated.** Only what a person reads: axis labels, the
titles of all 15 charts, and the CLI summary. CSV column names, log messages
and exception text stay English — those are data and developer surfaces, and a
script reading the `vrp_vol_points` column must not break because someone
changed the display language.

For the same reason the volatility-risk-premium verdict travels as a stable
code (`rich` / `cheap` / `in_line` / `na`) and is rendered into prose only at
display time. Adding a language means adding one dict to `volsurface/i18n.py`;
the test suite then enforces full key coverage and matching placeholders.

---

## 1. Project Overview

Two questions that are usually studied separately, joined at the end.

**What will volatility be?** GARCH-family models are fitted by maximum
likelihood to daily log returns, ranked on BIC, and — the part that matters —
validated *out of sample* with a walk-forward procedure against realised
volatility and against naive benchmarks they must beat to justify their
complexity.

**What is volatility being priced at?** A full option chain across strike and
maturity is cleaned, inverted to implied volatility through Black-76 on
put-call-parity forwards, and calibrated to a raw-SVI surface under explicit
no-arbitrage constraints.

**The difference is the trade.** ATM implied minus GARCH forecast, matched by
horizon, is the volatility risk premium. Places where the surface disagrees
*with itself* — total variance falling with maturity, a smile implying a
negative density, a tenor off the curve through its neighbours — are the
relative-value screen.

What distinguishes this from a textbook exercise:

- **Forwards from put-call parity**, not a guessed dividend yield. A 0.3% error
  in the forward manufactures ~1 vol point of fake skew — larger than most of
  the effects being measured.
- **Arbitrage constraints are enforced, not just checked.** SVI is calibrated
  subject to Durrleman's condition, so the fitted density cannot go negative.
- **Out-of-sample by construction**, with Diebold-Mariano tests and Newey-West
  standard errors, because overlapping forecast windows inflate naive t-stats
  by roughly √h.
- **The pipeline reports what it threw away.** A quote-quality funnel counts
  every filter, so the surface's provenance is auditable.

## 2. Real-World Finance Use Case

| Desk | How this machinery is used |
|---|---|
| **Options market making** | The surface *is* the pricing engine. Quotes are made off a calibrated smile, not off individual contracts, so that every strike is consistent with its neighbours and the book cannot be picked off on a butterfly. |
| **Volatility arbitrage** | The VRP is the core carry trade: sell options, delta-hedge, collect the premium. Sizing depends on whether today's premium is rich or cheap versus its own history. |
| **Derivatives risk management** | Greeks are only meaningful against a smooth, arbitrage-free surface. Vega ladders, skew risk and calendar risk are all read off it. |
| **Structured products** | Exotic pricing needs a full surface as the boundary condition for a local- or stochastic-volatility model. Feeding it an arbitrageable surface produces a model that cannot be calibrated. |
| **Portfolio hedging** | The 25-delta risk reversal is the direct price of crash protection; the term structure says whether to buy it short- or long-dated. |

Two findings from the live run that show why the diagnostics matter:

- **The historical premium is real.** VIX against subsequently realised
  volatility, 1,906 overlapping observations: mean **+3.96 vol points**,
  positive **84.7%** of days, Newey-West *t* = **7.5**. The worst single
  observation is **−66 vol points** — the premium exists precisely because
  sellers occasionally get destroyed.
- **The current snapshot is the other way round.** 30-day implied 12.3% versus
  a GARCH forecast of 14.6%: a **negative** premium of −2.3 vol points. Realised
  volatility had recently collapsed while the model was still mean-reverting
  upward. A study that only reported the long-run average would have missed
  the state the market is actually in.

## 3. System Architecture

```
                    ┌──────────────────────────────────────────┐
  yfinance ────┐    │              volsurface/                 │
  Polygon  ────┼──► │                                          │
  synthetic ───┘    │  data/     prices · options · clean      │  ← retries,
                    │            synthetic fallback            │    caching,
                    │      │                                   │    quote funnel
                    │      ▼                                   │
                    │  models/   black_scholes ── IV inversion │
                    │            garch ───────── MLE, forecast │
                    │            svi ─────────── smile fit     │
                    │            surface ─────── (k,T) query   │
                    │      │                                   │
                    │      ▼                                   │
                    │  analytics/ metrics · vrp · skew         │
                    │      │                                   │
                    │      ▼                                   │
                    │  viz/      theme (CVD-validated) · plots │
                    │      │                                   │
                    │      ▼                                   │
                    │  pipeline.py  ── stage-isolated runner   │
                    └──────────────────────────────────────────┘
                           │                    │
                    outputs/reports/*.csv   outputs/figures/*.png|html
```

**Layering rule:** each layer imports only from the ones above it. `analytics`
never touches a data provider; `models` never knows what a provider is. That is
why switching from Yahoo to Polygon is a one-line config change rather than a
rewrite.

**Stage isolation.** Volatility modelling and surface construction are
independent research questions that happen to share an underlying. If the
option feed fails on a weekend, the GARCH half still runs and reports; the
pipeline records a status per stage rather than dying. This is tested
(`test_pipeline_survives_a_broken_option_feed`).

## 4. Required APIs and Data Sources

| Source | What it provides | Cost | Notes |
|---|---|---|---|
| **Yahoo Finance** (`yfinance`) | Daily OHLCV, full option chains, VIX/VXN/RVX history | Free | The default. Rate-limited and occasionally returns empty frames — hence the retry decorator and the cache. Bid/ask can be stale outside RTH. |
| **Polygon.io** | NBBO option quotes, reliable open interest, historical chains | Paid tier | The recommended upgrade. Set `POLYGON_API_KEY` and `--provider polygon`. The only real path to a *history* of surfaces. |
| **Synthetic generator** (built in) | Returns with volatility clustering + an arbitrage-free SVI chain | Free | Not a toy: prices are generated from valid SVI parameters, so the calibration round-trip is a genuine test. Makes the project runnable offline and in CI. |
| **FRED / Treasury** (optional) | Risk-free curve | Free | Currently a flat rate; the parity forward absorbs most of the misspecification anyway. |

**Data-quality caveats you should know before trusting any output:**

- Yahoo option quotes are snapshots and are **not simultaneous across
  expiries**, which manufactures small calendar violations. The live run flagged
  two, both under 0.003 in total variance — noise, not opportunity.
- Open interest updates once daily, before the open.
- Listed equity options are **American**. Both models ship: European (Black-76,
  the fast default) and American (a vectorised binomial tree, `--american`).
  See *American exercise* below for what the choice is worth, measured.

## 5. Required Python Libraries

| Library | Used for |
|---|---|
| `numpy`, `pandas` | Vectorised numerics; everything is array-based, no per-quote Python loops in the hot paths |
| `scipy` | `least_squares` for SVI calibration, `ndtr` for the normal CDF, splines |
| `arch` | GARCH / EGARCH / GJR maximum-likelihood estimation and forecasting |
| `statsmodels` | Ljung-Box and Engle ARCH-LM diagnostics |
| `yfinance`, `requests` | Market data |
| `matplotlib` | The static chart library |
| `plotly` | The interactive 3-D surface |
| `pyarrow` | Parquet cache |
| `pytest` | 40 tests covering pricing, calibration, arbitrage and the pipeline |

## 6. Folder / File Structure

```
garch-vol-surface/
├── volsurface/
│   ├── config.py               # every tunable, as typed dataclasses
│   ├── i18n.py                 # en/ru string catalogue + t()
│   ├── utils.py                # logging, retries, parquet cache, numerics
│   ├── pipeline.py             # stage-isolated orchestration + PipelineResult
│   ├── portfolio.py            # position risk, vega ladders, stress grids
│   ├── history.py              # daily snapshot store + z-scores
│   ├── backtest.py             # VRP harvesting with transaction costs
│   ├── data/
│   │   ├── prices.py           # OHLCV, returns, vol-index history
│   │   ├── options.py          # chain retrieval, normalised to one schema
│   │   ├── clean.py            # the funnel: filters → forwards → IV → features
│   │   └── synthetic.py        # offline generator (GJR returns + SVI chain)
│   ├── models/
│   │   ├── black_scholes.py    # Black-76, Greeks, vectorised IV inversion
│   │   ├── garch.py            # estimation, selection, walk-forward forecasts
│   │   ├── svi.py              # raw-SVI calibration + arbitrage diagnostics
│   │   └── surface.py          # slice assembly, (k,T) interpolation, queries
│   ├── analytics/
│   │   ├── metrics.py          # QLIKE, Mincer-Zarnowitz, Diebold-Mariano
│   │   ├── vrp.py              # volatility risk premium, snapshot + history
│   │   └── skew.py             # RR/BF/term structure + the anomaly screen
│   └── viz/
│       ├── theme.py            # CVD-validated palette, matplotlib + plotly
│       └── plots.py            # 15 charts
├── notebooks/
│   ├── garch_iv_surface_colab.py     # source of truth (percent format)
│   └── garch_iv_surface_colab.ipynb  # built artefact — open this in Colab
├── app.py                      # Streamlit web interface
├── scripts/
│   ├── run_pipeline.py         # CLI
│   ├── build_dashboard.py      # figures + report → one HTML page
│   └── build_notebook.py       # .py → .ipynb
├── tests/                      # 91 tests
└── outputs/
    ├── figures/                # 15 PNG + 1 interactive HTML
    └── reports/                # 15 CSV tables + run_report.json
```

**Colab note.** The notebook is kept in `# %%` percent format and the `.ipynb`
is generated from it. Scripts diff cleanly in git, can be linted, and can be
executed head-to-tail in CI — none of which is true of a notebook JSON blob.

## 7. Step-by-Step Build Guide

Build it in this order; each step is verifiable before the next depends on it.

1. **Pricing core first** (`models/black_scholes.py`). Black-76 on forwards,
   Greeks, and a vectorised implied-vol inverter. *Verify:* put-call parity to
   1e-12, vega against a finite difference, and a price→IV→price round trip on
   50,000 random contracts.
2. **Synthetic data** (`data/synthetic.py`). Generate returns from a GJR-GARCH
   recursion and an option chain from *valid* SVI parameters. Do this before
   touching a real API — it gives you ground truth to test against and makes
   every later step debuggable offline.
3. **Loaders + cleaning** (`data/`). Retries, caching, the quote funnel,
   parity forwards, OTM selection, IV inversion. *Verify:* inverted IVs match
   the synthetic ground truth to <0.05 vol points, and the recovered implied
   dividend matches the generator's.
4. **GARCH** (`models/garch.py`). Fit the suite, select on BIC, forecast a term
   structure. *Verify:* persistence in (0.8, 1.0), long-run vol plausible, and
   forecasts mean-reverting toward it. **Watch out:** EGARCH's intercept lives
   in *log*-variance — read it naively and you report a 0.00002% long-run vol.
5. **Walk-forward validation.** Expanding window, refit monthly, filter forward
   in between. *Verify:* the forward realised-vol target uses only returns
   strictly after the decision date. This look-ahead bug is the single most
   common way volatility studies produce fake skill.
6. **SVI calibration** (`models/svi.py`). Multi-start bounded least squares with
   the no-arbitrage constraints as penalty terms. *Verify:* fit SVI to data
   generated from SVI and recover the parameters (this project recovers
   ρ = −0.700 against a true −0.70).
7. **Surface assembly** (`models/surface.py`). Interpolate **total variance**
   linearly in T. *Verify:* total variance non-decreasing in T at every
   log-moneyness — the property that keeps the surface calendar-arb-free.
8. **Analytics** (`analytics/`). QLIKE, Mincer-Zarnowitz, Diebold-Mariano, VRP,
   skew metrics, the screen.
9. **Charts** (`viz/`). Theme first, charts second.
10. **Pipeline + CLI + notebook.** Wire it together with per-stage error
    handling.

### Three mistakes that cost the most time

- **Interpolating implied vol across maturity instead of total variance.** It
  looks fine and silently creates free calendar spreads between listed expiries.
- **Using a robust loss that also squashes your constraint penalties.** An
  earlier version of the SVI fit used `soft_l1` over the combined residual
  vector; it quietly returned arbitrageable smiles because the penalties were
  being clipped along with the outliers. Robustness and constraints need to be
  separate mechanisms.
- **Treating a fixed strike band as a chain.** Real chains list strikes scaled
  by σ√T. A fixed ±30% band asks for a two-week option 30% out of the money and
  gets back a 250% "implied vol" that no market maker ever quoted.

## 8. Data Collection Pipeline

```
load_prices ──► retry(3, exponential backoff) ──► parquet cache (12h TTL)
                                                  │
                                                  ├─ provider failure
                                                  ▼
                                          synthetic fallback (loud warning)
```

- **Expiry selection** is log-spaced across the available tenors — dense at the
  front where the term structure moves, sparse at the back — rather than taking
  the first N and clustering on the front month.
- **Caching** is keyed on a hash of the request parameters, with a 12-hour TTL
  for chains (they go stale fast) and a longer one for price history. A cache
  read failure logs and refetches; a cache write failure never breaks a run.
- **`df.attrs['synthetic']`** propagates through the whole pipeline, so
  simulated data can never be silently reported as market data.

## 9. Data Cleaning & Feature Engineering

The funnel from the live SPY run:

| Stage | Quotes | Kept |
|---|---:|---:|
| raw | 4,417 | 100% |
| valid price & strike, de-duplicated | 4,417 | 100% |
| bid ≥ 0.05 | 3,937 | 89.1% |
| relative spread ≤ 35% | 3,937 | 89.1% |
| open interest ≥ 10 or traded today | 3,739 | 84.7% |
| moneyness K/F ∈ [0.70, 1.30] | 2,960 | 67.0% |
| **OTM only** | 1,620 | 36.7% |
| IV successfully inverted | 1,620 | 36.7% |

**Why each filter exists**

- *No zero bids.* A 0 × 0.05 market has a "mid" of 0.025 that nobody will
  trade, and it produces an implied vol that is pure fiction.
- *OTM only.* ITM options carry identical volatility information but far more
  intrinsic value, so the same dollar spread becomes a much larger error in vol.
- *Vega identifiability.* The inverter returns NaN — never a guess — when vega
  at the solution is below `1e-6·F`. There the price is flat in σ to within
  double precision, so any vol in a wide band reprices the quote. This removes
  the last ~0.4% of nonsense IVs in a 50,000-contract stress test.

**Engineered features**

| Feature | Definition | Why |
|---|---|---|
| `forward` | put-call parity regression of C−P on K | absorbs dividends and funding without guessing `q` |
| `k` | log(K/F) | the natural, maturity-comparable x-axis of a smile |
| `total_variance` | IV²·T | the quantity that must be monotone in T |
| `vega` | Black-76 | the calibration weight — fit what can be traded |
| `iv_bid`, `iv_ask` | inverted bid and ask | the bid/ask **in vol points**: the only scale on which a "signal" can be judged tradeable |
| `delta` | Black-76 | for 25Δ risk reversals and butterflies |

## 10. Core Models / Algorithms

### Implied-volatility inversion

Safeguarded Newton with a maintained bisection bracket, fully vectorised. Newton
alone diverges in the wings where vega → 0; Brent alone is ~50× slower on a
5,000-quote chain. **50,000 inversions in 66 ms, median error 2e-14.** Prices
outside the static no-arbitrage bounds are rejected up front.

### GARCH family

$$\sigma_t^2 = \omega + (\alpha + \gamma \mathbb{1}_{r_{t-1}<0})r_{t-1}^2 + \beta\sigma_{t-1}^2$$

Four specifications (Normal, Student-t, GJR-t, EGARCH-skew-t), selected on BIC.
Fitted on returns scaled ×100 — MLE on decimal returns is badly conditioned and
routinely stalls. Multi-step forecasts are analytic where available and
simulated for EGARCH, which has no closed-form multi-step recursion.

Horizon aggregation averages **variance**, not volatility, because an option
prices off expected integrated variance to expiry.

### Walk-forward validation

Expanding window, MLE refit every 21 days, variance recursion filtered forward
in between via `ARCHModel.fix` — how a desk actually runs it, and ~20× cheaper
than refitting daily.

### Raw SVI

$$w(k) = a + b\left[\rho(k-m) + \sqrt{(k-m)^2+\sigma^2}\right], \qquad w = \text{IV}^2 T$$

Calibrated by multi-start bounded least squares with three constraints appended
to the residual vector:

1. minimum total variance $a + b\sigma\sqrt{1-\rho^2} \ge 0$
2. Lee's wing bound $b(1+|\rho|) \le 4/T$
3. Durrleman $g(k) \ge 0$ on a dense grid — equivalent to a non-negative
   risk-neutral density

with **penalty continuation** (escalate the weight until the constraints bind)
and one IRLS re-weighting pass for outlier quotes. Round-trip test: fitting SVI
to SVI-generated prices recovers ρ to 3 decimals and b to 4.

### Surface interpolation

Linear in **total variance** against T between calibrated slices. Short end
holds volatility constant; long end holds forward variance constant. Verified
monotone in T across the whole grid, which is what makes the interpolated
surface calendar-arbitrage-free.

## 11. Visualisations & Dashboard Components

15 figures, all built on one palette that was run through a colour-vision-
deficiency validator rather than chosen by eye — categorical hues clear the
adjacent-pair CVD and normal-vision floors in both light and dark modes, and
the tenor ramp is single-hue with visible lightness steps so it survives
greyscale printing.

| Figure | Form | What it answers |
|---|---|---|
| `iv_surface_3d.html` | interactive plotly surface | the headline object; drag to rotate, hover for values |
| `iv_surface_heatmap.png` | heatmap + contours | the same surface, but you can *read numbers off it* |
| `smile_grid.png` | small multiples | does the fit go through the quotes, expiry by expiry? |
| `smile_overlay.png` | ordinal ramp, ≤5 tenors | how the smile flattens with maturity |
| `risk_neutral_density.png` | line | is the density non-negative? (the fastest arbitrage check) |
| `fit_residuals.png` | diverging scatter + spread band | are residuals smaller than the cost of crossing? |
| `term_structure.png` | 3 series, one axis | implied vs forward vs GARCH — the VRP is the shaded band |
| `skew_term.png` | 2 stacked panels | risk reversal, butterfly, ATM skew decay |
| `conditional_vol.png` | 2 stacked panels | volatility clustering and the model's read of it |
| `forecast_vs_realized.png` | small multiples by horizon | out-of-sample honesty |
| `model_scorecard.png` | ranked bars, gap-to-best | which model wins, and by how much |
| `vrp_term.png` | diverging bars | premium by horizon |
| `vrp_history.png` | 2 stacked panels | is the premium persistent? |
| `quote_funnel.png` | horizontal bars | data provenance |
| `anomalies.png` | ranked bars, status colours | the screen output |

Three rules applied throughout: **never a dual axis** (two measures of different
scale become two panels); **ordered things get an ordered ramp, identities get
categorical hues**; **a legend plus direct labels for every multi-series chart**,
so colour is never the only channel carrying meaning.

The surface plots **veil the region outside the quoted strike range at each
tenor**. A two-week option is not listed 30% out of the money, and colouring
that corner like fitted data is the most misleading thing a surface plot can do.

## 12. Performance Metrics

### Volatility forecasting

| Metric | What it catches |
|---|---|
| **QLIKE** = mean[log σ²_f + RV²/σ²_f] | The ranking loss. Robust to noise in the RV proxy and punishes *under*-forecasting far harder — the correct asymmetry for anyone short options. |
| **Mincer-Zarnowitz** RV = α + β·F | β < 1 is the classic "GARCH over-reacts" finding. Reported with a t-test of β = 1. |
| **RMSE / MAE** | Interpretable in vol points, but dominated by the few explosive days. |
| **Diebold-Mariano** | Whether a ranking difference is real. Newey-West long-run variance (h−1 Bartlett lags) plus the Harvey small-sample correction. |
| **Benchmarks** | 21-day historical vol and RiskMetrics EWMA(0.94). A GARCH that cannot beat EWMA has not earned its complexity. |

Live SPY, 1-day horizon: every GARCH specification beats both benchmarks on
QLIKE, and the Diebold-Mariano tests confirm the margin over EWMA is
significant at 5%. R² against realised volatility runs 5–20% — normal, and not
a failure: realised vol over a short window is dominated by noise.

### Surface calibration

| Metric | Live SPY result |
|---|---|
| Per-expiry fit RMSE | 0.04 – 0.35 vol points |
| Per-expiry bias | ≤ 0.08 vol points, no systematic sign |
| Butterfly-arbitrage-free slices | 12 / 12 |
| Calendar violations | 2, both < 0.003 in total variance (non-simultaneous snapshots) |
| Quotes surviving the funnel | 1,620 / 4,417 |
| Model IV inside the market's bid/ask | 13 – 45% by expiry |

That last row deserves an explanation, because it looks like a failure and is
not. SPY option markets are often quoted a **penny wide**, which at these
premiums is a fraction of a vol point — tighter than the 0.04–0.35 vol-point
residual of a five-parameter smile fitted across 100+ strikes. In other words
the benchmark is harder than "inside the spread" on the tightest listed market
in the world. On the synthetic control, where spreads are set to realistic
retail-like widths, the same code puts **100%** of model vols inside the
bid/ask. The honest reading: the fit is excellent in absolute terms, and on SPY
it is still not tight enough to trade a single strike against the smile without
a real transaction-cost model. That is exactly why the anomaly screen measures
residuals **in units of the quote's own half-spread** rather than in vol points.

### Volatility risk premium

Newey-West corrected, since overlapping 21-day windows inflate a naive t-stat by
roughly √21: mean **+3.96 vol points**, positive **84.7%** of days, *t* = **7.5**
over 1,906 observations.

## 13. Final Deliverables

- **`volsurface/`** — a layered, tested Python package (~7,300 lines) with
  production error handling and a synthetic fallback that makes every path
  runnable offline.
- **A Colab notebook** — 40 cells, executes head-to-tail in ~2 minutes,
  narrated so each modelling decision is explained where it is made.
- **A CLI** — `scripts/run_pipeline.py`, with meaningful exit codes.
- **15 figures** including an interactive 3-D surface.
- **15 CSV tables + a JSON run report** capturing config, per-stage status and
  headline results — so any figure can be traced back to the run that made it.
- **91 tests** covering parity, Greeks, inversion round-trips, arbitrage
  freedom, look-ahead bias, and graceful degradation when a feed dies.

## 14. Resume Description

> **Volatility Forecasting & Implied Volatility Surface Engine** — Python
> (numpy/scipy/arch/pandas/plotly)
>
> Built an institutional-grade volatility research engine spanning statistical
> forecasting and derivatives pricing. Fitted and compared four GARCH-family
> specifications (GARCH, GJR, EGARCH, Student-t/skew-t) by maximum likelihood
> and validated them out-of-sample with a walk-forward procedure, ranking on
> QLIKE with Diebold-Mariano tests and Newey-West standard errors; all
> specifications beat RiskMetrics EWMA and historical-vol benchmarks
> significantly. Constructed arbitrage-free implied-volatility surfaces from
> live option chains: recovered forwards via put-call-parity regression,
> inverted 1,600+ quotes per snapshot with a vectorised safeguarded-Newton
> solver (50k inversions in 66 ms), and calibrated raw SVI smiles under
> Durrleman butterfly and Lee wing constraints, fitting the market to 0.04–0.35
> vol points with zero arbitrage violations. Quantified the volatility risk
> premium (mean +3.96 vol points, Newey-West t = 7.5 over 1,900 observations)
> and built a relative-value screen for calendar, butterfly, term-structure and
> skew anomalies. Shipped with 40 tests, a stage-isolated pipeline that degrades
> gracefully on feed failure, and a colour-vision-validated chart library.

**Talking points if asked:**

- *Why total variance for interpolation?* Because linear interpolation of a
  non-decreasing total-variance term structure is still non-decreasing, so the
  interpolated surface stays calendar-arbitrage-free. Interpolating IV directly
  does not have that property.
- *Why QLIKE over RMSE?* Realised variance is a noisy proxy for a latent
  quantity; QLIKE is robust to that noise and penalises under-prediction
  asymmetrically, which matches the loss function of anyone short volatility.
- *What is the biggest weakness?* American exercise. Listed equity options are
  American and this prices them as European. For OTM index options the bias is
  small, but for single names with dividends it is the first thing to fix.

## 15. Potential Upgrades

**Modelling**

- **HAR-RV on intraday data.** With 5-minute bars, realised-volatility models
  routinely beat GARCH at horizons beyond a day.
- **Stochastic-volatility calibration** (Heston, SABR, rough Bergomi) fitted to
  the whole surface, giving a dynamic model rather than a static snapshot.
- **SSVI / eSSVI**: calibrate all maturities jointly so calendar-arbitrage
  freedom is guaranteed by construction rather than checked afterwards.
- **Regime switching** or a realised-GARCH that uses intraday realised measures
  as the driving variable.

**Data & infrastructure**

- **Polygon historical chains** → a *history* of surfaces, which is what turns
  the anomaly screen from cross-sectional to time-series and lets every metric
  be z-scored against its own past.
- **Timestamp alignment.** Snapshotting quotes with exchange timestamps and
  aligning them would remove the spurious calendar violations entirely.
- **Daily persistence** to a database, plus a scheduled job — the surface
  becomes a monitored asset rather than a one-off.
- **Earnings and dividend calendars** to explain away term-structure kinks
  automatically instead of leaving it to the reader.

**Trading**

- **Backtest the signals.** Every anomaly the screen flags should be tradeable
  as a delta-hedged position with transaction costs; without that, the screen is
  a hypothesis generator, not a strategy.
- **Vega-weighted portfolio construction** across the flagged opportunities,
  with limits on net vega, skew and calendar exposure.

---

## Licence & disclaimer

MIT. **This is research and educational software, not investment advice.** It
produces model output from public market data; nothing here is a recommendation
to trade, and the limitations listed above are real. Verify every number before
risking capital on it.
