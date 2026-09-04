"""
Streamlit front end.

    pip install streamlit
    streamlit run app.py

Pick an underlying, set the parameters, press the button. Everything here is a
thin wrapper around the same `volsurface` package the CLI and the notebook use
— the UI adds no analytics of its own, so a number on screen is the same number
the tests cover.

Two deliberate choices:

* **Nothing runs on load.** Building a surface hits a rate-limited data feed and
  fits a few dozen non-linear calibrations. A page that did that on every
  keystroke would be unusable and would get the user throttled, so the run is
  behind an explicit button.
* **Results are cached and keyed on the inputs.** Changing a chart tab must not
  re-download the chain.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    import streamlit as st
except ImportError:  # pragma: no cover
    raise SystemExit("Streamlit is not installed.  pip install streamlit")

from volsurface import Config, run_pipeline                       # noqa: E402
from volsurface import portfolio as PF                            # noqa: E402
from volsurface.backtest import (CostModel, compare_strategies,   # noqa: E402
                                 short_straddle_backtest)
from volsurface.data import load_prices                           # noqa: E402
from volsurface.data.prices import (compute_returns, load_vol_index,  # noqa: E402
                                    synthetic_vol_index)
from volsurface.i18n import set_language, t                       # noqa: E402
from volsurface.viz import plots as P, use_theme                  # noqa: E402

UI = {
    "en": {
        "title": "Volatility surface workbench",
        "tagline": "GARCH forecasting, an arbitrage-free implied vol surface, "
                   "and the premium between them.",
        "settings": "Settings", "ticker": "Underlying", "history": "Price history from",
        "provider": "Data source", "expiries": "Max expiries", "lang": "Language",
        "walk": "Run walk-forward validation (slower)",
        "snapshot": "Save a snapshot to history",
        "run": "Run analysis", "running": "Downloading data and calibrating…",
        "tab_surface": "Surface", "tab_garch": "Forecast", "tab_vrp": "Premium",
        "tab_screen": "Screen", "tab_risk": "Portfolio risk",
        "tab_backtest": "Backtest", "tab_data": "Data",
        "welcome": "Set the parameters on the left and press **Run analysis**.",
        "no_surface": "This run produced no surface — see the Data tab for why.",
        "stale": "Settings changed since this run. Press Run analysis again.",
        "bt_holding": "Holding period (trading days)",
        "bt_vega": "Vega notional", "bt_spread": "Option spread (vol points)",
        "bt_note": "Backtested at the index level on the listed vol index and "
                   "realised returns. Per-strike backtests need historical "
                   "option chains, which the free feed does not provide.",
        "risk_note": "An example book: a short front-month strangle against a "
                     "long back-month put, plus a delta hedge. Upload a CSV with "
                     "columns instrument, quantity, strike, expiry to use your own.",
        "upload": "Positions CSV (optional)",
        "totals": "Book totals", "synthetic": "Synthetic data — not a real market",
    },
    "ru": {
        "title": "Мастерская поверхности волатильности",
        "tagline": "Прогноз GARCH, безарбитражная поверхность подразумеваемой "
                   "волатильности и премия между ними.",
        "settings": "Параметры", "ticker": "Базовый актив",
        "history": "История цен с", "provider": "Источник данных",
        "expiries": "Максимум экспираций", "lang": "Язык",
        "walk": "Валидация вне выборки (дольше)",
        "snapshot": "Сохранить снимок в историю",
        "run": "Рассчитать", "running": "Загружаю данные и калибрую…",
        "tab_surface": "Поверхность", "tab_garch": "Прогноз", "tab_vrp": "Премия",
        "tab_screen": "Скрин", "tab_risk": "Риск портфеля",
        "tab_backtest": "Бэктест", "tab_data": "Данные",
        "welcome": "Задайте параметры слева и нажмите **Рассчитать**.",
        "no_surface": "В этом прогоне поверхность не построена — причина на "
                      "вкладке «Данные».",
        "stale": "Параметры изменились после расчёта. Нажмите «Рассчитать» снова.",
        "bt_holding": "Срок удержания (торговых дней)",
        "bt_vega": "Вега-ноционал", "bt_spread": "Спред опциона (пункты волы)",
        "bt_note": "Бэктест на уровне индекса: по биржевому индексу волатильности "
                   "и реализованным доходностям. Постраечный бэктест требует "
                   "исторических цепочек опционов, которых бесплатный источник "
                   "не даёт.",
        "risk_note": "Пример книги: короткий ближний стрэнгл против длинного "
                     "дальнего пута плюс дельта-хедж. Загрузите CSV с колонками "
                     "instrument, quantity, strike, expiry, чтобы считать свою.",
        "upload": "CSV с позициями (необязательно)",
        "totals": "Итоги по книге",
        "synthetic": "Синтетические данные — это не рынок",
    },
}

st.set_page_config(page_title="volsurface", page_icon="📉", layout="wide")


@st.cache_data(show_spinner=False, ttl=3600)
def _run(ticker: str, start: str, provider: str, max_expiries: int,
         lang: str, walk_forward: bool, snapshot: bool):
    """Cached pipeline run, keyed on every input that changes the result."""
    cfg = Config()
    cfg.data.ticker = ticker
    cfg.data.start = start
    cfg.data.provider = provider
    cfg.options.max_expiries = max_expiries
    cfg.language = lang
    cfg.analytics.save_snapshot = snapshot
    cfg.verbose = False
    return run_pipeline(cfg, make_figures=False, run_walk_forward=walk_forward)


@st.cache_data(show_spinner=False, ttl=3600)
def _vol_index(ticker: str, start: str, provider: str):
    from volsurface.config import DataConfig

    cfg = DataConfig(ticker=ticker, start=start, provider=provider)
    prices = load_prices(cfg)
    returns = compute_returns(prices)
    iv = load_vol_index(cfg)
    synthetic = iv is None
    if synthetic:
        iv = synthetic_vol_index(returns)
    return iv, returns, synthetic


def _show(fig):
    st.pyplot(fig, use_container_width=True)
    matplotlib.pyplot.close(fig)


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
lang = st.sidebar.selectbox("Language / Язык", ["en", "ru"],
                            format_func=lambda c: {"en": "English",
                                                   "ru": "Русский"}[c])
ui = UI[lang]
set_language(lang)
use_theme("light")

st.sidebar.header(ui["settings"])
ticker = st.sidebar.text_input(ui["ticker"], value="SPY").strip().upper()
provider = st.sidebar.selectbox(ui["provider"], ["yfinance", "synthetic"])
start = st.sidebar.text_input(ui["history"], value="2016-01-01")
max_expiries = st.sidebar.slider(ui["expiries"], 4, 20, 12)
walk_forward = st.sidebar.checkbox(ui["walk"], value=False)
snapshot = st.sidebar.checkbox(ui["snapshot"], value=False)
go = st.sidebar.button(ui["run"], type="primary", use_container_width=True)

st.title(ui["title"])
st.caption(ui["tagline"])

signature = (ticker, start, provider, max_expiries, lang, walk_forward, snapshot)
if go:
    st.session_state["signature"] = signature

if "signature" not in st.session_state:
    st.info(ui["welcome"])
    st.stop()

if st.session_state["signature"] != signature:
    st.warning(ui["stale"])

sig = st.session_state["signature"]
with st.spinner(ui["running"]):
    res = _run(*sig)

# --------------------------------------------------------------------------- #
# Headline
# --------------------------------------------------------------------------- #
head = res.headline()
if head.get("synthetic_data"):
    st.warning(ui["synthetic"])

cols = st.columns(5)
for col, key in zip(cols, ["spot", "atm_30d_iv", "vrp_vol_points",
                           "n_quotes", "n_anomalies"]):
    value = head.get(key)
    if value is None:
        continue
    if key == "atm_30d_iv":
        shown = f"{value * 100:.2f}%"
    elif key == "vrp_vol_points":
        shown = f"{value:+.2f}"
    elif isinstance(value, float):
        shown = f"{value:,.2f}"
    else:
        shown = f"{value:,}"
    col.metric(t(f"cli.{key}"), shown)

tabs = st.tabs([ui["tab_surface"], ui["tab_garch"], ui["tab_vrp"], ui["tab_screen"],
                ui["tab_risk"], ui["tab_backtest"], ui["tab_data"]])

# -- surface ---------------------------------------------------------------- #
with tabs[0]:
    if res.surface is None:
        st.error(ui["no_surface"])
    else:
        st.plotly_chart(P.plot_surface_3d(res.surface), use_container_width=True)
        _show(P.plot_surface_heatmap(res.surface))
        _show(P.plot_smile_grid(res.surface))
        _show(P.plot_risk_neutral_density(res.surface))
        st.dataframe(res.slice_table.round(4), use_container_width=True)

# -- garch ------------------------------------------------------------------ #
with tabs[1]:
    if res.best_fit is not None:
        _show(P.plot_conditional_vol(res.returns, res.best_fit))
        st.dataframe(res.model_table.round(4), use_container_width=True)
    if res.evaluation is not None and not res.evaluation.empty:
        _show(P.plot_model_scorecard(res.evaluation, "qlike"))
        st.dataframe(res.evaluation.round(4), use_container_width=True)

# -- premium ---------------------------------------------------------------- #
with tabs[2]:
    if res.term_structure is not None:
        _show(P.plot_term_structure(res.term_structure, res.garch_term_structure))
    if res.vrp is not None and not res.vrp.empty:
        _show(P.plot_vrp_term(res.vrp))
        st.dataframe(res.vrp.round(4), use_container_width=True)
    if res.vrp_history is not None and not res.vrp_history.empty:
        _show(P.plot_vrp_history(res.vrp_history))

# -- screen ----------------------------------------------------------------- #
with tabs[3]:
    if res.skew is not None:
        _show(P.plot_skew_term(res.skew))
    _show(P.plot_anomalies(res.anomalies))
    if res.anomalies is not None and len(res.anomalies):
        st.dataframe(res.anomalies, use_container_width=True)
    if res.history_zscores is not None and not res.history_zscores.empty:
        st.dataframe(res.history_zscores.round(3), use_container_width=True)

# -- portfolio risk --------------------------------------------------------- #
with tabs[4]:
    if res.surface is None:
        st.error(ui["no_surface"])
    else:
        st.caption(ui["risk_note"])
        upload = st.file_uploader(ui["upload"], type="csv")
        sticky = st.radio(t("pf.sticky_label"), ["moneyness", "strike"],
                          horizontal=True,
                          format_func=lambda m: t(f"pf.sticky_{m}"))
        try:
            book = (PF.Portfolio.from_records(pd.read_csv(upload).to_dict("records"))
                    if upload is not None else PF.example_portfolio(res.surface))
            report = PF.risk_report(book, res.surface, sticky=sticky)
            st.subheader(ui["totals"])
            totals = report["totals"]
            mcols = st.columns(4)
            for col, key in zip(mcols, ["delta_shares", "vega_per_vol_point",
                                        "theta_per_day", "value"]):
                col.metric(t(f"pf.{key}"), f"{totals.get(key, 0):,.0f}")
            st.dataframe(report["positions"].round(3), use_container_width=True)
            _show(P.plot_vega_ladder(report["vega_by_tenor"], by="tenor"))
            _show(P.plot_stress_grid(report["stress"]))
        except Exception as exc:
            st.error(f"{type(exc).__name__}: {exc}")

# -- backtest --------------------------------------------------------------- #
with tabs[5]:
    st.caption(ui["bt_note"])
    c1, c2, c3 = st.columns(3)
    holding = c1.slider(ui["bt_holding"], 5, 63, 21)
    vega_notional = c2.number_input(ui["bt_vega"], value=10_000, step=1_000)
    spread = c3.slider(ui["bt_spread"], 0.0, 2.0, 0.5, 0.1)
    try:
        iv_index, returns, synth = _vol_index(sig[0], sig[1], sig[2])
        if synth:
            st.warning(ui["synthetic"])
        costs = CostModel(option_spread_vol_points=spread)
        bt = short_straddle_backtest(iv_index, returns, holding_days=holding,
                                     vega_notional=float(vega_notional), costs=costs)
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Sharpe", f"{bt.stats.get('sharpe_annualised', float('nan')):.2f}")
        b2.metric("P&L", f"{bt.stats.get('total_pnl', 0):,.0f}")
        b3.metric("hit rate", f"{bt.stats.get('hit_rate_pct', 0):.0f}%")
        b4.metric("max DD", f"{bt.stats.get('max_drawdown', 0):,.0f}")
        _show(P.plot_equity_curve(bt))
        _show(P.plot_trade_distribution(bt))

        z = ((iv_index - iv_index.rolling(252).mean())
             / iv_index.rolling(252).std())
        table = compare_strategies(iv_index, returns, vrp_zscore=z,
                                   holding_days=holding,
                                   vega_notional=float(vega_notional), costs=costs)
        _show(P.plot_strategy_comparison(table))
        st.dataframe(table.round(2), use_container_width=True)
    except Exception as exc:
        st.error(f"{type(exc).__name__}: {exc}")

# -- data ------------------------------------------------------------------- #
with tabs[6]:
    st.subheader("status")
    st.json({"status": res.status, "errors": res.errors})
    if res.funnel is not None:
        _show(P.plot_quote_funnel(res.funnel))
        st.dataframe(res.funnel, use_container_width=True)
    if res.forwards is not None:
        st.dataframe(res.forwards.round(4), use_container_width=True)
    if res.quotes is not None:
        st.download_button("quotes.csv", res.quotes.to_csv(index=False),
                           file_name=f"{sig[0]}_quotes.csv", mime="text/csv")
