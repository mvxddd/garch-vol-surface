"""
Localisation for everything a user actually reads: chart text and CLI output.

Design notes
------------
* **Keys are semantic**, not English source strings (`surface.title`, not
  "Implied volatility surface"). Editing the English wording then never
  silently orphans a translation.
* **English is the fallback and the default.** A missing or malformed
  translation degrades to English with a warning rather than crashing a chart
  half-way through a run — losing a subtitle is annoying, losing the figure is
  not acceptable.
* **Only presentation is translated.** Column names in the CSV outputs, log
  messages, exception text and the code itself stay English, because those are
  data and developer surfaces: a downstream script that reads
  `vrp_vol_points` must not break when someone passes `--lang ru`.
* Financial vocabulary follows Russian market usage — практики говорят
  «волатильность», «страйк», «экспирация», «улыбка», «скью», а не кальки.

Usage
-----
    from volsurface.i18n import set_language, t
    set_language("ru")
    t("surface.title")                    -> "Поверхность подразумеваемой волатильности"
    t("smile.fit_rmse", rmse=0.12, n=33)  -> "ошибка подгонки 0.12 п.в.\\n33 котировок"
"""
from __future__ import annotations

from .utils import get_logger

LOG = get_logger("volsurface.i18n")

DEFAULT_LANGUAGE = "en"
_ACTIVE = {"lang": DEFAULT_LANGUAGE}


# --------------------------------------------------------------------------- #
# Catalogue
# --------------------------------------------------------------------------- #
STRINGS: dict[str, dict[str, str]] = {
    "en": {
        # -- shared axis labels & units --
        "axis.log_moneyness": "log-moneyness  log(K/F)",
        "axis.days_to_expiry": "days to expiry",
        "axis.tenor": "tenor",
        "axis.implied_vol_pct": "implied vol (%)",
        "axis.ann_vol_pct": "annualised vol (%)",
        "axis.vol_points": "vol points",
        "axis.return_pct": "return (%)",
        "axis.iv_pct_short": "IV (%)",
        "axis.density": "density",
        "axis.quotes": "quotes",
        "axis.per_log_moneyness": "per unit log-moneyness",
        # Short form for small-multiple panels, where the full label does not fit.
        "axis.log_moneyness_short": "log(K/F)",
        "unit.days_suffix": "d",
        "label.market_quotes": "market quotes",
        "label.forward_atm": "forward ATM",

        # -- 3D surface / heatmap --
        "surface.title": "Implied volatility surface",
        "surface.title_dated": "Implied volatility surface · {asof}",
        "surface.subtitle_3d": "{n} calibrated expiries ({method}) · forward-ATM at "
                               "log-moneyness 0 · drag to rotate",
        "surface.subtitle_heat": "{asof} · dots are the quotes the surface was fitted "
                                 "to · blank corners are strikes the market does not "
                                 "list at that tenor",
        "surface.hover": "log-moneyness %{x:.3f}<br>%{y:.0f} days<br><b>IV %{z:.2f}%</b>"
                         "<extra></extra>",
        "surface.trace_fitted": "fitted surface",
        "surface.colorbar": "IV (%)",

        # -- smiles --
        "smile.grid_title": "Volatility smile by expiry: market quotes vs calibrated fit",
        "smile.market_mid": "market mid",
        "smile.fitted": "fitted smile",
        "smile.bidask": "bid/ask in vol",
        "smile.fit_rmse": "fit RMSE {rmse:.2f} vp\n{n} quotes",
        "smile.panel_title": "{days}d   ATM {atm:.1f}%",
        "smile.overlay_title": "Volatility smile across the term structure",
        "smile.overlay_subtitle": "shaded light to dark by maturity · the downward "
                                  "slope to the left is the equity crash skew",

        # -- term structure --
        "term.title": "Volatility term structure: market vs model",
        "term.subtitle": "shaded band = volatility risk premium (implied above model)",
        "term.atm_implied": "ATM implied vol",
        "term.fwd_vol": "implied forward vol (between tenors)",
        "term.garch": "GARCH forecast vol",
        "term.lbl_implied": "implied",
        "term.lbl_forward": "forward",
        "term.lbl_garch": "GARCH",

        # -- skew --
        "skew.title": "Skew and convexity by tenor",
        "skew.subtitle": "positive risk reversal = puts bid over calls, the equity "
                         "crash skew",
        "skew.rr": "25Δ risk reversal (put − call)",
        "skew.bf": "25Δ butterfly (wings − ATM)",
        "skew.lbl_rr": "risk reversal",
        "skew.lbl_bf": "butterfly",
        "skew.atm_title": "ATM skew  ∂IV/∂log(K/F)",
        "skew.atm_subtitle": "flattens with maturity — the standard 1/√T decay",
        "skew.lbl_atm": "ATM slope",

        # -- density --
        "density.title": "Implied risk-neutral density by tenor",
        "density.subtitle": "any excursion below zero would be a butterfly arbitrage · "
                            "the fat left tail is the skew in probability space",

        # -- GARCH --
        "garch.returns_title": "Daily log returns",
        "garch.returns_subtitle": "volatility clustering — calm and violent periods "
                                  "arrive in runs",
        "garch.cond_title": "Conditional volatility vs realised",
        "garch.cond_subtitle": "persistence {persistence:.3f} — shocks decay with a "
                               "half-life of {halflife:.0f} days",
        "garch.realized": "realised vol ({window}d trailing)",
        "garch.conditional": "{model} conditional vol",
        "garch.long_run": "long-run vol {vol:.1f}%",
        "garch.oos_marker": "  out-of-sample from here",
        "garch.oos_title": "Out-of-sample volatility forecast vs realised · {model}",
        "garch.horizon_panel": "{h}-day horizon",
        "garch.realized_next": "realised (next {h}d)",
        "garch.forecast": "forecast",
        "garch.corr": "corr {corr:.2f}",

        # -- scorecard --
        "score.title": "Out-of-sample {metric}: gap to the best model",
        "score.subtitle": "shorter bar = better · raw {metric} printed on each bar · "
                          "naive benchmarks included so the GARCH models have to earn "
                          "their place",
        "score.gap": "gap to best",
        "score.gap_log": "gap to best (log scale)",

        # -- VRP --
        "vrp.term_title": "Volatility risk premium: implied minus GARCH forecast",
        "vrp.term_subtitle": "positive = the market charges more for volatility than "
                             "the model forecasts (the normal state)",
        "vrp.tick": "{days}d\nimplied {iv:.1f}% / model {g:.1f}%",
        "vrp.hist_title": "Implied volatility vs what actually happened",
        "vrp.hist_subtitle": "the wedge between the two lines is the premium option "
                             "sellers collect",
        "vrp.implied": "implied vol (option market, ex ante)",
        "vrp.realized_next": "realised vol over the next {h}d (ex post)",
        "vrp.premium_title": "Volatility risk premium",
        "vrp.premium_subtitle": "positive {pct:.0f}% of the time — sellers win often "
                                "and lose big, which is why the premium exists",
        "vrp.mean": "mean {v:+.2f} vp",
        "vrp.signal.rich": "implied rich vs model (short-vol bias)",
        "vrp.signal.cheap": "implied cheap vs model (long-vol bias)",
        "vrp.signal.in_line": "in line",
        "vrp.signal.na": "n/a",

        # -- data quality --
        "funnel.title": "Quote-quality funnel",
        "funnel.subtitle": "how many raw option quotes survive each filter on the way "
                           "to the surface",
        "resid.title": "Calibration residuals: market minus model",
        "resid.subtitle": "RMSE {rmse:.2f} vol points · residuals inside the grey band "
                          "are smaller than the cost of crossing the spread",
        "resid.halfspread": "market half-spread (untradeable zone)",
        "resid.above": "market above model",
        "resid.below": "market below model",

        # -- anomalies --
        "anom.title": "Relative-value screen",
        "anom.subtitle": "ranked by severity then magnitude · every flag needs a human "
                         "check for a stale quote or a scheduled event before it is a "
                         "trade",
        "anom.axis": "|z-score| vs the cross-sectional benchmark",
        "anom.none_title": "No anomalies flagged",
        "anom.none_body": "The surface is internally consistent: no calendar or "
                          "butterfly violations, no tenor out of line with its "
                          "neighbours.",
        "anom.hard": "  {sev}  ·  hard violation",
        "anom.z": "  {sev}  ·  z = {z:+.1f}",
        "sev.high": "HIGH",
        "sev.medium": "MEDIUM",
        "sev.low": "LOW",

        # -- CLI --
        "cli.header": "{ticker} — volatility study",
        "cli.stage_ok": "OK  ",
        "cli.stage_fail": "FAIL",
        "cli.stage_skip": "SKIP",
        "cli.ticker": "ticker",
        "cli.asof": "as of",
        "cli.spot": "spot",
        "cli.synthetic_data": "synthetic data",
        "cli.best_model": "best model",
        "cli.persistence": "persistence",
        "cli.long_run_vol": "long-run vol",
        "cli.best_oos_model": "best out-of-sample",
        "cli.best_oos_qlike": "best QLIKE",
        "cli.n_expiries": "expiries",
        "cli.n_quotes": "clean quotes",
        "cli.atm_30d_iv": "30d ATM implied vol",
        "cli.calendar_arbitrage": "calendar arbitrage",
        "cli.vrp_vol_points": "vol risk premium (vp)",
        "cli.vrp_signal": "signal",
        "cli.n_anomalies": "anomalies flagged",
        "cli.yes": "yes",
        "cli.no": "no",
    },

    "ru": {
        # -- общие подписи осей и единицы --
        "axis.log_moneyness": "лог-манинес  log(K/F)",
        "axis.days_to_expiry": "дней до экспирации",
        "axis.tenor": "срок",
        "axis.implied_vol_pct": "подразумеваемая волатильность (%)",
        "axis.ann_vol_pct": "волатильность, годовых (%)",
        "axis.vol_points": "пункты волатильности",
        "axis.return_pct": "доходность (%)",
        "axis.iv_pct_short": "IV (%)",
        "axis.density": "плотность",
        "axis.quotes": "котировки",
        "axis.per_log_moneyness": "на единицу лог-манинес",
        # Короткая форма для мелких панелей, где полная подпись не помещается.
        "axis.log_moneyness_short": "log(K/F)",
        "unit.days_suffix": "д",
        "label.market_quotes": "котировки рынка",
        "label.forward_atm": "ATM по форварду",

        # -- поверхность --
        "surface.title": "Поверхность подразумеваемой волатильности",
        "surface.title_dated": "Поверхность подразумеваемой волатильности · {asof}",
        "surface.subtitle_3d": "{n} откалиброванных экспираций ({method}) · ATM по "
                               "форварду при лог-манинес 0 · вращайте мышью",
        "surface.subtitle_heat": "{asof} · точки — котировки, по которым построена "
                                 "поверхность · пустые углы — страйки, которых рынок "
                                 "на этом сроке не котирует",
        "surface.hover": "лог-манинес %{x:.3f}<br>%{y:.0f} дней<br><b>IV %{z:.2f}%</b>"
                         "<extra></extra>",
        "surface.trace_fitted": "подогнанная поверхность",
        "surface.colorbar": "IV (%)",

        # -- улыбки --
        "smile.grid_title": "Улыбка волатильности по экспирациям: рынок и калибровка",
        "smile.market_mid": "середина рынка",
        "smile.fitted": "подогнанная улыбка",
        "smile.bidask": "спред в волатильности",
        "smile.fit_rmse": "ошибка подгонки {rmse:.2f} п.в.\n{n} котировок",
        "smile.panel_title": "{days}д   ATM {atm:.1f}%",
        "smile.overlay_title": "Улыбка волатильности по всей кривой сроков",
        "smile.overlay_subtitle": "от светлого к тёмному по мере роста срока · наклон "
                                  "влево — это скью, страх обвала",

        # -- кривая сроков --
        "term.title": "Временная структура волатильности: рынок и модель",
        "term.subtitle": "закрашенная область — премия за риск волатильности "
                         "(рынок выше модели)",
        "term.atm_implied": "подразумеваемая волатильность ATM",
        "term.fwd_vol": "форвардная волатильность (между сроками)",
        "term.garch": "прогноз GARCH",
        "term.lbl_implied": "рынок",
        "term.lbl_forward": "форвард",
        "term.lbl_garch": "GARCH",

        # -- скью --
        "skew.title": "Скью и выпуклость по срокам",
        "skew.subtitle": "положительный risk reversal = путы дороже коллов, то есть "
                         "рынок платит за страх обвала",
        "skew.rr": "25Δ risk reversal (пут − колл)",
        "skew.bf": "25Δ бабочка (крылья − ATM)",
        "skew.lbl_rr": "risk reversal",
        "skew.lbl_bf": "бабочка",
        "skew.atm_title": "Наклон скью на ATM  ∂IV/∂log(K/F)",
        "skew.atm_subtitle": "выполаживается с ростом срока — классический спад 1/√T",
        "skew.lbl_atm": "наклон ATM",

        # -- плотность --
        "density.title": "Подразумеваемая риск-нейтральная плотность по срокам",
        "density.subtitle": "любой заход ниже нуля — это арбитраж на бабочке · "
                            "тяжёлый левый хвост — тот же скью, но в вероятностях",

        # -- GARCH --
        "garch.returns_title": "Дневные логарифмические доходности",
        "garch.returns_subtitle": "кластеризация волатильности — спокойные и бурные "
                                  "периоды идут полосами",
        "garch.cond_title": "Условная волатильность против реализованной",
        "garch.cond_subtitle": "персистентность {persistence:.3f} — шок затухает "
                               "наполовину за {halflife:.0f} дней",
        "garch.realized": "реализованная волатильность (скользящие {window}д)",
        "garch.conditional": "условная волатильность, {model}",
        "garch.long_run": "долгосрочная волатильность {vol:.1f}%",
        "garch.oos_marker": "  отсюда вне выборки",
        "garch.oos_title": "Прогноз вне выборки против реализованной · {model}",
        "garch.horizon_panel": "горизонт {h} дн.",
        "garch.realized_next": "реализованная (следующие {h}д)",
        "garch.forecast": "прогноз",
        "garch.corr": "корр. {corr:.2f}",

        # -- сравнение моделей --
        "score.title": "{metric} вне выборки: отставание от лучшей модели",
        "score.subtitle": "короче столбик — лучше · на каждом столбике исходное "
                          "значение {metric} · наивные бенчмарки включены, чтобы "
                          "модели GARCH доказали свою полезность",
        "score.gap": "отставание от лучшей",
        "score.gap_log": "отставание от лучшей (лог. шкала)",

        # -- премия за риск --
        "vrp.term_title": "Премия за риск волатильности: рынок минус прогноз GARCH",
        "vrp.term_subtitle": "положительная = рынок берёт за волатильность больше, "
                             "чем прогнозирует модель (обычное состояние)",
        "vrp.tick": "{days}д\nрынок {iv:.1f}% / модель {g:.1f}%",
        "vrp.hist_title": "Подразумеваемая волатильность против того, что случилось",
        "vrp.hist_subtitle": "зазор между линиями — это премия, которую собирает "
                             "продавец опционов",
        "vrp.implied": "подразумеваемая (рынок опционов, до факта)",
        "vrp.realized_next": "реализованная за следующие {h}д (по факту)",
        "vrp.premium_title": "Премия за риск волатильности",
        "vrp.premium_subtitle": "положительна в {pct:.0f}% случаев — продавец часто "
                                "выигрывает понемногу и редко проигрывает много, "
                                "в этом и смысл премии",
        "vrp.mean": "среднее {v:+.2f} п.в.",
        "vrp.signal.rich": "рынок дороже модели (уклон в продажу волатильности)",
        "vrp.signal.cheap": "рынок дешевле модели (уклон в покупку волатильности)",
        "vrp.signal.in_line": "в норме",
        "vrp.signal.na": "нет данных",

        # -- качество данных --
        "funnel.title": "Воронка качества котировок",
        "funnel.subtitle": "сколько сырых котировок опционов переживает каждый фильтр "
                           "на пути к поверхности",
        "resid.title": "Остатки калибровки: рынок минус модель",
        "resid.subtitle": "RMSE {rmse:.2f} пункта волатильности · остатки внутри "
                          "серой полосы меньше, чем стоимость пересечения спреда",
        "resid.halfspread": "половина спреда (здесь торговать нельзя)",
        "resid.above": "рынок выше модели",
        "resid.below": "рынок ниже модели",

        # -- аномалии --
        "anom.title": "Скрин относительной стоимости",
        "anom.subtitle": "по убыванию серьёзности и величины · каждую находку нужно "
                         "проверить руками на устаревшую котировку или событие в "
                         "календаре, прежде чем считать её сделкой",
        "anom.axis": "|z-оценка| относительно кросс-секционного ориентира",
        "anom.none_title": "Аномалий не найдено",
        "anom.none_body": "Поверхность внутренне непротиворечива: нет нарушений по "
                          "календарю и бабочке, ни один срок не выбивается из ряда "
                          "соседних.",
        "anom.hard": "  {sev}  ·  явное нарушение",
        "anom.z": "  {sev}  ·  z = {z:+.1f}",
        "sev.high": "ВЫСОКАЯ",
        "sev.medium": "СРЕДНЯЯ",
        "sev.low": "НИЗКАЯ",

        # -- CLI --
        "cli.header": "{ticker} — исследование волатильности",
        "cli.stage_ok": "OK  ",
        "cli.stage_fail": "СБОЙ",
        "cli.stage_skip": "ПРОП",
        "cli.ticker": "тикер",
        "cli.asof": "на дату",
        "cli.spot": "спот",
        "cli.synthetic_data": "синтетические данные",
        "cli.best_model": "лучшая модель",
        "cli.persistence": "персистентность",
        "cli.long_run_vol": "долгосрочная волатильность",
        "cli.best_oos_model": "лучшая вне выборки",
        "cli.best_oos_qlike": "лучший QLIKE",
        "cli.n_expiries": "экспираций",
        "cli.n_quotes": "чистых котировок",
        "cli.atm_30d_iv": "IV ATM 30 дней",
        "cli.calendar_arbitrage": "календарный арбитраж",
        "cli.vrp_vol_points": "премия за риск (п.в.)",
        "cli.vrp_signal": "сигнал",
        "cli.n_anomalies": "найдено аномалий",
        "cli.yes": "да",
        "cli.no": "нет",
    },
}

LANGUAGES = tuple(STRINGS)


# --------------------------------------------------------------------------- #
# API
# --------------------------------------------------------------------------- #
def set_language(lang: str) -> str:
    """Set the active language. Unknown codes fall back to English with a warning."""
    code = (lang or DEFAULT_LANGUAGE).lower().split("-")[0]
    if code not in STRINGS:
        LOG.warning("Unknown language %r — falling back to %r (available: %s)",
                    lang, DEFAULT_LANGUAGE, ", ".join(LANGUAGES))
        code = DEFAULT_LANGUAGE
    _ACTIVE["lang"] = code
    return code


def get_language() -> str:
    """The language currently applied to charts and CLI output."""
    return _ACTIVE["lang"]


def t(key: str, **fmt) -> str:
    """
    Translate `key` into the active language and interpolate `fmt`.

    Degrades rather than raises, in three steps: a missing key falls back to
    English, then to the key itself; a formatting error returns the unformatted
    template. A chart with one odd label is recoverable — a chart that raised
    half-way through rendering is not.
    """
    lang = _ACTIVE["lang"]
    template = STRINGS.get(lang, {}).get(key)
    if template is None:
        template = STRINGS[DEFAULT_LANGUAGE].get(key)
        if template is None:
            LOG.warning("Missing translation key %r", key)
            return key
        if lang != DEFAULT_LANGUAGE:
            LOG.warning("Key %r not translated into %r — using English", key, lang)
    if not fmt:
        return template
    try:
        return template.format(**fmt)
    except (KeyError, IndexError, ValueError) as exc:
        LOG.warning("Could not format %r (%s) — returning the raw template", key, exc)
        return template


def coverage() -> dict[str, float]:
    """Share of the English keys present in each language (asserted in the tests)."""
    base = set(STRINGS[DEFAULT_LANGUAGE])
    return {lang: round(len(base & set(keys)) / len(base) * 100, 1)
            for lang, keys in STRINGS.items()}
