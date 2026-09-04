"""Localisation: catalogue integrity, fallbacks, and that charts still render."""
from __future__ import annotations

import re

import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")

from volsurface import i18n
from volsurface.config import OptionsConfig, SurfaceConfig
from volsurface.data.clean import prepare_quotes
from volsurface.data.synthetic import synthetic_option_chain
from volsurface.models.surface import build_surface
from volsurface.viz import plots as P

SPOT, ASOF = 450.0, pd.Timestamp("2026-09-02")

# Entries that stay Latin in every language on purpose: units, acronyms, and
# the market jargon Russian traders themselves use untranslated.
NOT_TRANSLATABLE = {
    "axis.iv_pct_short",   # "IV (%)"
    "surface.colorbar",    # "IV (%)"
    "term.lbl_garch",      # "GARCH"
    "skew.lbl_rr",         # "risk reversal" — said in English on a Russian desk
    "anom.z",              # "z = ..." — a statistical symbol
    "axis.log_moneyness_short",   # "log(K/F)" — a formula, not prose
    "cli.stage_ok",        # "OK"
}


@pytest.fixture(autouse=True)
def _reset_language():
    """Never let one test's language leak into the next."""
    yield
    i18n.set_language("en")


@pytest.fixture(scope="module")
def surface():
    chain = synthetic_option_chain(SPOT, asof=ASOF)
    cfg = OptionsConfig()
    quotes, forwards, _ = prepare_quotes(chain, cfg, spot=SPOT, asof=ASOF)
    return build_surface(quotes, forwards, SPOT, SurfaceConfig(), asof=ASOF,
                         risk_free_rate=cfg.risk_free_rate)


# --------------------------------------------------------------------------- #
# Catalogue integrity
# --------------------------------------------------------------------------- #
def test_every_language_covers_every_key():
    """A partially translated language silently ships English text."""
    assert i18n.coverage() == {lang: 100.0 for lang in i18n.LANGUAGES}


def test_no_language_has_stray_keys():
    """A key only in the translation is a typo that will never be used."""
    base = set(i18n.STRINGS["en"])
    for lang, strings in i18n.STRINGS.items():
        assert set(strings) - base == set(), f"{lang} has keys English does not"


def test_placeholders_match_across_languages():
    """
    A translation whose placeholders differ from English raises at .format()
    time — inside a chart, mid-run. Catch it here instead.
    """
    ph = lambda s: set(re.findall(r"\{(\w+)[^}]*\}", s))  # noqa: E731
    for key, english in i18n.STRINGS["en"].items():
        for lang in i18n.LANGUAGES:
            if lang == "en":
                continue
            assert ph(i18n.STRINGS[lang][key]) == ph(english), \
                f"{lang}:{key} placeholders differ from English"


def test_russian_is_actually_russian():
    """Guard against an untranslated entry copy-pasted from English."""
    cyrillic = re.compile(r"[а-яА-Я]")
    for key, value in i18n.STRINGS["ru"].items():
        # Some entries are legitimately symbols, acronyms or market jargon that
        # Russian desks use untranslated ("risk reversal", "GARCH", "IV").
        if key in NOT_TRANSLATABLE:
            continue
        assert cyrillic.search(value), f"ru:{key} looks untranslated: {value!r}"


def test_the_russian_check_is_not_vacuous():
    """
    The test above is only meaningful if it inspects most of the catalogue.
    If NOT_TRANSLATABLE ever grew to cover everything, it would pass trivially.
    """
    checked = len(i18n.STRINGS["ru"]) - len(NOT_TRANSLATABLE)
    assert checked > 0.9 * len(i18n.STRINGS["ru"])


# --------------------------------------------------------------------------- #
# Behaviour
# --------------------------------------------------------------------------- #
def test_default_is_english():
    assert i18n.get_language() == "en"
    assert i18n.t("surface.title") == "Implied volatility surface"


def test_switching_language_changes_output():
    i18n.set_language("ru")
    assert i18n.t("surface.title") == "Поверхность подразумеваемой волатильности"
    i18n.set_language("en")
    assert i18n.t("surface.title") == "Implied volatility surface"


def test_unknown_language_falls_back_to_english():
    assert i18n.set_language("klingon") == "en"
    assert i18n.t("surface.title") == "Implied volatility surface"


def test_locale_variants_are_accepted():
    assert i18n.set_language("ru-RU") == "ru"
    assert i18n.set_language("RU") == "ru"


def test_missing_key_returns_the_key_not_an_exception():
    assert i18n.t("no.such.key") == "no.such.key"


def test_bad_format_arguments_do_not_raise():
    """A chart must never die because a caller forgot a placeholder."""
    out = i18n.t("smile.fit_rmse")            # both placeholders missing
    assert isinstance(out, str) and out


def test_interpolation_works_in_both_languages():
    for lang, needle in (("en", "quotes"), ("ru", "котировок")):
        i18n.set_language(lang)
        assert needle in i18n.t("smile.fit_rmse", rmse=0.12, n=33)


# --------------------------------------------------------------------------- #
# Charts actually render in Russian
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("lang", ["en", "ru"])
def test_core_charts_render_in_every_language(surface, lang, tmp_path):
    import matplotlib.pyplot as plt

    i18n.set_language(lang)
    for name, fn in (("heatmap", P.plot_surface_heatmap),
                     ("smiles", P.plot_smile_grid),
                     ("overlay", P.plot_smile_overlay),
                     ("density", P.plot_risk_neutral_density),
                     ("residuals", P.plot_fit_residuals)):
        fig = fn(surface, path=tmp_path / f"{name}_{lang}.png")
        assert (tmp_path / f"{name}_{lang}.png").exists()
        plt.close(fig)


def test_russian_titles_reach_the_figure(surface):
    """Not just "it rendered" — the Russian string is really on the axes."""
    import matplotlib.pyplot as plt

    i18n.set_language("ru")
    fig = P.plot_surface_heatmap(surface)
    # The house style sets a left-aligned title, and get_title() defaults to
    # the centre one — which is empty here.
    title = fig.axes[0].get_title(loc="left")
    assert "Поверхность" in title, title
    plt.close(fig)


def test_data_columns_stay_english_under_translation(surface):
    """
    Translating the display must never rename data. A downstream script reading
    `vrp_vol_points` has to keep working with --lang ru.
    """
    from volsurface.analytics.skew import skew_metrics

    i18n.set_language("ru")
    cols = set(skew_metrics(surface).columns)
    assert {"days", "T", "forward", "atm_iv", "risk_reversal"} <= cols
    assert not any(re.search(r"[а-яА-Я]", c) for c in cols)


def test_vrp_signal_is_a_stable_code_not_prose():
    """The signal travels as a code so it survives translation and filtering."""
    from volsurface.analytics.vrp import _vrp_signal, vrp_signal_text

    assert _vrp_signal(0.20, 0.10) == "rich"
    assert _vrp_signal(0.10, 0.20) == "cheap"
    assert _vrp_signal(0.20, 0.20) == "in_line"
    assert _vrp_signal(float("nan"), 0.2) == "na"

    i18n.set_language("ru")
    assert "рынок дороже" in vrp_signal_text("rich")
    i18n.set_language("en")
    assert "rich" in vrp_signal_text("rich")


def test_config_language_applies_without_rendering_figures():
    """
    Regression: cfg.language used to be applied only inside the figure step, so
    a caller who set it and passed make_figures=False silently got English
    charts when he plotted them himself.
    """
    from volsurface import Config, run_pipeline

    cfg = Config()
    cfg.data.provider = "synthetic"
    cfg.data.use_cache = False
    cfg.data.start = "2022-01-01"
    cfg.language = "ru"
    cfg.garch.specs = (("GARCH(1,1)-t", "Garch", 1, 0, 1, "t"),)

    run_pipeline(cfg, make_figures=False, run_walk_forward=False)
    assert i18n.get_language() == "ru"
    assert i18n.t("surface.title").startswith("Поверхность")
