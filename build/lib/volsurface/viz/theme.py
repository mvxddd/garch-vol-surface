"""
Chart theme: one validated palette, applied to matplotlib and plotly alike.

Every colour here comes from a palette that was run through a colour-vision-
deficiency validator rather than chosen by eye. The checks that matter:

* categorical slots (used for model / metric identity) clear the adjacent-pair
  CVD separation and normal-vision floors in **both** light and dark modes;
* the tenor ramp is a single-hue ordinal ramp with visible lightness steps, so
  "longer maturity = darker" survives greyscale printing;
* three light-mode slots sit below 3:1 contrast on the light surface, so every
  chart in this project ships a legend *and* direct labels — identity is never
  carried by colour alone.

Encoding rules applied throughout `viz.plots`:
    identity (models, metrics)   -> categorical slots, assigned in fixed order
    magnitude (IV, maturity)     -> single-hue sequential / ordinal ramp
    polarity (VRP, residuals)    -> diverging blue<->red with a grey midpoint
    state (anomaly severity)     -> reserved status colours, always with a label
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Theme:
    """A complete, validated colour scheme for one mode."""

    name: str
    surface: str
    page: str
    ink: str
    ink_secondary: str
    ink_muted: str
    grid: str
    axis: str
    # Identity — fixed order, never cycled. A 7th series folds into "Other".
    categorical: tuple[str, ...]
    # Magnitude — single hue, light -> dark.
    sequential: tuple[str, ...]
    # Ordinal (discrete ordered marks, e.g. tenors): wider lightness steps.
    ordinal: tuple[str, ...]
    # Polarity.
    diverging: tuple[str, str, str]          # (negative, midpoint, positive)
    # State — reserved, never reused as a series colour.
    status: dict[str, str] = field(default_factory=dict)


LIGHT = Theme(
    name="light",
    surface="#fcfcfb", page="#f9f9f7",
    ink="#0b0b0b", ink_secondary="#52514e", ink_muted="#898781",
    grid="#e1e0d9", axis="#c3c2b7",
    categorical=("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"),
    sequential=("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6",
                "#256abf", "#1c5cab", "#104281", "#0d366b"),
    ordinal=("#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"),
    diverging=("#d03b3b", "#f0efec", "#2a78d6"),
    status={"good": "#0ca30c", "warning": "#fab219",
            "serious": "#ec835a", "critical": "#d03b3b"},
)

DARK = Theme(
    name="dark",
    surface="#1a1a19", page="#0d0d0d",
    ink="#ffffff", ink_secondary="#c3c2b7", ink_muted="#898781",
    grid="#2c2c2a", axis="#383835",
    categorical=("#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300"),
    sequential=("#0d366b", "#104281", "#1c5cab", "#256abf", "#2a78d6",
                "#3987e5", "#6da7ec", "#9ec5f4", "#cde2fb"),
    ordinal=("#9ec5f4", "#5598e7", "#3987e5", "#256abf", "#104281"),
    diverging=("#e66767", "#383835", "#3987e5"),
    status={"good": "#0ca30c", "warning": "#fab219",
            "serious": "#ec835a", "critical": "#d03b3b"},
)

THEMES = {"light": LIGHT, "dark": DARK}
_ACTIVE = {"theme": LIGHT}

# System UI sans, with the fallbacks that actually exist inside a Colab image.
FONT_STACK = ["system-ui", "-apple-system", "Segoe UI", "Helvetica Neue",
              "DejaVu Sans", "Arial", "sans-serif"]


def active() -> Theme:
    """The theme currently applied to matplotlib."""
    return _ACTIVE["theme"]


def use(theme: str | Theme = "light") -> Theme:
    """
    Apply a theme to matplotlib's rcParams and remember it for plotly.

    Sets the recessive-chrome defaults once so no individual chart has to:
    hairline horizontal grid only, no top/right spines, muted tick labels,
    thin marks.
    """
    import matplotlib as mpl

    th = THEMES[theme] if isinstance(theme, str) else theme
    _ACTIVE["theme"] = th

    mpl.rcParams.update({
        "figure.facecolor": th.surface,
        "figure.edgecolor": th.surface,
        "savefig.facecolor": th.surface,
        "savefig.edgecolor": th.surface,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "axes.facecolor": th.surface,
        "axes.edgecolor": th.axis,
        "axes.labelcolor": th.ink_secondary,
        "axes.titlecolor": th.ink,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlepad": 10,
        "axes.labelsize": 10,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.prop_cycle": mpl.cycler(color=list(th.categorical)),
        "grid.color": th.grid,
        "grid.linewidth": 0.7,
        "grid.alpha": 1.0,
        "xtick.color": th.ink_muted,
        "ytick.color": th.ink_muted,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "text.color": th.ink,
        "font.family": "sans-serif",
        "font.sans-serif": FONT_STACK,
        "font.size": 10,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "legend.labelcolor": th.ink_secondary,
        "lines.linewidth": 2.0,
        "lines.markersize": 5,
        "lines.solid_capstyle": "round",
        "figure.autolayout": False,
    })
    return th


def plotly_layout(title: str = "", subtitle: str = "") -> dict:
    """Matching plotly layout dict (used by the interactive 3-D surface)."""
    th = active()
    head = f"<b>{title}</b>"
    if subtitle:
        head += f'<br><span style="font-size:12px;color:{th.ink_secondary}">{subtitle}</span>'
    return {
        "template": "plotly_white" if th.name == "light" else "plotly_dark",
        "paper_bgcolor": th.surface,
        "plot_bgcolor": th.surface,
        "font": {"family": ", ".join(FONT_STACK), "color": th.ink, "size": 12},
        "title": {"text": head, "x": 0.02, "xanchor": "left",
                  "font": {"size": 17, "color": th.ink}},
        "margin": {"l": 10, "r": 10, "t": 90, "b": 10},
        "hoverlabel": {"font": {"family": ", ".join(FONT_STACK), "size": 12}},
    }


def plotly_colorscale() -> list[list]:
    """Sequential blue ramp as a plotly colorscale (magnitude encoding)."""
    ramp = active().sequential
    return [[i / (len(ramp) - 1), c] for i, c in enumerate(ramp)]


def tenor_colors(n: int) -> list[str]:
    """
    `n` colours from the ordinal tenor ramp (short = light, long = dark).

    Past 5 tenors the ramp's lightness steps stop being distinguishable, so
    charts that need more use small multiples instead of more colours — see
    `plot_smile_grid`.
    """
    ramp = active().ordinal
    if n <= 1:
        return [ramp[len(ramp) // 2]]
    if n <= len(ramp):
        idx = [round(i * (len(ramp) - 1) / (n - 1)) for i in range(n)]
        return [ramp[i] for i in idx]
    seq = active().sequential
    idx = [round(i * (len(seq) - 1) / (n - 1)) for i in range(n)]
    return [seq[i] for i in idx]


def severity_color(severity: str) -> str:
    """Reserved status colour for an anomaly severity (always shown with text)."""
    return active().status.get(
        {"high": "critical", "medium": "serious", "low": "warning"}.get(
            str(severity).lower(), "warning"), "#fab219")
