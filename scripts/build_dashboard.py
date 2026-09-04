#!/usr/bin/env python3
"""
Assemble one browsable page from a completed run.

    python scripts/build_dashboard.py --out outputs --lang ru
    open outputs/figures/index.html

The pipeline already writes 15 figures and a JSON run report; this stitches
them into a single self-contained page so a run can be *looked at* rather than
opened file by file. Everything is inlined (figures as data URIs) so the page
can be emailed or dropped in Slack and still render.

It reads whatever the run actually produced — a partial run (say, the option
feed was down) yields a page with the sections that exist and an explicit note
about the ones that do not, rather than broken images.
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from volsurface.i18n import set_language, t  # noqa: E402

# Figure order, and which i18n key titles each panel.
PANELS: list[tuple[str, str]] = [
    ("iv_surface_heatmap", "surface.title"),
    ("smile_grid", "smile.grid_title"),
    ("smile_overlay", "smile.overlay_title"),
    ("risk_neutral_density", "density.title"),
    ("fit_residuals", "resid.title"),
    ("term_structure", "term.title"),
    ("skew_term", "skew.title"),
    ("vrp_term", "vrp.term_title"),
    ("vrp_history", "vrp.hist_title"),
    ("conditional_vol", "garch.cond_title"),
    ("forecast_vs_realized", "garch.oos_title"),
    ("model_scorecard", "score.title"),
    ("quote_funnel", "funnel.title"),
    ("anomalies", "anom.title"),
]

# Headline fields worth promoting to stat tiles, in display order.
TILES = ["spot", "atm_30d_iv", "vrp_vol_points", "n_quotes", "n_expiries",
         "n_anomalies"]

UI = {
    "en": {"subtitle": "Volatility study", "interactive": "Interactive 3-D surface",
           "open": "Open in a new tab", "missing": "not produced by this run",
           "generated": "generated", "synthetic": "SYNTHETIC DATA — not a real market",
           "report": "Run report", "stages": "Pipeline stages"},
    "ru": {"subtitle": "Исследование волатильности",
           "interactive": "Интерактивная 3D-поверхность",
           "open": "Открыть в новой вкладке", "missing": "в этом прогоне не создан",
           "generated": "собрано", "synthetic": "СИНТЕТИЧЕСКИЕ ДАННЫЕ — не рынок",
           "report": "Отчёт о прогоне", "stages": "Стадии пайплайна"},
}


def _data_uri(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def build(out_dir: Path, lang: str) -> Path:
    set_language(lang)
    ui = UI.get(lang, UI["en"])
    fig_dir = out_dir / "figures"
    report_path = out_dir / "reports" / "run_report.json"
    if not fig_dir.is_dir():
        raise SystemExit(f"No figures under {fig_dir} — run the pipeline first.")

    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    head = report.get("headline", {})
    status = report.get("status", {})

    tiles = []
    for key in TILES:
        if key not in head:
            continue
        value = head[key]
        if key in {"atm_30d_iv"} and isinstance(value, (int, float)):
            value = f"{value * 100:.2f}%"
        elif key == "vrp_vol_points":
            value = f"{value:+.2f}"
        elif isinstance(value, float):
            value = f"{value:,.2f}"
        elif isinstance(value, int):
            value = f"{value:,}"
        tiles.append(f'<div class="tile"><div class="k">{t(f"cli.{key}")}</div>'
                     f'<div class="v">{value}</div></div>')

    cards = []
    for name, title_key in PANELS:
        png = fig_dir / f"{name}.png"
        heading = t(title_key, metric="QLIKE", model="", asof="", n=0, method="",
                    h="", pct=0, persistence=0, halflife=0)
        if png.exists():
            # No figcaption: every figure already carries its own title, baked
            # in by the chart library. Repeating it above the image just prints
            # the same sentence twice.
            cards.append(f'<figure><img loading="lazy" alt="{heading}" '
                         f'src="{_data_uri(png)}"></figure>')
        else:
            cards.append(f'<figure class="absent"><figcaption>{heading}</figcaption>'
                         f'<p>{ui["missing"]}</p></figure>')

    surface_3d = fig_dir / "iv_surface_3d.html"
    interactive = ""
    if surface_3d.exists():
        interactive = (
            f'<section class="interactive"><h2>{ui["interactive"]}</h2>'
            f'<iframe src="iv_surface_3d.html" title="{ui["interactive"]}"></iframe>'
            f'<p><a href="iv_surface_3d.html" target="_blank">{ui["open"]} →</a></p>'
            f'</section>')

    stage_rows = "".join(
        f'<li class="{state}"><span>{stage}</span><b>{state}</b></li>'
        for stage, state in status.items())

    banner = ""
    if head.get("synthetic_data"):
        banner = f'<div class="banner">{ui["synthetic"]}</div>'

    ticker = head.get("ticker", "")
    asof = head.get("asof", "")
    html = f"""<!doctype html>
<html lang="{lang}"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{ticker} — {ui['subtitle']}</title>
<style>
:root{{--bg:#fbfbfc;--card:#fff;--ink:#10141c;--ink2:#4a5364;--ink3:#7b8496;
--rule:#e3e7ee;--accent:#2a78d6;--warn:#a06510;--ok:#0f7a4a;--bad:#c0392b}}
@media(prefers-color-scheme:dark){{:root{{--bg:#0e1117;--card:#161b23;--ink:#eef1f6;
--ink2:#a8b2c2;--ink3:#7b8496;--rule:#242c38;--accent:#3987e5;--warn:#d9a441;
--ok:#3faa77;--bad:#e66767}}}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.6 system-ui,-apple-system,"Segoe UI",sans-serif}}
.wrap{{max-width:1180px;margin:0 auto;padding:36px 22px 80px}}
header{{border-bottom:2px solid var(--ink);padding-bottom:20px;margin-bottom:26px}}
h1{{margin:0 0 4px;font-size:2rem;letter-spacing:-.02em}}
.sub{{color:var(--ink2);font-size:.95rem}}
.banner{{background:var(--warn);color:#fff;padding:9px 14px;border-radius:5px;
font-weight:600;font-size:.85rem;letter-spacing:.04em;margin-bottom:18px}}
.tiles{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
gap:1px;background:var(--rule);border:1px solid var(--rule);border-radius:7px;
overflow:hidden;margin:22px 0 30px}}
.tile{{background:var(--card);padding:15px 17px}}
.tile .k{{font-size:11px;text-transform:uppercase;letter-spacing:.07em;
color:var(--ink3);margin-bottom:5px}}
.tile .v{{font-size:1.45rem;font-weight:650;font-variant-numeric:tabular-nums;
letter-spacing:-.02em}}
figure{{margin:0 0 26px;background:var(--card);border:1px solid var(--rule);
border-radius:8px;overflow:hidden}}
figcaption{{padding:13px 17px;font-weight:600;font-size:.94rem;
border-bottom:1px solid var(--rule);color:var(--ink3)}}
/* Figures are rendered by matplotlib in the light theme and baked into PNGs,
   so they cannot follow the page theme. Give them a light card of their own
   rather than floating a white image on a dark ground. */
figure{{background:#fcfcfb}}
figure img{{display:block;width:100%;height:auto}}
figure.absent p{{padding:26px 17px;color:var(--ink3);margin:0;font-style:italic}}
.interactive{{margin:0 0 30px}}
.interactive h2{{font-size:1.05rem;margin:0 0 12px}}
.interactive iframe{{width:100%;height:620px;border:1px solid var(--rule);
border-radius:8px;background:var(--card)}}
.interactive a{{color:var(--accent);font-size:.9rem}}
h2.section{{font-size:1.05rem;margin:34px 0 12px;padding-top:20px;
border-top:1px solid var(--rule)}}
ul.stages{{list-style:none;padding:0;margin:0;display:grid;
grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:8px}}
ul.stages li{{background:var(--card);border:1px solid var(--rule);border-radius:6px;
padding:9px 13px;display:flex;justify-content:space-between;font-size:.86rem}}
ul.stages b{{font-size:11px;letter-spacing:.06em;text-transform:uppercase}}
li.ok b{{color:var(--ok)}} li.failed b{{color:var(--bad)}}
li.skipped b{{color:var(--ink3)}}
footer{{margin-top:34px;padding-top:16px;border-top:1px solid var(--rule);
color:var(--ink3);font-size:.83rem}}
</style></head><body><div class="wrap">
<header>
  <h1>{ticker} — {ui['subtitle']}</h1>
  <div class="sub">{asof}</div>
</header>
{banner}
<div class="tiles">{''.join(tiles)}</div>
{interactive}
{''.join(cards)}
<h2 class="section">{ui['stages']}</h2>
<ul class="stages">{stage_rows}</ul>
<footer>{ui['generated']} {datetime.now():%Y-%m-%d %H:%M} · volsurface</footer>
</div></body></html>"""

    dest = fig_dir / "index.html"
    dest.write_text(html, encoding="utf-8")
    n = sum(1 for name, _ in PANELS if (fig_dir / f"{name}.png").exists())
    print(f"Wrote {dest}  ({n}/{len(PANELS)} figures, "
          f"{dest.stat().st_size / 1024 / 1024:.1f} MB)")
    return dest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build a single-page dashboard "
                                             "from a completed run.")
    ap.add_argument("--out", default="outputs", help="pipeline output directory")
    ap.add_argument("--lang", default="en", choices=["en", "ru"])
    args = ap.parse_args(argv)
    build(Path(args.out), args.lang)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
