#!/usr/bin/env python3
"""
Command-line entry point.

    python scripts/run_pipeline.py --ticker SPY --start 2016-01-01
    python scripts/run_pipeline.py --ticker AAPL --provider polygon --no-walk-forward
    python scripts/run_pipeline.py --provider synthetic --quiet     # offline demo

Exit codes: 0 = every stage succeeded, 1 = the run completed with at least one
failed stage (inspect the printed status table), 2 = the run could not start.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from a checkout without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from volsurface import Config, run_pipeline                     # noqa: E402
from volsurface.i18n import LANGUAGES, set_language, t           # noqa: E402
from volsurface.utils import get_logger                          # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="GARCH volatility forecasting and implied-vol surface study.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ticker", default="SPY", help="underlying symbol")
    p.add_argument("--start", default="2016-01-01", help="history start date")
    p.add_argument("--end", default=None, help="history end date (default: today)")
    p.add_argument("--provider", default="yfinance",
                   choices=["yfinance", "polygon", "synthetic"])
    p.add_argument("--max-expiries", type=int, default=12)
    p.add_argument("--surface-method", default="svi", choices=["svi", "spline"])
    p.add_argument("--rate", type=float, default=0.042,
                   help="flat risk-free rate used for discounting")
    p.add_argument("--out", default="outputs", help="output directory")
    p.add_argument("--no-figures", action="store_true")
    p.add_argument("--no-walk-forward", action="store_true",
                   help="skip the (slow) out-of-sample validation")
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--no-fallback", action="store_true",
                   help="fail loudly instead of falling back to synthetic data")
    p.add_argument("--american", action="store_true",
                   help="invert quotes under American exercise (binomial tree); "
                        "slower, but removes the upward bias on long-dated puts")
    p.add_argument("--snapshot", action="store_true",
                   help="save today's surface to the history store and z-score "
                        "it against its own past")
    p.add_argument("--lang", default="en", choices=list(LANGUAGES),
                   help="language for chart text and this summary")
    p.add_argument("--quiet", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    log = get_logger("volsurface", verbose=not args.quiet)

    cfg = Config()
    cfg.verbose = not args.quiet
    cfg.language = args.lang
    set_language(args.lang)
    cfg.data.ticker = args.ticker
    cfg.data.start = args.start
    cfg.data.end = args.end
    cfg.data.provider = args.provider
    cfg.data.use_cache = not args.no_cache
    cfg.data.allow_synthetic_fallback = not args.no_fallback
    cfg.options.max_expiries = args.max_expiries
    cfg.options.risk_free_rate = args.rate
    cfg.surface.method = args.surface_method
    cfg.options.exercise_style = "american" if args.american else "european"
    cfg.analytics.save_snapshot = args.snapshot
    cfg.analytics.history_dir = Path(args.out) / "history"
    cfg.output_dir = Path(args.out)
    cfg.figure_dir = Path(args.out) / "figures"
    cfg.report_dir = Path(args.out) / "reports"

    try:
        res = run_pipeline(cfg, make_figures=not args.no_figures,
                           run_walk_forward=not args.no_walk_forward)
    except Exception as exc:
        log.error("Pipeline could not start: %s", exc)
        return 2

    res.save()
    print("\n" + "=" * 70)
    print("  " + t("cli.header", ticker=cfg.data.ticker))
    print("=" * 70)
    for key, value in res.headline().items():
        # Field names are translated; the values stay as computed. Booleans read
        # as words rather than Python literals, which is the one place a
        # non-English reader would otherwise trip.
        if isinstance(value, bool):
            value = t("cli.yes") if value else t("cli.no")
        elif key == "n_snapshots":
            value = f"{value}"
        elif key == "vrp_signal":
            value = t(f"vrp.signal.{value}")
        label = t(f"cli.{key}")
        print(f"  {label:<28} {value}")
    print("-" * 70)
    marks = {"ok": t("cli.stage_ok"), "failed": t("cli.stage_fail"),
             "skipped": t("cli.stage_skip")}
    for stage, state in res.status.items():
        print(f"  [{marks.get(state, '?   ')}] {stage}"
              + (f"  — {res.errors[stage]}" if stage in res.errors else ""))
    print("=" * 70)
    return 1 if any(v == "failed" for v in res.status.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
