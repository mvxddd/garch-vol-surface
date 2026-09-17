#!/usr/bin/env python3
"""
Scheduled universe pass — the shape a service runs on.

    python scripts/precompute.py --tickers SPY QQQ NVDA
    python scripts/precompute.py --universe --provider polygon --production
    python scripts/precompute.py --cross-section

Calculates ahead of demand instead of on it: one pass writes a snapshot per
underlying, and a request afterwards is a read. Run it nightly.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from volsurface import Config
from volsurface.batch import DEFAULT_UNIVERSE, cross_section, run_universe
from volsurface.history import SurfaceHistory
from volsurface.utils import get_logger


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Precompute surfaces for a universe of underlyings.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--tickers", nargs="+", help="explicit list")
    p.add_argument("--universe", action="store_true",
                   help=f"use the built-in universe ({len(DEFAULT_UNIVERSE)} names)")
    p.add_argument("--provider", default="yfinance",
                   choices=["yfinance", "polygon", "synthetic"])
    p.add_argument("--production", action="store_true",
                   help="use Config.for_production: no synthetic fallback, and "
                        "refuses a provider not licensed to redistribute")
    p.add_argument("--allow-unlicensed", action="store_true",
                   help="staging escape hatch for --production")
    p.add_argument("--out", default="outputs")
    p.add_argument("--pause", type=float, default=1.0,
                   help="seconds between tickers, to respect rate limits")
    p.add_argument("--no-resume", action="store_true",
                   help="recompute names already snapshotted today")
    p.add_argument("--american", action="store_true")
    p.add_argument("--cross-section", action="store_true",
                   help="print the latest snapshot for every ticker and exit")
    p.add_argument("--quiet", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    log = get_logger("volsurface", verbose=not args.quiet)

    tickers = args.tickers or (list(DEFAULT_UNIVERSE) if args.universe
                               else ["SPY"])
    history_dir = Path(args.out) / "history"

    if args.cross_section:
        table = cross_section(SurfaceHistory(history_dir), tickers)
        print(table.to_string(index=False) if len(table)
              else "No snapshots stored yet — run a pass first.")
        return 0

    try:
        cfg = (Config.for_production(tickers[0], provider=args.provider,
                                     allow_unlicensed=args.allow_unlicensed)
               if args.production else Config())
    except PermissionError as exc:
        log.error("%s", exc)
        return 2

    cfg.data.provider = args.provider
    cfg.options.exercise_style = "american" if args.american else "european"
    cfg.output_dir = Path(args.out)
    cfg.figure_dir = Path(args.out) / "figures"
    cfg.report_dir = Path(args.out) / "reports"
    cfg.analytics.history_dir = history_dir
    cfg.analytics.save_snapshot = True
    cfg.verbose = not args.quiet

    result = run_universe(tickers, base_config=cfg, pause_seconds=args.pause,
                          resume=not args.no_resume)

    print("\n" + "=" * 72)
    print(result.to_frame().to_string(index=False))
    print("-" * 72)
    for key, value in result.summary().items():
        print(f"  {key:<20} {value}")
    if result.failed:
        print(f"\n  failed: {', '.join(result.failed)}")
    print("=" * 72)

    # Exit non-zero only if nothing worked: a universe pass that loses two
    # delisted names is a success, not a failure, and a scheduler should not
    # be paged for it.
    return 0 if result.succeeded else 1


if __name__ == "__main__":
    raise SystemExit(main())
