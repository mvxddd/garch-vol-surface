#!/usr/bin/env python3
"""
Convert the percent-format notebook source into a Colab-ready .ipynb.

Keeping the notebook in `# %%` script form is deliberate: it diffs cleanly in
git, it can be linted and imported, and the .ipynb is a build artefact rather
than a thing to hand-edit. Run this after changing the source.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "notebooks" / "garch_iv_surface_colab.py"
DST = ROOT / "notebooks" / "garch_iv_surface_colab.ipynb"

CELL_RE = re.compile(r"^# %%(?P<md>\s*\[markdown\])?\s*$")


def parse_cells(text: str) -> list[tuple[str, str]]:
    cells: list[tuple[str, str]] = []
    kind, buf = None, []
    for line in text.splitlines():
        m = CELL_RE.match(line)
        if m:
            if kind is not None:
                cells.append((kind, "\n".join(buf).strip("\n")))
            kind, buf = ("markdown" if m.group("md") else "code"), []
            continue
        buf.append(line)
    if kind is not None:
        cells.append((kind, "\n".join(buf).strip("\n")))
    return [(k, s) for k, s in cells if s.strip()]


def to_source(kind: str, body: str) -> list[str]:
    if kind == "markdown":
        # Strip the leading "# " comment marker from markdown cells.
        body = "\n".join(ln[2:] if ln.startswith("# ") else
                         ("" if ln.strip() == "#" else ln)
                         for ln in body.splitlines())
    lines = body.split("\n")
    return [ln + "\n" for ln in lines[:-1]] + [lines[-1]]


def main() -> int:
    if not SRC.exists():
        print(f"missing source: {SRC}", file=sys.stderr)
        return 1
    cells = parse_cells(SRC.read_text())
    nb = {
        "cells": [
            ({"cell_type": "markdown", "metadata": {}, "source": to_source(k, s)}
             if k == "markdown" else
             {"cell_type": "code", "metadata": {}, "execution_count": None,
              "outputs": [], "source": to_source(k, s)})
            for k, s in cells
        ],
        "metadata": {
            "colab": {"provenance": [], "toc_visible": True},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }
    DST.write_text(json.dumps(nb, indent=1))
    n_md = sum(1 for k, _ in cells if k == "markdown")
    print(f"Wrote {DST.relative_to(ROOT)} — {len(cells)} cells "
          f"({n_md} markdown, {len(cells) - n_md} code)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
