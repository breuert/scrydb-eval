#!/usr/bin/env python3
"""
effectiveness_table.py
=======================

Rebuilds the paper's retrieval-effectiveness LaTeX table straight from the
``scores.csv`` already written by ``effectiveness.py``, without re-running
the (expensive) qrels evaluation itself -- same pattern as
``latency_table.py`` for ``efficiency.py``.

Usage
-----
    python effectiveness_table.py
    python effectiveness_table.py --scores results/effectiveness/scores.csv --output table.tex
    python effectiveness_table.py --datasets trec-covid scifact  # subset only

Run `python effectiveness_table.py --help` for all options.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from effectiveness import DATASET_LABELS, to_latex

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCORES = REPO_ROOT / "results" / "effectiveness" / "scores.csv"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Rebuild the retrieval-effectiveness LaTeX table from an existing "
        "scores.csv (as written by effectiveness.py), without re-evaluating any runs."
    )
    p.add_argument("--scores", type=Path, default=DEFAULT_SCORES, help="Path to scores.csv")
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output .tex path (default: alongside --scores, as table.tex)",
    )
    p.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Subset of datasets to include (default: every dataset present in --scores)",
    )
    p.add_argument("--decimals", type=int, default=3, help="Decimal places in the LaTeX table")
    p.add_argument(
        "--no-bold-best",
        action="store_true",
        help="Don't bold the best-performing method and underline the "
        "second-best (including the baseline) in each row",
    )
    p.add_argument(
        "--header-angle",
        type=int,
        default=90,
        help="Rotate retrieval-method column headers by this many degrees "
        "for a denser table (0 to disable rotation, default: 90)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.scores)

    seen = list(dict.fromkeys(df["dataset"]))
    datasets = args.datasets or (
        [d for d in DATASET_LABELS if d in seen] + [d for d in seen if d not in DATASET_LABELS]
    )

    latex = to_latex(
        df,
        datasets,
        decimals=args.decimals,
        bold_best=not args.no_bold_best,
        header_angle=args.header_angle,
    )

    output = args.output or (args.scores.parent / "table.tex")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(latex + "\n", encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
