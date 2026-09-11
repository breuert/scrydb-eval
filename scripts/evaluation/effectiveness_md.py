#!/usr/bin/env python3
"""
effectiveness_md.py
===================

Renders the retrieval-effectiveness table as GitHub-flavoured **Markdown**,
so the numbers can be dropped straight into a README instead of a LaTeX
document -- the Markdown sibling of ``effectiveness_table.py``. It reads the
same ``scores.csv`` that ``effectiveness.py`` writes and never re-runs the
(expensive) qrels evaluation.

Differences from the LaTeX table, forced by Markdown's smaller vocabulary:

  * no ``\\multirow``: the dataset name is repeated on each of its measure
    rows rather than spanning them
  * no ``\\underline``: the second-best score in a row is *italicised*
    (``--second-best underline`` uses a ``<u>`` tag instead)
  * ``cos_int8``/``cos_float`` become ``cos<sub>int8</sub>``/
    ``cos<sub>float</sub>``, the Markdown equivalent of the paper's
    ``cos\\textsubscript{}`` notation (``--labels plain`` keeps them literal)

Layouts:

  ``blocks``       one wide table, rows = (dataset, measure) -- the same
                   shape as the paper's ``table.tex`` (default)
  ``per-measure``  one table per measure, rows = datasets -- narrower, and
                   much easier to read in a README, especially when combined
                   with ``--measures nDCG@10``

Usage
-----
    python effectiveness_md.py
    python effectiveness_md.py --layout per-measure --measures nDCG@10
    python effectiveness_md.py --inject ../../README.md
    python effectiveness_md.py --output -          # print to stdout

Run `python effectiveness_md.py --help` for all options.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from effectiveness import (
        BASELINE_KEY,
        BASELINE_LABEL,
        DATASET_LABELS,
        MEASURES,
        METHODS,
    )

    _HAVE_SPEC = True
except ImportError as exc:  # pragma: no cover - depends on the environment
    # effectiveness.py imports repro_eval to *compute* the scores; rendering
    # an already-computed scores.csv doesn't need it, so degrade to the
    # labels and ordering the CSV itself carries rather than failing.
    print(
        f"note: falling back to the labels/order in the CSV (cannot import effectiveness.py: {exc})",
        file=sys.stderr,
    )
    _HAVE_SPEC = False
    BASELINE_KEY, BASELINE_LABEL = "baseline", "MTEB (Qwen3-8B)"
    DATASET_LABELS, MEASURES, METHODS = {}, {}, {}

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCORES = REPO_ROOT / "results" / "effectiveness" / "scores.csv"

MISSING = "--"
_SUBSCRIPT_RE = re.compile(r"cos_(int8|float)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Render the retrieval-effectiveness table as Markdown from an existing "
        "scores.csv (as written by effectiveness.py), without re-evaluating any runs."
    )
    p.add_argument("--scores", type=Path, default=DEFAULT_SCORES, help="Path to scores.csv")
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output .md path, or '-' for stdout (default: alongside --scores, as table.md; "
        "ignored unless given when --inject is used)",
    )
    p.add_argument(
        "--inject",
        type=Path,
        default=None,
        help="Splice the table into this Markdown file, between "
        "'<!-- BEGIN <marker> -->' / '<!-- END <marker> -->' comments (appended to the "
        "end of the file if those markers aren't there yet)",
    )
    p.add_argument(
        "--marker",
        default="effectiveness-table",
        help="Name used in the --inject marker comments (default: effectiveness-table)",
    )
    p.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Subset of datasets to include (default: every dataset present in --scores)",
    )
    p.add_argument(
        "--measures",
        nargs="+",
        default=None,
        help="Subset of measures to include, e.g. 'nDCG@10' (default: every measure present)",
    )
    p.add_argument(
        "--layout",
        choices=("blocks", "per-measure"),
        default="blocks",
        help="'blocks': one table, rows = (dataset, measure), like the paper's table.tex. "
        "'per-measure': one table per measure, rows = datasets (default: blocks)",
    )
    p.add_argument("--decimals", type=int, default=3, help="Decimal places in the table")
    p.add_argument(
        "--no-bold-best",
        action="store_true",
        help="Don't mark the best-performing method and the runner-up (including the "
        "baseline) in each row",
    )
    p.add_argument(
        "--second-best",
        choices=("italic", "underline"),
        default="italic",
        help="How to mark the second-best score per row: *italics* or an <u> tag "
        "(default: italic)",
    )
    p.add_argument(
        "--labels",
        choices=("sub", "plain"),
        default="sub",
        help="'sub': render cos_int8/cos_float with <sub> tags, matching the paper. "
        "'plain': leave them as-is (default: sub)",
    )
    p.add_argument(
        "--title",
        default="Retrieval effectiveness",
        help="Heading text placed above the table (default: 'Retrieval effectiveness')",
    )
    p.add_argument(
        "--heading-level",
        type=int,
        default=2,
        help="Markdown heading level for --title, 0 for no heading (default: 2)",
    )
    p.add_argument("--no-caption", action="store_true", help="Omit the caption/legend line below the table")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Markdown primitives
# ---------------------------------------------------------------------------
def md_label(label: str, style: str) -> str:
    """Method/dataset label as a Markdown table cell: cos_int8 -> cos<sub>int8</sub>
    (``style="sub"``), and any pipe escaped so it can't split the cell."""
    if style == "sub":
        label = _SUBSCRIPT_RE.sub(r"cos<sub>\1</sub>", label)
    return label.replace("|", "\\|")


def render_table(header: list[str], rows: list[list[str]], aligns: str) -> list[str]:
    """A padded GFM pipe table. *aligns* is one 'l'/'r' character per column;
    the padding is cosmetic (it keeps the raw Markdown readable) but the
    ':---'/'---:' separator row is what actually aligns the rendered table."""
    widths = [max([3, len(header[i])] + [len(row[i]) for row in rows]) for i in range(len(header))]

    def line(cells: list[str]) -> str:
        padded = [c.ljust(w) if a == "l" else c.rjust(w) for c, w, a in zip(cells, widths, aligns)]
        return "| " + " | ".join(padded) + " |"

    sep = [":" + "-" * (w - 1) if a == "l" else "-" * (w - 1) + ":" for w, a in zip(widths, aligns)]
    return [line(header), "| " + " | ".join(sep) + " |"] + [line(row) for row in rows]


def rank_values(values: dict[str, float]) -> tuple[float | None, float | None]:
    """Best and second-best *distinct* unrounded values in a row. Methods that
    tie exactly (e.g. reranking the same candidates into the same order) all
    get the same mark, which is why the caption says ties are decided before
    rounding."""
    present = {v for v in values.values() if not np.isnan(v)}
    if not present:
        return None, None
    distinct = sorted(present, reverse=True)
    return distinct[0], (distinct[1] if len(distinct) > 1 else None)


def format_cell(value: float, best: float | None, second: float | None, args: argparse.Namespace) -> str:
    if value is None or np.isnan(value):
        return MISSING
    cell = f"{value:.{args.decimals}f}"
    if args.no_bold_best:
        return cell
    if best is not None and value == best:
        return f"**{cell}**"
    if second is not None and value == second:
        return f"<u>{cell}</u>" if args.second_best == "underline" else f"*{cell}*"
    return cell


def score_cells(row_values: "pd.Series", columns: list[str], args: argparse.Namespace) -> list[str]:
    column_values = {k: row_values.get(k, np.nan) for k in columns}
    best, second = (None, None) if args.no_bold_best else rank_values(column_values)
    return [format_cell(column_values[k], best, second, args) for k in columns]


# ---------------------------------------------------------------------------
# Document
# ---------------------------------------------------------------------------
def caption(args: argparse.Namespace, measures: list[str]) -> str:
    text = (
        f"Retrieval effectiveness ({', '.join(measures)}) of each retrieval method against "
        "the full-precision embedding baseline reported by MTEB for Qwen3-Embedding-8B."
    )
    if not args.no_bold_best:
        second = "<u>underlined</u>" if args.second_best == "underline" else "*italicised*"
        text += (
            f" Best score per row in **bold**, second-best {second}; tied methods share the "
            "mark, with ties determined by the unrounded scores."
        )
    return f"_{text}_"


def build_markdown(
    df: pd.DataFrame,
    datasets: list[str],
    measures: list[str],
    columns: list[str],
    labels: dict[str, str],
    args: argparse.Namespace,
) -> str:
    pivot = df.pivot_table(index=["dataset", "measure"], columns="method_key", values="value")
    method_headers = [md_label(labels[k], args.labels) for k in columns]

    parts: list[str] = ["<!-- Auto-generated by src/evaluation/effectiveness_md.py -- do not edit by hand. -->"]
    if args.heading_level > 0:
        parts.append(f"{'#' * args.heading_level} {args.title}")

    if args.layout == "blocks":
        header = ["Dataset", "Measure"] + method_headers
        rows = []
        for dataset in datasets:
            for measure in measures:
                if (dataset, measure) not in pivot.index:
                    continue
                cells = score_cells(pivot.loc[(dataset, measure)], columns, args)
                rows.append([md_label(DATASET_LABELS.get(dataset, dataset), args.labels), measure] + cells)
        parts.append("\n".join(render_table(header, rows, "ll" + "r" * len(columns))))
    else:
        header = ["Dataset"] + method_headers
        for measure in measures:
            rows = []
            for dataset in datasets:
                if (dataset, measure) not in pivot.index:
                    continue
                cells = score_cells(pivot.loc[(dataset, measure)], columns, args)
                rows.append([md_label(DATASET_LABELS.get(dataset, dataset), args.labels)] + cells)
            if not rows:
                continue
            if args.heading_level > 0:
                parts.append(f"{'#' * min(args.heading_level + 1, 6)} {measure}")
            else:
                parts.append(f"**{measure}**")
            parts.append("\n".join(render_table(header, rows, "l" + "r" * len(columns))))

    if not args.no_caption:
        parts.append(caption(args, measures))
    return "\n\n".join(parts) + "\n"


def inject(path: Path, marker: str, block: str) -> None:
    """Replace the '<!-- BEGIN marker -->'..'<!-- END marker -->' section of
    *path* with *block*, appending a fresh section if the markers are absent."""
    begin, end = f"<!-- BEGIN {marker} -->", f"<!-- END {marker} -->"
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    if (begin in text) != (end in text):
        raise SystemExit(f"{path} has only one of the two markers ({begin} / {end}); fix it by hand first.")
    section = f"{begin}\n{block.rstrip()}\n{end}"
    if begin in text:
        head, rest = text.split(begin, 1)
        _, tail = rest.split(end, 1)
        text = head + section + tail
    else:
        text = (text.rstrip("\n") + "\n\n" if text.strip() else "") + section + "\n"
    path.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.scores)

    seen_datasets = list(dict.fromkeys(df["dataset"]))
    datasets = args.datasets or (
        [d for d in DATASET_LABELS if d in seen_datasets] + [d for d in seen_datasets if d not in DATASET_LABELS]
    )
    unknown = [d for d in datasets if d not in seen_datasets]
    if unknown:
        raise SystemExit(f"Not in {args.scores}: {', '.join(unknown)}. Available: {', '.join(seen_datasets)}")

    seen_measures = list(dict.fromkeys(df["measure"]))
    measures = args.measures or (
        [m for m in MEASURES if m in seen_measures] + [m for m in seen_measures if m not in MEASURES]
    )
    unknown = [m for m in measures if m not in seen_measures]
    if unknown:
        raise SystemExit(f"Not in {args.scores}: {', '.join(unknown)}. Available: {', '.join(seen_measures)}")

    # Labels come from METHODS where available, so the table reflects the
    # current method names even when rebuilt from an older scores.csv.
    seen_methods = list(dict.fromkeys(df["method_key"]))
    method_order = [k for k in METHODS if k in seen_methods] if _HAVE_SPEC else [
        k for k in seen_methods if k != BASELINE_KEY
    ]
    columns = method_order + ([BASELINE_KEY] if BASELINE_KEY in seen_methods else [])
    labels = dict(zip(df["method_key"], df["method"]))
    if _HAVE_SPEC:
        labels.update({k: v[0] for k, v in METHODS.items()})
        labels[BASELINE_KEY] = BASELINE_LABEL

    markdown = build_markdown(df, datasets, measures, columns, labels, args)

    if args.inject is not None:
        inject(args.inject, args.marker, markdown)
        print(f"Updated {args.inject} (section '{args.marker}')")
        if args.output is None:
            return

    output = args.output or (args.scores.parent / "table.md")
    if str(output) == "-":
        sys.stdout.write(markdown)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(markdown, encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
