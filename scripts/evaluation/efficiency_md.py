#!/usr/bin/env python3
"""
efficiency_md.py
================

Renders the query-latency table as GitHub-flavoured **Markdown**, so the
numbers can be dropped straight into a README instead of a LaTeX document --
the Markdown sibling of ``efficiency_table.py``. It reads the same CSVs that
``efficiency.py`` writes (``summary.csv``, ``overall_summary.csv``,
``per_query_means.csv``) and never re-runs the (expensive) benchmark.

Same content as ``efficiency_table.py``: rows = datasets plus a
cross-dataset **Mean** row, columns = methods (mean latency in ms, no std),
with two context columns after "Dataset" so latency can be read against
dataset scale and query verbosity:

  * corpus size (hardcoded in ``efficiency_table.DATASET_SIZES``, since
    ``summary.csv`` carries no document count)
  * mean query length in words, computed over exactly the queries that were
    timed for that row (the ids in ``per_query_means.csv``, with text looked
    up from ``data/datasets/beir/<dataset>/queries.jsonl``). That file is
    not part of the repo, so the column degrades to "--" with a warning when
    it isn't there; ``--no-query-length`` drops it entirely.

Markdown has no ``\\underline``, so the second-fastest method per row is
*italicised* instead (``--second-best underline`` uses a ``<u>`` tag), and
``cos_int8``/``cos_float`` are rendered as ``cos<sub>int8</sub>``/
``cos<sub>float</sub>`` -- the Markdown equivalent of the paper's
``cos\\textsubscript{}`` notation.

Usage
-----
    python efficiency_md.py
    python efficiency_md.py --methods bm25 bm25_hamming rrf_hamming
    python efficiency_md.py --inject ../../README.md
    python efficiency_md.py --output -          # print to stdout

Run `python efficiency_md.py --help` for all options.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from efficiency import METHODS
from efficiency_table import (
    DATASET_LABELS,
    DATASET_SIZES,
    DEFAULT_DATASETS_DIR,
    DEFAULT_RESULTS_DIR,
    load_query_lengths,
)

MISSING = "--"
_SUBSCRIPT_RE = re.compile(r"cos_(int8|float)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Render the query-latency table as Markdown (mean latency per "
        "dataset/method, plus corpus size and mean query length columns) from "
        "efficiency.py's summary.csv / overall_summary.csv / per_query_means.csv, "
        "without re-running the benchmark."
    )
    p.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory containing summary.csv, overall_summary.csv, and per_query_means.csv "
        "(as written by efficiency.py)",
    )
    p.add_argument(
        "--datasets-dir",
        type=Path,
        default=DEFAULT_DATASETS_DIR,
        help="Directory containing one <dataset>/queries.jsonl per BEIR dataset",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output .md path, or '-' for stdout (default: <results-dir>/efficiency_table.md; "
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
        default="efficiency-table",
        help="Name used in the --inject marker comments (default: efficiency-table)",
    )
    p.add_argument(
        "--methods",
        nargs="+",
        default=None,
        choices=list(METHODS.keys()),
        help="Subset of methods to include as table columns (default: every method present "
        "in --results-dir's summary.csv, in the canonical order from efficiency.py's METHODS)",
    )
    p.add_argument("--latency-decimals", type=int, default=1, help="Decimal places for latency (ms) cells")
    p.add_argument("--length-decimals", type=int, default=1, help="Decimal places for query-length (words) cells")
    p.add_argument(
        "--no-query-length",
        action="store_true",
        help="Drop the mean-query-length column (skips reading queries.jsonl entirely)",
    )
    p.add_argument(
        "--no-bold-best",
        action="store_true",
        help="Don't mark the fastest and second-fastest method in each row",
    )
    p.add_argument(
        "--second-best",
        choices=("italic", "underline"),
        default="italic",
        help="How to mark the second-fastest method per row: *italics* or an <u> tag "
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
        default="Query latency",
        help="Heading text placed above the table (default: 'Query latency')",
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
# Query length -- same source of truth as efficiency_table.py, but a dataset
# whose queries.jsonl is missing (the data/ tree isn't in the repo) warns and
# falls back to "--" instead of aborting the whole table.
# ---------------------------------------------------------------------------
def query_length_stats(per_query: pd.DataFrame, datasets_dir: Path, datasets: list[str]) -> dict[str, float]:
    stats: dict[str, float] = {}
    for dataset in datasets:
        query_ids = set(per_query.loc[per_query.dataset == dataset, "query_id"].astype(str).unique())
        if not query_ids:
            continue
        try:
            lengths = load_query_lengths(datasets_dir, dataset, query_ids)
        except (FileNotFoundError, ValueError) as exc:
            print(f"warning: no query-length for {dataset}: {exc}", file=sys.stderr)
            continue
        stats[dataset] = float(np.mean(lengths))
    return stats


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


def latency_cells(row_values: "pd.Series", method_order: list[str], args: argparse.Namespace) -> list[str]:
    """One formatted cell per method, fastest in the row marked bold and
    second-fastest italic/underlined (lower is better here, unlike the
    effectiveness table)."""
    column_values = {k: row_values.get(k, np.nan) for k in method_order}
    present = {k: v for k, v in column_values.items() if not np.isnan(v)}
    best_key = second_key = None
    if not args.no_bold_best and present:
        ranked = sorted(present, key=lambda k: present[k])  # ascending: fastest first
        best_key = ranked[0]
        if len(ranked) > 1:
            second_key = ranked[1]

    cells = []
    for k in method_order:
        v = column_values[k]
        if np.isnan(v):
            cells.append(MISSING)
            continue
        cell = f"{v:.{args.latency_decimals}f}"
        if k == best_key:
            cell = f"**{cell}**"
        elif k == second_key:
            cell = f"<u>{cell}</u>" if args.second_best == "underline" else f"*{cell}*"
        cells.append(cell)
    return cells


# ---------------------------------------------------------------------------
# Document
# ---------------------------------------------------------------------------
def caption(args: argparse.Namespace, with_length: bool) -> str:
    text = (
        "Mean query latency (ms) of each retrieval strategy, per BEIR dataset and averaged "
        "across datasets (**Mean** row), alongside corpus size"
    )
    text += (
        " and mean query length (words, computed over the queries timed for that row)."
        if with_length
        else "."
    )
    if not args.no_bold_best:
        second = "<u>underlined</u>" if args.second_best == "underline" else "*italicised*"
        text += f" Fastest method per row in **bold**, second-fastest {second}."
    return f"_{text}_"


def build_markdown(
    summary: pd.DataFrame,
    overall_summary: pd.DataFrame,
    length_stats: dict[str, float],
    datasets: list[str],
    method_order: list[str],
    args: argparse.Namespace,
) -> str:
    # Labels come from METHODS, not the CSV's "method" column, so the table
    # always reflects the current method names even when it is regenerated
    # from a summary.csv written by an older version of METHODS.
    labels = {k: v[0] for k, v in METHODS.items()}
    pivot = summary.pivot_table(index="dataset", columns="method_key", values="mean_ms")
    overall_row = overall_summary.set_index("method_key")["mean_ms"]
    # Drop the column entirely rather than print a column of "--" when no
    # dataset's queries.jsonl could be read.
    with_length = not args.no_query_length and bool(length_stats)

    header = ["Dataset", "Size"] + (["Query Length"] if with_length else [])
    header += [md_label(labels[k], args.labels) for k in method_order]
    aligns = "lr" + ("r" if with_length else "") + "r" * len(method_order)

    rows = []
    for dataset in datasets:
        if dataset not in pivot.index:
            continue
        row = [md_label(DATASET_LABELS.get(dataset, dataset), args.labels), DATASET_SIZES.get(dataset, MISSING)]
        if with_length:
            row.append(f"{length_stats[dataset]:.{args.length_decimals}f}" if dataset in length_stats else MISSING)
        rows.append(row + latency_cells(pivot.loc[dataset], method_order, args))

    mean_row = ["**Mean**", MISSING] + ([MISSING] if with_length else [])
    rows.append(mean_row + latency_cells(overall_row, method_order, args))

    parts: list[str] = ["<!-- Auto-generated by src/evaluation/efficiency_md.py -- do not edit by hand. -->"]
    if args.heading_level > 0:
        parts.append(f"{'#' * args.heading_level} {args.title}")
    parts.append("\n".join(render_table(header, rows, aligns)))
    if not args.no_caption:
        parts.append(caption(args, with_length))
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

    summary = pd.read_csv(args.results_dir / "summary.csv")
    overall_summary = pd.read_csv(args.results_dir / "overall_summary.csv")
    per_query = pd.read_csv(args.results_dir / "per_query_means.csv", dtype={"query_id": str})

    seen = list(dict.fromkeys(summary["dataset"]))
    datasets = [d for d in DATASET_LABELS if d in seen] + [d for d in seen if d not in DATASET_LABELS]

    seen_methods = set(overall_summary["method_key"])
    method_order = [m for m in (args.methods or METHODS) if m in seen_methods]
    if not method_order:
        raise SystemExit("None of the requested --methods are present in the results CSVs.")
    summary = summary[summary.method_key.isin(method_order)]
    overall_summary = overall_summary.set_index("method_key").loc[method_order].reset_index()

    length_stats = {} if args.no_query_length else query_length_stats(per_query, args.datasets_dir, datasets)

    markdown = build_markdown(summary, overall_summary, length_stats, datasets, method_order, args)

    if args.inject is not None:
        inject(args.inject, args.marker, markdown)
        print(f"Updated {args.inject} (section '{args.marker}')")
        if args.output is None:
            return

    output = args.output or (args.results_dir / "efficiency_table.md")
    if str(output) == "-":
        sys.stdout.write(markdown)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(markdown, encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
