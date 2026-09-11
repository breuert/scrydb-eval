#!/usr/bin/env python3
"""
efficiency_table.py
=================

Builds the paper's query-latency LaTeX table straight from the CSVs already
written by ``efficiency.py`` (``summary.csv``, ``overall_summary.csv``,
``per_query_means.csv``), without re-running the (expensive) benchmark
itself -- same pattern as ``plot_efficiency.py``.

Same look as ``efficiency.py``'s own ``table.tex`` (rows = datasets + a
cross-dataset **Mean** row, columns = methods, mean latency only -- no std,
fastest method per row bolded, second-fastest underlined), plus two context
columns inserted right after "Dataset" so latency can be read against
dataset scale and query verbosity:

  * corpus size (hardcoded from the BEIR paper / MTEB, since ``summary.csv``
    doesn't carry a document count)
  * mean query length (in words), computed from the *same* queries that
    were timed for that row -- i.e. the query ids appearing in
    ``per_query_means.csv`` for that dataset, with text looked up from
    ``data/datasets/beir/<dataset>/queries.jsonl``. Using exactly the timed
    sample (rather than the dataset's full query set) keeps this column
    tied to the queries the latency numbers in the same row were measured
    on.

Usage
-----
    python efficiency_table.py
    python efficiency_table.py --results-dir results/efficiency --output tab.tex
    python efficiency_table.py --methods bm25 bm25_hamming rrf_hamming  # subset only
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from efficiency import METHODS, shortstack_label

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_DIR = REPO_ROOT / "results" / "efficiency"
DEFAULT_DATASETS_DIR = REPO_ROOT / "data" / "datasets" / "beir"

# BEIR index name -> display corpus size, as reported in the BEIR paper /
# MTEB (matches data/datasets/beir/<dataset>/corpus.jsonl line counts).
DATASET_SIZES = {
    "arguana": "8.67K",
    "fiqa": "57K",
    "nfcorpus": "3.6K",
    "quora": "523K",
    "scidocs": "25K",
    "scifact": "5K",
    "webis-touche2020": "382K",
    "trec-covid": "171K",
}

DATASET_LABELS = {
    "arguana": "ArguAna",
    "fiqa": "FiQA",
    "nfcorpus": "NFCorpus",
    "quora": "Quora",
    "scidocs": "SciDocs",
    "scifact": "SciFact",
    "webis-touche2020": "Touché",
    "trec-covid": "TREC-COVID",
}

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build the query-latency LaTeX table (mean latency per dataset/method, "
        "plus corpus size and mean query length columns), in the same style as "
        "efficiency.py's own table.tex, from efficiency.py's summary.csv / "
        "overall_summary.csv / per_query_means.csv."
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
    p.add_argument("--output", type=Path, default=None, help="Output .tex path (default: <results-dir>/efficiency_table.tex)")
    p.add_argument(
        "--methods",
        nargs="+",
        default=None,
        choices=list(METHODS.keys()),
        help="Subset of methods to include as table columns (default: every method present "
        "in --results-dir's summary.csv, in the canonical order from efficiency.py's METHODS).",
    )
    p.add_argument("--latency-decimals", type=int, default=1, help="Decimal places for latency (ms) cells")
    p.add_argument("--length-decimals", type=int, default=1, help="Decimal places for query-length (words) cells")
    p.add_argument(
        "--no-bold-best",
        action="store_true",
        help="Don't bold the fastest method and underline the second-fastest in each row",
    )
    p.add_argument(
        "--header-angle",
        type=int,
        default=90,
        help="Rotate method column headers by this many degrees (0 to disable, default: 90)",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Query length
# ---------------------------------------------------------------------------
def load_query_lengths(datasets_dir: Path, dataset: str, query_ids: set[str]) -> np.ndarray:
    """Word-count length of each id in *query_ids*, read from queries.jsonl."""
    path = datasets_dir / dataset / "queries.jsonl"
    lengths = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            qid = str(record["_id"])
            if qid in query_ids:
                lengths[qid] = len(record["text"].split())
    missing = query_ids - lengths.keys()
    if missing:
        raise ValueError(f"[{dataset}] {len(missing)} query id(s) not found in {path}: {sorted(missing)[:5]}...")
    return np.array([lengths[qid] for qid in query_ids], dtype=float)


def query_length_stats(per_query: pd.DataFrame, datasets_dir: Path, datasets: list[str]) -> dict[str, float]:
    """dataset -> mean query length in words, over its timed query sample."""
    stats = {}
    for dataset in datasets:
        query_ids = set(per_query.loc[per_query.dataset == dataset, "query_id"].astype(str).unique())
        if not query_ids:
            continue
        lengths = load_query_lengths(datasets_dir, dataset, query_ids)
        stats[dataset] = float(np.mean(lengths))
    return stats


# ---------------------------------------------------------------------------
# LaTeX table -- same shape/style as efficiency.py's own to_latex() (rows =
# datasets + a cross-dataset Mean row, mean latency only, fastest method per
# row bolded / second-fastest underlined), plus "Size" and "Query Length"
# context columns inserted right after "Dataset".
# ---------------------------------------------------------------------------
def to_latex(
    summary: pd.DataFrame,
    overall_summary: pd.DataFrame,
    length_stats: dict[str, float],
    datasets: list[str],
    method_order: list[str],
    latency_decimals: int,
    length_decimals: int,
    bold_best: bool,
    header_angle: int = 90,
) -> str:
    # Labels come from METHODS, not the CSV's "method" column, so table.tex
    # always reflects the current names even when regenerated from a CSV
    # written by an older version of METHODS.
    labels = {k: v[0] for k, v in METHODS.items()}
    pivot = summary.pivot_table(index="dataset", columns="method_key", values="mean_ms")
    overall_row = overall_summary.set_index("method_key")["mean_ms"]

    lines = []
    lines.append("% Auto-generated by src/evaluation/efficiency_table.py -- do not edit by hand.")
    lines.append("% Requires \\usepackage{booktabs} in the preamble.")
    lines.append("\\begin{table}[!t]")
    lines.append("\\centering")
    lines.append("\\resizebox{\\textwidth}{!}{%")
    col_spec = "|".join(["l", "r", "r"] + ["r"] * len(method_order))
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")
    method_headers = [
        f"\\rotatebox{{{header_angle}}}{{{shortstack_label(labels[k])}}}"
        if header_angle
        else shortstack_label(labels[k])
        for k in method_order
    ]
    query_length_header = f"\\rotatebox{{{header_angle}}}{{Query Length}}" if header_angle else "Query Length"
    header = ["Dataset", "Size", query_length_header] + method_headers
    lines.append(" & ".join(header) + " \\\\")
    lines.append("\\midrule")

    def row_cells(row_values: "pd.Series") -> list[str]:
        column_values = {k: row_values.get(k, np.nan) for k in method_order}
        best_key = second_key = None
        present = {k: v for k, v in column_values.items() if not np.isnan(v)}
        if bold_best and present:
            ranked = sorted(present, key=lambda k: present[k])  # ascending: fastest first
            best_key = ranked[0]
            if len(ranked) > 1:
                second_key = ranked[1]
        cells = []
        for k in method_order:
            v = column_values[k]
            if np.isnan(v):
                cell = "--"
            else:
                cell = f"{v:.{latency_decimals}f}"
                if k == best_key:
                    cell = f"\\textbf{{{cell}}}"
                elif k == second_key:
                    cell = f"\\underline{{{cell}}}"
            cells.append(cell)
        return cells

    for dataset in datasets:
        if dataset not in pivot.index:
            continue
        size = DATASET_SIZES.get(dataset, "--")
        length_cell = f"{length_stats[dataset]:.{length_decimals}f}" if dataset in length_stats else "--"
        label = DATASET_LABELS.get(dataset, dataset)
        lines.append(" & ".join([label, size, length_cell] + row_cells(pivot.loc[dataset])) + " \\\\")

    lines.append("\\midrule")
    lines.append(" & ".join(["\\textbf{Mean}", "--", "--"] + row_cells(overall_row)) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}%")
    lines.append("}")
    lines.append(
        "\\caption{Mean query latency (ms) of each retrieval strategy, per BEIR dataset "
        "and averaged across datasets (\\textbf{Mean} row), alongside corpus size and mean "
        "query length (words, computed over the queries timed for that row). Fastest method "
        "per row in \\textbf{bold}, second-fastest \\underline{underlined}.}"
    )
    lines.append("\\label{tab:retrieval-latency}")
    lines.append("\\end{table}")
    return "\n".join(lines)


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

    length_stats = query_length_stats(per_query, args.datasets_dir, datasets)

    latex = to_latex(
        summary,
        overall_summary,
        length_stats,
        datasets,
        method_order,
        latency_decimals=args.latency_decimals,
        length_decimals=args.length_decimals,
        bold_best=not args.no_bold_best,
        header_angle=args.header_angle,
    )

    output = args.output or (args.results_dir / "efficiency_table.tex")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(latex + "\n", encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
