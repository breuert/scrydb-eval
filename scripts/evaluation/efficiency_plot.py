#!/usr/bin/env python3
"""
efficiency_plot.py
===================

Regenerates the efficiency bar charts (``bar_by_dataset.pdf``,
``bar_overall.pdf``) straight from the CSVs already written by
``efficiency.py`` (``summary.csv``, ``overall_summary.csv``), without
re-running the (expensive) benchmark itself. Useful for re-styling or
re-labeling the plots after the fact.

BEIR index names (as used for the ``<dataset>.db`` files and the ``dataset``
column of the CSVs) are mapped to the display names used in the paper's
prose (see ``DATASET_LABELS`` below) for the x-axis of ``bar_by_dataset.pdf``.

Usage
-----
    python efficiency_plot.py
    python efficiency_plot.py --results-dir results/efficiency --output-dir paper/figures
    python efficiency_plot.py --methods bm25 bm25_hamming rrf_hamming  # subset only
    python efficiency_plot.py --style patterns   # hatch patterns instead of colors
    python efficiency_plot.py --style grayscale  # gray levels instead of colors
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from efficiency import METHODS, _METHOD_COLORS, _METHOD_GRAYS, _METHOD_HATCHES, _style_axes, mathtext_label

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_DIR = REPO_ROOT / "results" / "efficiency"

# BEIR index name (dataset column in the CSVs) -> display name, matching
# paper/sections/experiments.tex.
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
        description="Regenerate the efficiency bar charts from summary.csv / "
        "overall_summary.csv, with BEIR index names mapped to their display names."
    )
    p.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory containing summary.csv and overall_summary.csv (as written by efficiency.py)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write bar_by_dataset.pdf / bar_overall.pdf to (default: --results-dir)",
    )
    p.add_argument(
        "--methods",
        nargs="+",
        default=None,
        choices=list(METHODS.keys()),
        help="Subset of methods to plot (default: every method present in --results-dir's "
        "CSVs, in the canonical order from efficiency.py's METHODS).",
    )
    p.add_argument(
        "--style",
        choices=["color", "patterns", "grayscale"],
        default="color",
        help="Bar chart fill style: 'color' (default), 'patterns' (white fill, black hatch "
        "per method), or 'grayscale' (distinct gray levels per method). 'patterns' and "
        "'grayscale' are colorblind- and print-friendly.",
    )
    return p.parse_args()


def make_plots(
    summary: pd.DataFrame,
    overall_summary: pd.DataFrame,
    datasets: list[str],
    out_dir: Path,
    style: str = "color",
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "pdf.fonttype": 42,  # embed as real (editable/selectable) text, not curves
        }
    )

    method_order = list(overall_summary["method_key"])
    # Labels come from METHODS, not the CSV's "method" column, so the plots
    # always reflect the current names even when regenerated from a CSV
    # written by an older version of METHODS.
    method_labels = [mathtext_label(METHODS[k][0]) for k in method_order]
    dataset_labels = [DATASET_LABELS.get(d, d) for d in datasets]

    # Lookups by method_key, not positional zip: a method's color/hatch/gray
    # is fixed in efficiency.py and must not shift when --methods narrows
    # which of the thirteen this plot includes.
    def method_style(mkey: str) -> dict:
        if style == "patterns":
            # White fill + black hatch: colorblind/print-safe, distinguishable
            # by shape alone.
            return {"color": "white", "edgecolor": "black", "linewidth": 0.8, "hatch": _METHOD_HATCHES[mkey]}
        if style == "grayscale":
            # Distinct gray levels, light to dark, with a black edge so the
            # lightest bars still read against a white background.
            return {"color": _METHOD_GRAYS[mkey], "edgecolor": "black", "linewidth": 0.6}
        return {"color": _METHOD_COLORS[mkey], "edgecolor": "none"}

    # --- Grouped bar chart: latency by dataset x method --------------------
    n_methods = len(method_order)
    x = np.arange(len(datasets))
    width = 0.8 / n_methods

    # A legend row fits ~4 entries readably; more methods need more rows,
    # which need more headroom above the axes and a taller figure.
    legend_ncol = min(4, n_methods)
    legend_rows = math.ceil(n_methods / legend_ncol)
    fig, ax = plt.subplots(figsize=(9, 4.2 + 0.32 * (legend_rows - 1)))
    for i, (mkey, mlabel) in enumerate(zip(method_order, method_labels)):
        sub = summary[summary.method_key == mkey].set_index("dataset").reindex(datasets)
        means = sub["mean_ms"].to_numpy()
        lower = means - sub["ci95_lower_ms"].to_numpy()
        upper = sub["ci95_upper_ms"].to_numpy() - means
        offset = (i - (n_methods - 1) / 2) * width
        ax.bar(
            x + offset,
            means,
            width=width,
            yerr=[lower, upper],
            capsize=2,
            label=mlabel,
            zorder=3,
            **method_style(mkey),
        )
    ax.set_xticks(x)
    ax.set_xticklabels(dataset_labels, rotation=25, ha="right")
    ax.set_ylabel("Mean query latency (ms), log scale")
    ax.set_yscale("log")
    _style_axes(ax)
    legend_y = 1 + 0.1 * legend_rows
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, legend_y), ncol=legend_ncol, frameon=False)
    fig.savefig(out_dir / "bar_by_dataset.pdf", bbox_inches="tight")
    plt.close(fig)

    # --- Bar chart: latency by method, averaged across datasets -----------
    means = overall_summary["mean_ms"].to_numpy()
    lower = means - overall_summary["ci95_lower_ms"].to_numpy()
    upper = overall_summary["ci95_upper_ms"].to_numpy() - means

    fig, ax = plt.subplots(figsize=(max(5.5, 0.5 * n_methods), 4))
    xo = np.arange(n_methods)
    for i, mkey in enumerate(method_order):
        ax.bar(
            xo[i],
            means[i],
            yerr=[[lower[i]], [upper[i]]],
            capsize=4,
            zorder=3,
            **method_style(mkey),
        )
    ax.set_xticks(xo)
    ax.set_xticklabels(method_labels, rotation=30, ha="right")
    ax.set_ylabel("Mean query latency (ms), log scale")
    ax.set_yscale("log")
    ax.set_title(f"Mean $\\pm$ 95% CI across {len(datasets)} datasets")
    _style_axes(ax)
    fig.savefig(out_dir / "bar_overall.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = args.output_dir or args.results_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(args.results_dir / "summary.csv")
    overall_summary = pd.read_csv(args.results_dir / "overall_summary.csv")

    seen = list(dict.fromkeys(summary["dataset"]))
    datasets = [d for d in DATASET_LABELS if d in seen] + [d for d in seen if d not in DATASET_LABELS]

    seen_methods = set(overall_summary["method_key"])
    method_order = [m for m in (args.methods or METHODS) if m in seen_methods]
    if not method_order:
        raise SystemExit("None of the requested --methods are present in the results CSVs.")
    summary = summary[summary.method_key.isin(method_order)]
    overall_summary = overall_summary.set_index("method_key").loc[method_order].reset_index()

    make_plots(summary, overall_summary, datasets, out_dir, style=args.style)
    print(f"Wrote bar_by_dataset.pdf and bar_overall.pdf to {out_dir}/")


if __name__ == "__main__":
    main()
