#!/usr/bin/env python3
"""
effectiveness_plot.py
======================

Renders a heatmap of retrieval effectiveness (dataset x method) straight
from the ``scores.csv`` already written by ``effectiveness.py``, without
re-running the (expensive) qrels evaluation itself -- same pattern as
``plot_efficiency.py`` for ``efficiency.py``'s CSVs.

Table~\\ref{tab:retrieval-effectiveness} has 14 columns (13 scrydb
configurations + the MTEB baseline) and, at up to 4 measures per dataset, is
dense enough that cross-method/cross-dataset patterns (e.g. how tightly the
Hamming/cos_int8/cos_float rerank variants cluster, or where RRF falls behind)
are hard to read off directly. A heatmap of a single measure -- nDCG@10 by
default, the primary metric the paper's prose already treats as a shorthand
for the other three -- makes those patterns visible at a glance, while the
cell text keeps it exact enough to double check against the table. The best
and second-best method per row are boxed solid/dashed, mirroring the
table's \\textbf{}/\\underline{} convention.

Outputs (written to --output-dir)
----------------------------------
  heatmap_<measure>.pdf   -- one heatmap, or a 2x2 grid if --measure all

Usage
-----
    python effectiveness_plot.py
    python effectiveness_plot.py --measure AP
    python effectiveness_plot.py --measure all
    python effectiveness_plot.py --datasets scifact nfcorpus trec-covid
    python effectiveness_plot.py --scores results/effectiveness/scores.csv --output-dir paper/figures/effectiveness

Run `python effectiveness_plot.py --help` for all options.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from effectiveness import BASELINE_KEY, DATASET_LABELS, MEASURES, METHODS
from efficiency import mathtext_label

# Shorter baseline column label than effectiveness.py's BASELINE_LABEL, matching
# the header already used in table.tex.
BASELINE_PLOT_LABEL = "MTEB (Qwen3-8B)"

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCORES = REPO_ROOT / "results" / "effectiveness" / "scores.csv"

# Row order for the heatmap, matching table.tex (Touche before COVID, unlike
# DATASET_LABELS' dict order) rather than sorting alphabetically.
ROW_ORDER = ["arguana", "fiqa", "nfcorpus", "quora", "scidocs", "scifact", "webis-touche2020", "trec-covid"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Render a dataset x method heatmap of retrieval effectiveness from "
        "an existing scores.csv (as written by effectiveness.py), without re-evaluating "
        "any runs."
    )
    p.add_argument("--scores", type=Path, default=DEFAULT_SCORES, help="Path to scores.csv")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write heatmap PDF(s) to (default: alongside --scores)",
    )
    p.add_argument(
        "--measure",
        default="nDCG@10",
        choices=list(MEASURES.keys()) + ["all"],
        help="Which measure to plot (default: nDCG@10, the paper's primary metric), "
        "or 'all' for a 2x2 grid of all four.",
    )
    p.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Subset of datasets to include (default: every dataset present in --scores)",
    )
    p.add_argument("--decimals", type=int, default=3, help="Decimal places shown in each cell")
    p.add_argument(
        "--no-bold-best",
        action="store_true",
        help="Don't box the best-performing method (solid) and second-best (dashed) per row",
    )
    return p.parse_args()


def _slug(measure: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", measure.lower())


def _method_order_and_labels(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    keys = [k for k in METHODS if k in df.method_key.unique()]
    labels = [mathtext_label(METHODS[k][0]) for k in keys]
    if BASELINE_KEY in df.method_key.unique():
        keys.append(BASELINE_KEY)
        labels.append(BASELINE_PLOT_LABEL)
    return keys, labels


def _dataset_order(df: pd.DataFrame, requested: list[str] | None) -> list[str]:
    seen = list(dict.fromkeys(df["dataset"]))
    if requested is not None:
        return [d for d in requested if d in seen]
    return [d for d in ROW_ORDER if d in seen] + [d for d in seen if d not in ROW_ORDER]


def _draw_heatmap(ax, values: np.ndarray, row_labels: list[str], col_labels: list[str], decimals: int, bold_best: bool) -> None:
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap("Purples")
    finite = values[~np.isnan(values)]
    vmin, vmax = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)
    im = ax.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")

    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(col_labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(row_labels), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", length=0)

    norm_mid = vmin + 0.6 * (vmax - vmin)  # cells past this are dark enough to need white text
    for i in range(values.shape[0]):
        row = values[i]
        present = [j for j in range(len(row)) if not np.isnan(row[j])]
        ranked = sorted(present, key=lambda j: row[j], reverse=True)
        best_j = ranked[0] if ranked else None
        second_j = ranked[1] if len(ranked) > 1 else None
        for j in range(values.shape[1]):
            v = row[j]
            if np.isnan(v):
                continue
            text_color = "white" if v >= norm_mid else "black"
            ax.text(j, i, f"{v:.{decimals}f}", ha="center", va="center", color=text_color, fontsize=7.5)
            if bold_best and j == best_j:
                ax.add_patch(
                    plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="black", linewidth=1.6, zorder=5)
                )
            elif bold_best and j == second_j:
                ax.add_patch(
                    plt.Rectangle(
                        (j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="black", linewidth=1.1, linestyle="--", zorder=5
                    )
                )

    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.ax.tick_params(labelsize=8)


def make_plots(
    df: pd.DataFrame,
    datasets: list[str],
    measure: str,
    out_dir: Path,
    decimals: int,
    bold_best: bool,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 9,
            "pdf.fonttype": 42,
        }
    )

    row_labels = [DATASET_LABELS.get(d, d) for d in datasets]
    method_keys, col_labels = _method_order_and_labels(df)

    if measure == "all":
        measures = list(MEASURES.keys())
        fig, axes = plt.subplots(2, 2, figsize=(13, 9))
        for ax, m in zip(axes.flat, measures):
            pivot = df[df.measure == m].pivot_table(index="dataset", columns="method_key", values="value")
            values = pivot.reindex(index=datasets, columns=method_keys).to_numpy(dtype=float)
            _draw_heatmap(ax, values, row_labels, col_labels, decimals, bold_best)
            ax.set_title(m)
        fig.suptitle("Retrieval effectiveness by dataset and method", y=1.0)
        fig.tight_layout()
        out_path = out_dir / "heatmap_all.pdf"
    else:
        pivot = df[df.measure == measure].pivot_table(index="dataset", columns="method_key", values="value")
        values = pivot.reindex(index=datasets, columns=method_keys).to_numpy(dtype=float)
        fig, ax = plt.subplots(figsize=(11, 0.45 * len(datasets) + 2.5))
        _draw_heatmap(ax, values, row_labels, col_labels, decimals, bold_best)
        ax.set_title(f"Retrieval effectiveness ({measure}) by dataset and method")
        fig.tight_layout()
        out_path = out_dir / f"heatmap_{_slug(measure)}.pdf"

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.scores)
    if df.empty:
        raise SystemExit(f"ERROR: {args.scores} is empty.")

    datasets = _dataset_order(df, args.datasets)
    if not datasets:
        raise SystemExit("ERROR: no datasets to plot (check --datasets against scores.csv).")

    out_dir = args.output_dir or args.scores.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = make_plots(
        df,
        datasets,
        args.measure,
        out_dir,
        decimals=args.decimals,
        bold_best=not args.no_bold_best,
    )
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
