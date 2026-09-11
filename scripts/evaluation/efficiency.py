#!/usr/bin/env python3
"""
efficiency.py
=============

Benchmarks *query-time* (wall-clock) retrieval latency of all thirteen
retrieval pipelines exposed by ``scrydb.Index``'s ``mode=``/``precision=``/
``rerank=`` vocabulary, across every BEIR dataset indexed under
``data/indices/beir``:

  BM25                                mode="lexical"
  BM25 + Hamming                      mode="lexical",  rerank="binary"
  BM25 + cos_int8                     mode="lexical",  rerank="int8"
  BM25 + cos_float                    mode="lexical",  rerank="float"
  Hamming                             mode="semantic", precision="binary"
  Hamming + cos_int8                  mode="semantic", precision="binary", rerank="int8"
  Hamming + cos_float                 mode="semantic", precision="binary", rerank="float"
  cos_int8                            mode="semantic", precision="int8"
  cos_int8 + cos_float                mode="semantic", precision="int8",  rerank="float"
  cos_float                           mode="semantic", precision="float"
  RRF(Hamming)                        mode="hybrid",   precision="binary"
  RRF(cos_int8)                       mode="hybrid",   precision="int8"
  RRF(cos_float)                      mode="hybrid",   precision="float"

These are exactly the thirteen pipelines ``scrydb.Index.search``/
``batch_search`` expose via their ``mode=``/``precision=``/``rerank=``
vocabulary (see ``src/search.py``). ``--methods`` (default: all thirteen)
selects which of them a given run actually benchmarks, and the same subset
argument is available on ``plot_efficiency.py``/``latency_table.py`` to
regenerate the paper's table/plots for a chosen subset without re-running
the benchmark.

Design decisions
-----------------
* Every timed call is driven by a *stored query id* via ``batch_search``, so
  it reuses the query's precomputed embedding (populated by
  ``index_queries(..., embedding_field=...)`` when the BEIR indices were
  built) instead of re-encoding query text with a SentenceTransformer. This
  isolates pure *database-side* retrieval cost (SQL execution + any
  Python/NumPy rerank arithmetic) from model-inference latency, which is a
  separate, batchable/cacheable/GPU-dependent concern.

* For each (dataset, method, query), one untimed warm-up call is issued
  first to populate the OS/SQLite page cache for that query's rows, then
  ``--n-reps`` timed repetitions are recorded. Per-query means are computed
  first, and all summary statistics (mean / std / 95% CI) are computed
  *across queries* -- queries, not repeated calls of the same query, are the
  independent sampling unit (repeated calls of the same query share a query
  plan and cache state and would otherwise pseudo-replicate).

* The cross-dataset "overall" comparison treats *datasets* as the sampling
  unit (mean of each dataset's per-query mean, CI across datasets), so that
  a dataset with many more queries than another doesn't dominate the
  average.

Outputs (written to --output-dir)
----------------------------------
  raw_timings.csv        -- every timed call: dataset, method, query_id, rep, ms
  per_query_means.csv    -- mean latency per (dataset, method, query)
  summary.csv            -- mean/std/sem/95% CI per (dataset, method)
  overall_summary.csv    -- mean/std/sem/95% CI per method, across datasets
  table.tex              -- LaTeX table: rows=datasets (+ Mean), columns=methods
  bar_by_dataset.pdf     -- grouped bar chart, latency by dataset x method
  bar_overall.pdf        -- bar chart, latency by method, averaged across datasets

Usage
-----
    python efficiency.py
    python efficiency.py --datasets scifact nfcorpus --n-queries 30 --n-reps 20
    python efficiency.py --methods bm25 bm25_hamming rrf_hamming  # subset only
    python efficiency.py --style patterns   # hatch patterns instead of colors
    python efficiency.py --style grayscale  # gray levels instead of colors

Run `python efficiency.py --help` for all options.
"""

from __future__ import annotations

import argparse
import colorsys
import math
import random
import re
import sys
import time
import warnings
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy import stats as _scipy_stats
except ImportError:  # pragma: no cover - scipy is an optional niceity here
    _scipy_stats = None

# ---------------------------------------------------------------------------
# Paths -- anchored on this file's location (not the cwd), matching
# effectiveness.py, so the script works the same run from the repo root or
# from src/evaluation.
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INDICES_DIR = REPO_ROOT / "data" / "indices" / "beir"

# ---------------------------------------------------------------------------
# Retrieval methods: key -> (column label, batch_search kwargs). Order here
# is the column/bar order everywhere below, grouped by pipeline family
# (lexical-only, semantic-only, hybrid) -- see _METHOD_FAMILY below, which
# drives the plots' color/hatch/gray assignment.
# ---------------------------------------------------------------------------
METHODS = OrderedDict(
    [
        ("bm25", ("BM25", dict(mode="lexical"))),
        ("bm25_hamming", ("BM25 + Hamming", dict(mode="lexical", rerank="binary"))),
        ("bm25_cosine_int8", ("BM25 + cos_int8", dict(mode="lexical", rerank="int8"))),
        ("bm25_cosine_float", ("BM25 + cos_float", dict(mode="lexical", rerank="float"))),
        ("hamming", ("Hamming", dict(mode="semantic", precision="binary"))),
        (
            "hamming_cosine_int8",
            ("Hamming + cos_int8", dict(mode="semantic", precision="binary", rerank="int8")),
        ),
        (
            "hamming_cosine_float",
            ("Hamming + cos_float", dict(mode="semantic", precision="binary", rerank="float")),
        ),
        ("cosine_int8", ("cos_int8", dict(mode="semantic", precision="int8"))),
        (
            "cosine_int8_float",
            ("cos_int8 + cos_float", dict(mode="semantic", precision="int8", rerank="float")),
        ),
        ("cosine_float", ("cos_float", dict(mode="semantic", precision="float"))),
        ("rrf_hamming", ("RRF(Hamming)", dict(mode="hybrid", precision="binary"))),
        ("rrf_cosine_int8", ("RRF(cos_int8)", dict(mode="hybrid", precision="int8"))),
        ("rrf_cosine_float", ("RRF(cos_float)", dict(mode="hybrid", precision="float"))),
    ]
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark scrydb.Index query latency for all thirteen retrieval "
        "pipelines (BM25; BM25 + Hamming/cos_int8/cos_float; Hamming; "
        "Hamming/cos_int8/cos_int8+float/cos_float; and the three "
        "RRF variants), across every BEIR dataset."
    )
    p.add_argument(
        "--indices-dir",
        type=Path,
        default=DEFAULT_INDICES_DIR,
        help="Directory containing one <dataset>.db scrydb index per BEIR dataset",
    )
    p.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Subset of datasets to benchmark (default: auto-discover *.db in --indices-dir)",
    )
    p.add_argument(
        "--vec-ext",
        default="auto",
        help="Path to a compiled sqlite-vec extension build ('vec_ext_path' in "
        "scrydb.Index.open), 'auto' (default) to use the one bundled with the "
        "installed sqlite-vec package, or 'none' to skip loading it (only 'bm25' "
        "will then work -- every other method needs semantic/hybrid vector search).",
    )
    p.add_argument(
        "--n-queries",
        type=int,
        default=50,
        help="Number of queries to sample per dataset (uses all available if fewer).",
    )
    p.add_argument(
        "--n-reps",
        type=int,
        default=20,
        help="Timed repetitions per (dataset, method, query), plus one untimed warm-up call.",
    )
    p.add_argument("--top-k", type=int, default=10, help="Final results returned per query")
    p.add_argument(
        "--rerank-depth",
        type=int,
        default=100,
        help="Candidate pool depth fed into the Hamming/Cosine rerank stage "
        "('rerank_depth' in scrydb.Index.batch_search).",
    )
    p.add_argument(
        "--candidate-limit",
        type=int,
        default=50,
        help="Per-stage candidate pool size for the hybrid RRF fusion (lexical and "
        "semantic sides), matching scrydb.Index.batch_search's 'candidate_limit'.",
    )
    p.add_argument("--rrf-k", type=int, default=60, help="RRF smoothing constant")
    p.add_argument("--seed", type=int, default=42, help="Random seed for query sampling")
    p.add_argument("--output-dir", type=Path, default=Path("./results/efficiency"))
    p.add_argument(
        "--methods",
        nargs="+",
        default=list(METHODS.keys()),
        choices=list(METHODS.keys()),
        help="Subset of methods to run (default: all thirteen)",
    )
    p.add_argument("--decimals", type=int, default=2, help="Decimal places (ms) in the LaTeX table")
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
    p.add_argument(
        "--style",
        choices=["color", "patterns", "grayscale"],
        default="color",
        help="Bar chart fill style: 'color' (default), 'patterns' (white fill, black hatch "
        "per method), or 'grayscale' (distinct gray levels per method). 'patterns' and "
        "'grayscale' are colorblind- and print-friendly.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------
def discover_datasets(indices_dir: Path) -> list[str]:
    return sorted(f.stem for f in indices_dir.glob("*.db"))


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------
def time_call(fn) -> float:
    """Run *fn* once and return elapsed wall-clock time in milliseconds."""
    t0 = time.perf_counter()
    fn()
    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0


def mean_std_sem_ci(values: np.ndarray, confidence: float = 0.95) -> dict:
    """Return mean, sample std (ddof=1), sem, and a (lower, upper) CI."""
    n = len(values)
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if n > 1 else 0.0
    sem = std / math.sqrt(n) if n > 1 else 0.0

    if n > 1:
        if _scipy_stats is not None:
            t_crit = _scipy_stats.t.ppf((1 + confidence) / 2.0, df=n - 1)
        else:
            t_crit = 1.959963984540054  # normal-approximation fallback
        margin = t_crit * sem
    else:
        margin = 0.0

    return {
        "n": n,
        "mean_ms": mean,
        "std_ms": std,
        "sem_ms": sem,
        "ci95_lower_ms": mean - margin,
        "ci95_upper_ms": mean + margin,
    }


# ---------------------------------------------------------------------------
# Benchmark loop
# ---------------------------------------------------------------------------
def benchmark_dataset(index, dataset: str, args, query_ids: list[str]) -> list[dict]:
    records = []
    total = len(args.methods) * len(query_ids)
    done = 0
    t_start = time.time()

    for method_key in args.methods:
        label, kwargs = METHODS[method_key]
        for qid in query_ids:
            done += 1

            def call(qid=qid, kwargs=kwargs):
                return index.batch_search(
                    queries=qid,
                    top_k=args.top_k,
                    rerank_depth=args.rerank_depth,
                    candidate_limit=args.candidate_limit,
                    rrf_k=args.rrf_k,
                    **kwargs,
                )

            try:
                call()  # untimed warm-up
            except Exception as exc:  # pragma: no cover
                warnings.warn(f"[{dataset}/{label}] warm-up failed for query {qid!r}: {exc}")
                continue

            for rep in range(args.n_reps):
                try:
                    elapsed_ms = time_call(call)
                except Exception as exc:
                    warnings.warn(f"[{dataset}/{label}] call failed for query {qid!r}, rep {rep}: {exc}")
                    continue
                records.append(
                    {
                        "dataset": dataset,
                        "method_key": method_key,
                        "method": label,
                        "query_id": qid,
                        "rep": rep,
                        "elapsed_ms": elapsed_ms,
                    }
                )

            if done % 20 == 0 or done == total:
                print(
                    f"  [{dataset}] {done}/{total} method-query combos "
                    f"({time.time() - t_start:0.1f}s elapsed)",
                    file=sys.stderr,
                )
    return records


def run_benchmark(args, datasets: list[str]) -> pd.DataFrame:
    from scrydb import Index
    import scrydb.core as scrydb_core

    # Each timed call below invokes batch_search for a single query id; its
    # internal tqdm progress bar would otherwise be constructed (and
    # printed) tens of thousands of times over the run. This is a fixed
    # per-call cost applied identically to every method, so it wouldn't bias
    # the comparison, but suppressing it gives a cleaner console.
    scrydb_core.tqdm = lambda iterable, **kwargs: iterable

    vec_ext = None if args.vec_ext.lower() == "none" else args.vec_ext
    rng = random.Random(args.seed)
    all_records: list[dict] = []

    for dataset in datasets:
        db_path = args.indices_dir / f"{dataset}.db"
        index = Index.open(db_path=db_path, vec_ext_path=vec_ext)

        all_query_ids = list(index.queries)
        if not all_query_ids:
            print(f"WARNING: no queries stored in {db_path}, skipping.", file=sys.stderr)
            index.close()
            continue

        if len(all_query_ids) > args.n_queries:
            query_ids = rng.sample(all_query_ids, args.n_queries)
        else:
            query_ids = list(all_query_ids)
            print(
                f"NOTE: [{dataset}] only {len(all_query_ids)} queries available "
                f"(< requested {args.n_queries}); using all of them.",
                file=sys.stderr,
            )

        print(
            f"Benchmarking '{dataset}' ({len(index.documents)} docs): "
            f"{len(args.methods)} method(s) x {len(query_ids)} queries x "
            f"{args.n_reps} reps ...",
            file=sys.stderr,
        )
        all_records.extend(benchmark_dataset(index, dataset, args, query_ids))
        index.close()

    return pd.DataFrame.from_records(all_records)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def aggregate(raw_df: pd.DataFrame, datasets: list[str], method_order: list[str]):
    """Return (per_query_means, summary, overall_summary)."""
    per_query = (
        raw_df.groupby(["dataset", "method_key", "method", "query_id"], as_index=False)["elapsed_ms"]
        .mean()
        .rename(columns={"elapsed_ms": "mean_elapsed_ms"})
    )

    summary_rows = []
    for dataset in datasets:
        for method_key in method_order:
            mask = (per_query.dataset == dataset) & (per_query.method_key == method_key)
            values = per_query.loc[mask, "mean_elapsed_ms"].to_numpy()
            if len(values) == 0:
                continue
            label = per_query.loc[mask, "method"].iloc[0]
            stats = mean_std_sem_ci(values)
            stats.update({"dataset": dataset, "method_key": method_key, "method": label})
            summary_rows.append(stats)
    cols = ["dataset", "method_key", "method", "n", "mean_ms", "std_ms", "sem_ms", "ci95_lower_ms", "ci95_upper_ms"]
    summary = pd.DataFrame(summary_rows)[cols]

    # Overall: datasets are the sampling unit (one value per dataset -- its
    # per-query mean latency), so a dataset with many more sampled queries
    # than another doesn't dominate the cross-dataset average.
    overall_rows = []
    for method_key in method_order:
        mask = summary.method_key == method_key
        values = summary.loc[mask, "mean_ms"].to_numpy()
        if len(values) == 0:
            continue
        label = summary.loc[mask, "method"].iloc[0]
        stats = mean_std_sem_ci(values)
        stats.update({"method_key": method_key, "method": label})
        overall_rows.append(stats)
    overall_cols = ["method_key", "method", "n", "mean_ms", "std_ms", "sem_ms", "ci95_lower_ms", "ci95_upper_ms"]
    overall_summary = pd.DataFrame(overall_rows)[overall_cols]

    return per_query, summary, overall_summary


# ---------------------------------------------------------------------------
# Plotting -- PDF, tight layout, sized/styled for a scientific publication.
#
# With thirteen methods, thirteen arbitrarily distinct hues would blow past
# the ~8-hue budget for a colorblind-safe categorical palette. Instead each
# method's *family* (lexical-only, semantic-only, hybrid) gets one of three
# validated, mutually CVD-safe hues, and methods within a family are shades
# of that hue (lighter = more rerank stages layered on top). Both the family
# and a method's position within it are fixed here -- independent of which
# subset --methods actually selects for a given run -- so a method's color
# never changes across differently-filtered plots/tables.
# ---------------------------------------------------------------------------
_METHOD_FAMILY = OrderedDict(
    [
        ("bm25", "lexical"),
        ("bm25_hamming", "lexical"),
        ("bm25_cosine_int8", "lexical"),
        ("bm25_cosine_float", "lexical"),
        ("hamming", "semantic"),
        ("hamming_cosine_int8", "semantic"),
        ("hamming_cosine_float", "semantic"),
        ("cosine_int8", "semantic"),
        ("cosine_int8_float", "semantic"),
        ("cosine_float", "semantic"),
        ("rrf_hamming", "hybrid"),
        ("rrf_cosine_int8", "hybrid"),
        ("rrf_cosine_float", "hybrid"),
    ]
)
assert list(_METHOD_FAMILY) == list(METHODS), "_METHOD_FAMILY must cover exactly METHODS, same order"

# First three slots of the validated default categorical palette (blue,
# orange, aqua) -- the only three that stay CVD-safe even under the strict
# all-pairs test, not just adjacent-pair, so it's safe to use them as a
# legend and a grouped bar chart both put every pair on screen together.
_FAMILY_BASE_COLOR = {"lexical": "#2a78d6", "semantic": "#eb6834", "hybrid": "#1baf7a"}
_FAMILY_HATCH_CHAR = {"lexical": "/", "semantic": "x", "hybrid": "o"}


def _family_members() -> "OrderedDict[str, list[str]]":
    members: "OrderedDict[str, list[str]]" = OrderedDict((fam, []) for fam in _FAMILY_BASE_COLOR)
    for key, fam in _METHOD_FAMILY.items():
        members[fam].append(key)
    return members


def _lighten(hex_color: str, amount: float) -> str:
    """Lighten *hex_color* toward white by *amount* in [0, 1] (HLS lightness)."""
    r, g, b = (int(hex_color[i : i + 2], 16) / 255 for i in (1, 3, 5))
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    l = l + (1 - l) * amount
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return "#{:02x}{:02x}{:02x}".format(round(r * 255), round(g * 255), round(b * 255))


def _build_method_colors() -> dict[str, str]:
    """method_key -> hex. One base hue per family, evenly-spaced lighter
    shades for the family's later (more heavily reranked) members."""
    colors = {}
    for fam, members in _family_members().items():
        base = _FAMILY_BASE_COLOR[fam]
        n = len(members)
        for i, key in enumerate(members):
            amount = 0.0 if n <= 1 else (i / (n - 1)) * 0.55  # cap so the lightest shade stays legible
            colors[key] = _lighten(base, amount)
    return colors


def _build_method_hatches() -> dict[str, str]:
    """method_key -> hatch string. Hatch *shape* encodes family, hatch
    *density* (repeat count) encodes position within the family."""
    hatches = {}
    for fam, members in _family_members().items():
        ch = _FAMILY_HATCH_CHAR[fam]
        for i, key in enumerate(members):
            hatches[key] = ch * (i + 1)
    return hatches


# Alternatives to _METHOD_COLORS, selected via --style to keep the bar
# charts distinguishable for color-blind readers and in black-and-white
# print:
#   "patterns"  -- white fill, black hatch
#   "grayscale" -- distinct gray levels, light to dark
_METHOD_COLORS = _build_method_colors()
_METHOD_HATCHES = _build_method_hatches()
_METHOD_GRAYS = {
    key: f"{g:.2f}" for key, g in zip(_METHOD_FAMILY, np.linspace(0.85, 0.1, num=len(_METHOD_FAMILY)))
}
PLOT_STYLES = ("color", "patterns", "grayscale")


def _style_axes(ax) -> None:
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def make_plots(
    summary: pd.DataFrame,
    overall_summary: pd.DataFrame,
    datasets: list[str],
    out_dir: Path,
    style: str = "color",
) -> None:
    if style not in PLOT_STYLES:
        raise ValueError(f"style must be one of {PLOT_STYLES}, got {style!r}")
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
    method_labels = list(overall_summary["method"])
    # Lookups by method_key, not positional zip: a method's color/hatch/gray
    # is fixed (see _METHOD_COLORS etc. above) and must not shift when
    # --methods narrows which of the thirteen a given plot includes.
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
    ax.set_xticklabels(datasets, rotation=25, ha="right")
    ax.set_ylabel("Mean query latency (ms), log scale")
    ax.set_yscale("log")
    _style_axes(ax)
    legend_y = 1.02 + 0.12 * legend_rows
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


# ---------------------------------------------------------------------------
# LaTeX table -- rows are datasets (+ a cross-dataset Mean row), columns are
# methods, mirroring effectiveness.py's to_latex() but for latency (lower is
# better, so the fastest method per row is bolded rather than the highest).
# ---------------------------------------------------------------------------
# Matches the "cos_int8"/"cos_float" tokens used in METHODS labels, so they
# can be rendered as a real subscript (LaTeX \textsubscript, matplotlib
# mathtext) matching the cos_{int8}/cos_{float} notation used in the paper's
# prose (see paper/sections/experiments.tex).
_SUBSCRIPT_RE = re.compile(r"cos_(int8|float)")


def _tex_subscript(label: str) -> str:
    """Render "cos_int8"/"cos_float" tokens in *label* as LaTeX
    \\textsubscript{}, e.g. "cos_int8" -> "cos\\textsubscript{int8}"."""
    return _SUBSCRIPT_RE.sub(r"cos\\textsubscript{\1}", label)


def mathtext_label(label: str) -> str:
    """Render *label* for matplotlib tick/legend text, with "cos_int8"/
    "cos_float" tokens as a real subscript via mathtext (e.g. "cos_int8" ->
    "cos$_{\\mathrm{int8}}$"), matching the paper's cos\\textsubscript{}
    notation as closely as plain matplotlib text allows. ``\\mathrm{}`` keeps
    the subscript upright -- mathtext otherwise renders bare math-mode text
    in italics."""
    return _SUBSCRIPT_RE.sub(lambda m: f"cos$_{{\\mathrm{{{m.group(1)}}}}}$", label)


def shortstack_label(label: str) -> str:
    """Wrap *label* as a left-aligned two-line ``\\shortstack``, breaking
    right before " + " (e.g. "BM25 + cos_float" -> "BM25" / "+ cos_float")
    -- keeps multi-word method names narrow without needing to rotate them.
    Labels without a "+" (e.g. "cos_float") stay on one line. "cos_int8"/
    "cos_float" tokens are rendered as \\textsubscript{}, matching the
    cos\\textsubscript{} notation used for retrieval method names in the
    paper's prose."""
    if " + " not in label:
        return f"\\shortstack[l]{{{_tex_subscript(label)}}}"
    line1, rest = label.split(" + ", 1)
    line2 = f"+ {_tex_subscript(rest)}"
    return f"\\shortstack[l]{{{_tex_subscript(line1)}\\\\{line2}}}"


def to_latex(
    summary: pd.DataFrame,
    overall_summary: pd.DataFrame,
    datasets: list[str],
    method_order: list[str],
    decimals: int,
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
    lines.append("% Auto-generated by src/evaluation/efficiency.py -- do not edit by hand.")
    lines.append("% Requires \\usepackage{booktabs} in the preamble.")
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\resizebox{\\textwidth}{!}{%")
    col_spec = "|".join(["l"] + ["r"] * len(method_order))
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")
    method_headers = [
        f"\\rotatebox{{{header_angle}}}{{{shortstack_label(labels[k])}}}"
        if header_angle
        else shortstack_label(labels[k])
        for k in method_order
    ]
    lines.append(" & ".join(["Dataset"] + method_headers) + " \\\\")
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
                cell = f"{v:.{decimals}f}"
                if k == best_key:
                    cell = f"\\textbf{{{cell}}}"
                elif k == second_key:
                    cell = f"\\underline{{{cell}}}"
            cells.append(cell)
        return cells

    for dataset in datasets:
        if dataset not in pivot.index:
            continue
        lines.append(" & ".join([dataset] + row_cells(pivot.loc[dataset])) + " \\\\")

    lines.append("\\midrule")
    lines.append(" & ".join(["\\textbf{Mean}"] + row_cells(overall_row)) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}%")
    lines.append("}")
    lines.append(
        "\\caption{Mean query latency (ms) of each retrieval strategy, per BEIR dataset "
        "and averaged across datasets (\\textbf{Mean} row). Fastest method per row in "
        "\\textbf{bold}, second-fastest \\underline{underlined}.}"
    )
    lines.append("\\label{tab:retrieval-efficiency}")
    lines.append("\\end{table}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    datasets = args.datasets or discover_datasets(args.indices_dir)
    if not datasets:
        print(f"ERROR: no *.db indices found under {args.indices_dir}.", file=sys.stderr)
        sys.exit(1)
    print(f"Datasets: {', '.join(datasets)}", file=sys.stderr)
    print(f"Methods: {', '.join(args.methods)}", file=sys.stderr)

    raw_df = run_benchmark(args, datasets)
    if raw_df.empty:
        print("ERROR: no successful timed calls were recorded.", file=sys.stderr)
        sys.exit(1)

    datasets = [d for d in datasets if d in raw_df.dataset.unique()]
    method_order = [k for k in args.methods if k in raw_df.method_key.unique()]

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_df.to_csv(out_dir / "raw_timings.csv", index=False)

    per_query, summary, overall_summary = aggregate(raw_df, datasets, method_order)
    per_query.to_csv(out_dir / "per_query_means.csv", index=False)
    summary.to_csv(out_dir / "summary.csv", index=False)
    overall_summary.to_csv(out_dir / "overall_summary.csv", index=False)

    make_plots(summary, overall_summary, datasets, out_dir, style=args.style)

    latex = to_latex(
        summary,
        overall_summary,
        datasets,
        method_order,
        decimals=args.decimals,
        bold_best=not args.no_bold_best,
        header_angle=args.header_angle,
    )
    (out_dir / "table.tex").write_text(latex + "\n", encoding="utf-8")

    pd.set_option("display.float_format", lambda v: f"{v:0.3f}")
    wide = summary.pivot_table(index="dataset", columns="method_key", values="mean_ms")
    wide = wide.reindex(columns=method_order).rename(columns=dict(zip(overall_summary.method_key, overall_summary.method)))
    wide = wide.reindex(datasets)
    print("\n=== Mean query latency (ms) per dataset / method ===")
    print(wide.to_string())
    print("\n=== Mean query latency (ms) per method, averaged across datasets ===")
    print(
        overall_summary.set_index("method")[["n", "mean_ms", "std_ms", "ci95_lower_ms", "ci95_upper_ms"]].to_string()
    )
    print(
        f"\nWrote raw_timings.csv, per_query_means.csv, summary.csv, overall_summary.csv, "
        f"table.tex, bar_by_dataset.pdf, and bar_overall.pdf to {out_dir}/"
    )


if __name__ == "__main__":
    main()
