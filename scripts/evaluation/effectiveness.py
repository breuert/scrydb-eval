#!/usr/bin/env python3
"""
effectiveness.py
================

Evaluates retrieval effectiveness for every (dataset, retrieval method) run
found under ``data/runs/beir`` against its TREC-format qrels, and compares
the results against the full-precision embedding baseline reported by MTEB
for ``Qwen/Qwen3-Embedding-8B``.

Retrieval methods (columns of the final table), matching the ``mode=``/
``precision=``/``rerank=`` parameters of ``scrydb.Index.search``/
``batch_search``:

  BM25                              mode="lexical",  rerank=False
  BM25 + Hamming                    mode="lexical",  rerank="binary",  rerank_depth=1000
  BM25 + cos_int8                   mode="lexical",  rerank="int8",    rerank_depth=1000
  BM25 + cos_float                  mode="lexical",  rerank="float",   rerank_depth=1000
  Hamming                           mode="semantic", precision="binary"
  Hamming + cos_int8                mode="semantic", precision="binary", rerank="int8",  rerank_depth=1000
  Hamming + cos_float               mode="semantic", precision="binary", rerank="float", rerank_depth=1000
  cos_int8                          mode="semantic", precision="int8"
  cos_int8 + cos_float              mode="semantic", precision="int8",   rerank="float", rerank_depth=1000
  cos_float                         mode="semantic", precision="float"
  RRF(Hamming)                      mode="hybrid",   precision="binary", candidate_limit=1000
  RRF(cos_int8)                     mode="hybrid",   precision="int8",   candidate_limit=1000
  RRF(cos_float)                    mode="hybrid",   precision="float",  candidate_limit=1000
  MTEB (Qwen3-8B)                   MTEB-reported full-precision embeddings

Run files are matched by filename suffix (see METHODS below), e.g. for
"trec-covid":

  trec-covid-bm25.txt                       -> BM25
  trec-covid-bm25-hamming.txt                -> BM25 + Hamming
  trec-covid-bm25-cosine_int8.txt            -> BM25 + cos_int8
  trec-covid-bm25-cosine_float.txt           -> BM25 + cos_float
  trec-covid-hamming.txt                     -> Hamming
  trec-covid-hamming-cosine_int8.txt         -> Hamming + cos_int8
  trec-covid-hamming-cosine_float.txt        -> Hamming + cos_float
  trec-covid-cosine_int8.txt                 -> cos_int8
  trec-covid-cosine_int8-cosine_float.txt    -> cos_int8 + cos_float
  trec-covid-cosine_float.txt                -> cos_float
  trec-covid-rrf-hamming.txt                 -> RRF(Hamming)
  trec-covid-rrf-cosine_int8.txt             -> RRF(cos_int8)
  trec-covid-rrf-cosine_float.txt            -> RRF(cos_float)

Datasets are auto-discovered from whichever "*-bm25.txt" (BM25) run
files are present, so adding a new dataset's runs to data/runs/beir is
enough for it to show up here -- nothing needs to be hardcoded.

Measures: AP, RR, P@10, nDCG@10 (via repro_eval/ir_measures/pytrec_eval on
the run + qrels), matched against MTEB's map_at_1000, mrr_at_1000,
precision_at_10, ndcg_at_10. Both sides are computed to the same rank depth
(the runs are all "-1000" files, and MTEB reports the same "_at_1000" /
"_at_10" cutoffs), so the two are directly comparable even though they come
from different evaluation libraries.

Self-matches: BEIR draws some datasets' queries from the corpus itself, so a
query's own document can be retrieved for it. BEIR's reference implementation
never scores that document, and the MTEB baseline inherits the exclusion, so
every run is filtered the same way here before it is evaluated (see
``strip_self_matches``); ``--keep-self-matches`` restores the unfiltered
behaviour for quantifying the artifact. Of the eight datasets only ArguAna is
materially affected -- the self-match takes rank 1 for 91% of its queries, and
excluding it raises cos_float from 0.514 to 0.724 nDCG@10 -- with a further 9
incidental id collisions on FiQA and none anywhere else.

Outputs (written to --output-dir):
  scores.csv   -- long-format table: dataset, method_key, method, measure, value
  table.tex    -- LaTeX table, methods as columns, rows grouped by dataset

Usage
-----
    python effectiveness.py
    python effectiveness.py --datasets trec-covid scifact --output-dir ./out

Run `python effectiveness.py --help` for all options.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from repro_eval.Evaluator import RpdEvaluator

from efficiency import _tex_subscript

# ---------------------------------------------------------------------------
# Paths -- anchored on this file's location (not the cwd) so the script
# works the same whether it's run from the repo root or from src/evaluation.
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNS_DIR = REPO_ROOT / "data" / "runs" / "beir"
DEFAULT_QRELS_DIR = REPO_ROOT / "data" / "datasets" / "beir"
DEFAULT_MTEB_DIR = (
    REPO_ROOT
    / "data"
    / "mteb"
    / "results"
    / "results"
    / "Qwen__Qwen3-Embedding-8B"
    / "4e423935c619ae4df87b646a3ce949610c66241c"
)

# ---------------------------------------------------------------------------
# Retrieval methods: key -> (column label, run filename suffix). Order here
# is the column order of the final table. Datasets are discovered from
# whichever files exist for the "bm25" (BM25) entry below.
# ---------------------------------------------------------------------------
METHODS = OrderedDict(
    [
        ("bm25", ("BM25", "-bm25.txt")),
        ("bm25_hamming", ("BM25 + Hamming", "-bm25-hamming.txt")),
        ("bm25_cosine_int8", ("BM25 + cos_int8", "-bm25-cosine_int8.txt")),
        ("bm25_cosine_float", ("BM25 + cos_float", "-bm25-cosine_float.txt")),
        ("hamming", ("Hamming", "-hamming.txt")),
        ("hamming_cosine_int8", ("Hamming + cos_int8", "-hamming-cosine_int8.txt")),
        ("hamming_cosine_float", ("Hamming + cos_float", "-hamming-cosine_float.txt")),
        ("cosine_int8", ("cos_int8", "-cosine_int8.txt")),
        ("cosine_int8_cosine_float", ("cos_int8 + cos_float", "-cosine_int8-cosine_float.txt")),
        ("cosine_float", ("cos_float", "-cosine_float.txt")),
        ("hybrid_hamming", ("RRF(Hamming)", "-rrf-hamming.txt")),
        ("hybrid_cosine_int8", ("RRF(cos_int8)", "-rrf-cosine_int8.txt")),
        ("hybrid_cosine_float", ("RRF(cos_float)", "-rrf-cosine_float.txt")),
    ]
)
BASELINE_KEY = "baseline"
BASELINE_LABEL = "MTEB (Qwen3-8B)"

# Measures reported (repro_eval/ir_measures key -> MTEB key), in the order
# they appear as rows within each dataset's block.
MEASURES = OrderedDict(
    [
        ("AP", "map_at_1000"),
        ("RR", "mrr_at_1000"),
        ("P@10", "precision_at_10"),
        ("nDCG@10", "ndcg_at_10"),
    ]
)

# BEIR/scrydb dataset name -> MTEB result filename. Includes datasets that
# don't have runs yet so newly added runs pick up a baseline automatically.
DATASET_TO_MTEB_FILE = {
    "arguana": "ArguAna.json",
    "fiqa": "FiQA2018.json",
    "nfcorpus": "NFCorpus.json",
    "quora": "QuoraRetrieval.json",
    "scidocs": "SCIDOCS.json",
    "scifact": "SciFact.json",
    "trec-covid": "TRECCOVID.json",
    "webis-touche2020": "Touche2020.json",
}

# BEIR/scrydb dataset name -> display name for the first column of the
# LaTeX table.
DATASET_LABELS = {
    "arguana": "ArguAna",
    "fiqa": "FiQA",
    "nfcorpus": "NFCorpus",
    "quora": "Quora",
    "scidocs": "SciDocs",
    "scifact": "SciFact",
    "webis-touche2020": "Touché",
    "trec-covid": "COVID",
}

# Row-block order of the LaTeX table. Alphabetical by display name except
# for Touché, which precedes COVID. Datasets missing here (a newly added
# one) are appended in alphabetical order of their directory name.
DATASET_ORDER = list(DATASET_LABELS)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate all BEIR runs against their qrels and compare "
        "against the MTEB full-precision embedding baseline."
    )
    p.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    p.add_argument("--qrels-dir", type=Path, default=DEFAULT_QRELS_DIR)
    p.add_argument("--mteb-dir", type=Path, default=DEFAULT_MTEB_DIR)
    p.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Subset of datasets to evaluate (default: auto-discover from --runs-dir)",
    )
    p.add_argument(
        "--qrels-split",
        default="test",
        help="qrels split filename stem under <dataset>/qrels/ (default: test)",
    )
    p.add_argument("--output-dir", type=Path, default=Path("./results/effectiveness"))
    p.add_argument(
        "--keep-self-matches",
        action="store_true",
        help="Score each query's own document if the run retrieved it. BEIR's "
        "reference implementation excludes it, so the default (excluding it) is "
        "what makes these runs comparable to the MTEB baseline; this flag is for "
        "quantifying the artifact, not for reporting.",
    )
    p.add_argument("--decimals", type=int, default=3, help="Decimal places in the LaTeX table")
    p.add_argument(
        "--no-bold-best",
        action="store_true",
        help="Don't bold the best-performing method and underline the "
        "second-best (including the baseline) in each row; ties share the "
        "mark",
    )
    p.add_argument(
        "--header-angle",
        type=int,
        default=90,
        help="Rotate retrieval-method column headers by this many degrees "
        "for a denser table (0 to disable rotation, default: 90)",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------
def discover_datasets(runs_dir: Path) -> list[str]:
    _, anchor_suffix = METHODS["bm25"]
    datasets = sorted(
        f.name[: -len(anchor_suffix)]
        for f in runs_dir.glob(f"*{anchor_suffix}")
        if f.name.endswith(anchor_suffix)
    )
    return sorted(datasets, key=lambda d: (DATASET_ORDER.index(d) if d in DATASET_ORDER else len(DATASET_ORDER), d))


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def strip_self_matches(run_path: Path, out_path: Path) -> int:
    """Write *run_path* to *out_path* without each query's own document.

    Several BEIR datasets draw their queries from the corpus itself, so the
    query document is trivially its own nearest neighbour: on ArguAna it is
    retrieved for 1298 of 1406 queries and takes rank 1 in 1282 of them,
    consuming a top-10 slot that BEIR's reference implementation never
    offers. That implementation drops the self-match while collecting
    results (``if corpus_id != query_id`` in ``exact_search.py``), so the
    MTEB figures we compare against are computed without it and our runs
    have to be filtered the same way to be comparable.

    Ranks are renumbered contiguously from 1 after the removal. Scores are
    left untouched -- pytrec_eval (via repro_eval) ranks by score and
    ignores the rank column, so renumbering is for the benefit of anyone
    reading the filtered file rather than for the evaluation itself.

    Returns the number of removed lines.
    """
    removed = 0
    kept_per_query: defaultdict[str, int] = defaultdict(int)
    with run_path.open("r", encoding="utf-8") as src, out_path.open("w", encoding="utf-8") as dst:
        for line in src:
            fields = line.split()
            if len(fields) < 6:
                dst.write(line)
                continue
            qid, q0, docid, _rank, score, tag = fields[:6]
            if docid == qid:
                removed += 1
                continue
            kept_per_query[qid] += 1
            dst.write(f"{qid}\t{q0}\t{docid}\t{kept_per_query[qid]}\t{score}\t{tag}\n")
    return removed


def evaluate_run(run_path: Path, qrels_path: Path, drop_self_matches: bool = True) -> dict[str, float]:
    """Mean AP / RR / P@10 / nDCG@10 across all queries in *run_path*."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        if drop_self_matches:
            filtered = Path(tmp_dir) / run_path.name
            removed = strip_self_matches(run_path, filtered)
            if removed:
                print(f"  dropped {removed} self-match(es) from {run_path.name}", file=sys.stderr)
            run_path = filtered
        rpd_eval = RpdEvaluator(qrels_orig_path=str(qrels_path), run_b_orig_path=str(run_path))
        rpd_eval.evaluate()
        scores = rpd_eval.run_b_orig_score
    return {measure: float(np.mean([s[measure] for s in scores.values()])) for measure in MEASURES}


def load_baseline(mteb_path: Path) -> dict[str, float]:
    with mteb_path.open("r", encoding="utf-8") as f:
        result = json.load(f)
    test_scores = result["scores"]["test"][0]
    return {measure: float(test_scores[mteb_key]) for measure, mteb_key in MEASURES.items()}


def collect_scores(args: argparse.Namespace, datasets: list[str]) -> pd.DataFrame:
    """Long-format DataFrame: dataset, method_key, method, measure, value."""
    records = []

    for dataset in datasets:
        qrels_path = args.qrels_dir / dataset / "qrels" / f"{args.qrels_split}.trec"
        if not qrels_path.exists():
            print(f"WARNING: no qrels for '{dataset}' at {qrels_path}, skipping.", file=sys.stderr)
            continue

        for method_key, (method_label, suffix) in METHODS.items():
            run_path = args.runs_dir / f"{dataset}{suffix}"
            if not run_path.exists():
                print(f"NOTE: no run for '{dataset}' / '{method_label}' ({run_path.name}), skipping.", file=sys.stderr)
                continue
            print(f"Evaluating {dataset} / {method_label} ...", file=sys.stderr)
            scores = evaluate_run(run_path, qrels_path, drop_self_matches=not args.keep_self_matches)
            for measure, value in scores.items():
                records.append(
                    {
                        "dataset": dataset,
                        "method_key": method_key,
                        "method": method_label,
                        "measure": measure,
                        "value": value,
                    }
                )

        mteb_file = DATASET_TO_MTEB_FILE.get(dataset)
        mteb_path = args.mteb_dir / mteb_file if mteb_file else None
        if mteb_path is None or not mteb_path.exists():
            print(f"WARNING: no MTEB baseline for '{dataset}' at {mteb_path}, skipping baseline.", file=sys.stderr)
            continue
        baseline_scores = load_baseline(mteb_path)
        for measure, value in baseline_scores.items():
            records.append(
                {
                    "dataset": dataset,
                    "method_key": BASELINE_KEY,
                    "method": BASELINE_LABEL,
                    "measure": measure,
                    "value": value,
                }
            )

    return pd.DataFrame.from_records(records)


# ---------------------------------------------------------------------------
# LaTeX table
# ---------------------------------------------------------------------------
def header_label(label: str) -> str:
    """Wrap *label* as a single-line left-aligned ``\\shortstack`` for the
    column header, e.g. "BM25 + cos_float" -> "BM25 + cos\\textsubscript{float}".
    Unlike the efficiency table's two-line ``shortstack_label``, the method
    names stay on one line here -- rotated by 90 degrees they run along the
    table's height, where there is room for them."""
    return f"\\shortstack[l]{{{_tex_subscript(label)}}}"


def to_latex(
    df: pd.DataFrame,
    datasets: list[str],
    decimals: int,
    bold_best: bool,
    header_angle: int = 90,
) -> str:
    method_order = [k for k in METHODS if k in df.method_key.unique()]
    all_columns = method_order + ([BASELINE_KEY] if BASELINE_KEY in df.method_key.unique() else [])
    labels = {**{k: v[0] for k, v in METHODS.items()}, BASELINE_KEY: BASELINE_LABEL}

    pivot = df.pivot_table(index=["dataset", "measure"], columns="method_key", values="value")

    lines = []
    lines.append("% Auto-generated by src/evaluation/effectiveness.py -- do not edit by hand.")
    lines.append("% Requires \\usepackage{booktabs}, \\usepackage{multirow}, and \\usepackage{graphicx} in the preamble.")
    lines.append("\\begin{table}[!t]")
    lines.append("\\centering")
    lines.append("\\resizebox{\\textwidth}{!}{%")
    col_spec = "|".join(["l", "l"] + ["r"] * len(all_columns))
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")
    method_headers = [
        f"\\rotatebox{{{header_angle}}}{{{header_label(labels[k])}}}"
        if header_angle
        else header_label(labels[k])
        for k in all_columns
    ]
    header = ["Dataset", "Measure"] + method_headers
    lines.append(" & ".join(header) + " \\\\")
    lines.append("\\midrule")

    for dataset in datasets:
        if dataset not in pivot.index.get_level_values("dataset"):
            continue
        measures_present = [m for m in MEASURES if (dataset, m) in pivot.index]
        for i, measure in enumerate(measures_present):
            row_values = pivot.loc[(dataset, measure)]
            column_values = {k: row_values.get(k, np.nan) for k in all_columns}
            # Best and second-best are the two highest distinct *unrounded*
            # scores, each shared by however many methods achieve it: methods
            # that tie exactly (reranking the same candidates in the same
            # order, say) all get the same mark. Two cells can therefore print
            # the same rounded value and still be marked differently, which is
            # why the caption says the ties are decided before rounding.
            present = {k: v for k, v in column_values.items() if not np.isnan(v)}
            best_value = None
            second_value = None
            if bold_best and present:
                distinct = sorted(set(present.values()), reverse=True)
                best_value = distinct[0]
                if len(distinct) > 1:
                    second_value = distinct[1]

            cells = []
            for k in all_columns:
                v = column_values[k]
                if np.isnan(v):
                    cell = "--"
                else:
                    cell = f"{v:.{decimals}f}"
                    if v == best_value:
                        cell = f"\\textbf{{{cell}}}"
                    elif v == second_value:
                        cell = f"\\underline{{{cell}}}"
                cells.append(cell)

            dataset_label = DATASET_LABELS.get(dataset, dataset)
            dataset_cell = f"\\multirow{{{len(measures_present)}}}{{*}}{{{dataset_label}}}" if i == 0 else ""
            lines.append(" & ".join([dataset_cell, measure] + cells) + " \\\\")
        lines.append("\\midrule")

    if lines[-1] == "\\midrule":
        lines.pop()
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}%")
    lines.append("}")
    lines.append(
        "\\caption{Retrieval effectiveness (AP, RR, P@10, nDCG@10) of each retrieval "
        "method against the full-precision embedding baseline reported by MTEB for "
        "Qwen3-Embedding-8B. Best score per row in \\textbf{bold}, "
        "second-best \\underline{underlined}; tied methods share the mark, "
        "with ties determined by the unrounded scores.}"
    )
    lines.append("\\label{tab:retrieval-effectiveness}")
    lines.append("\\end{table}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    datasets = args.datasets or discover_datasets(args.runs_dir)
    if not datasets:
        print(f"ERROR: no datasets found under {args.runs_dir} (looked for *{METHODS['bm25'][1]}).", file=sys.stderr)
        sys.exit(1)
    print(f"Datasets: {', '.join(datasets)}", file=sys.stderr)

    df = collect_scores(args, datasets)
    if df.empty:
        print("ERROR: no scores were computed.", file=sys.stderr)
        sys.exit(1)

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "scores.csv", index=False)

    latex = to_latex(
        df,
        datasets,
        decimals=args.decimals,
        bold_best=not args.no_bold_best,
        header_angle=args.header_angle,
    )
    (out_dir / "table.tex").write_text(latex + "\n", encoding="utf-8")

    pd.set_option("display.float_format", lambda v: f"{v:0.4f}")
    method_order = [k for k in METHODS if k in df.method_key.unique()]
    all_columns = method_order + ([BASELINE_KEY] if BASELINE_KEY in df.method_key.unique() else [])
    labels = {**{k: v[0] for k, v in METHODS.items()}, BASELINE_KEY: BASELINE_LABEL}
    wide = df.pivot_table(index=["dataset", "measure"], columns="method_key", values="value")
    wide = wide.reindex(columns=all_columns).rename(columns=labels)
    wide = wide.reindex(
        pd.MultiIndex.from_tuples(
            [(d, m) for d in datasets for m in MEASURES if (d, m) in wide.index]
        )
    )
    print("\n=== Mean retrieval effectiveness per dataset / method ===")
    print(wide.to_string())
    print(f"\nWrote scores.csv and table.tex to {out_dir}/")


if __name__ == "__main__":
    main()
