#!/usr/bin/env python3
"""Produce a TREC run with lexical (BM25) search.

The smallest complete experiment in this repo: open a prebuilt scrydb
index, rank the corpus for every query stored in it with BM25 over FTS5,
and write the ranking as a TREC run file that trec_eval/ir_measures -- and
this repo's own ``scripts/evaluation/effectiveness.py`` -- can score.

Lexical search is the one mode that needs nothing but SQLite. No model is
loaded, no query is encoded, and the sqlite-vec extension is never touched,
so the index is opened with ``vec_ext_path=None`` unless --rerank asks for
a vector second stage. An index built without a single embedding produces
exactly the same BM25 run as the multi-gigabyte one carrying all three
precisions.

The queries come from the index itself: ``batch_search(queries=None)``
replays every stored query, reading each one's ``text`` field. That is what
makes a run reproducible -- the same database replays the same experiment,
with no separate query file to keep in sync. Each text is sanitized and
OR-joined before it reaches FTS5's ``MATCH`` (``raw=False``), so BEIR's
natural-language queries, question marks and all, do not trip the FTS5
parser; pass --raw only for hand-written FTS5 expressions.

Run with:
    python examples/lexical.py nfcorpus
    python examples/lexical.py nfcorpus --limit 5 --preview 5      # smoke test
    python examples/lexical.py nfcorpus --rerank float --rerank-depth 1000

Results are cut at 1000 per query, the TREC convention and the depth the
published runs use, and the output filename follows the convention
``effectiveness.py`` globs for, so a finished run is picked up without
configuration:

    <dataset>-bm25.txt                  (no --rerank)
    <dataset>-bm25-hamming.txt          --rerank binary
    <dataset>-bm25-cosine_int8.txt      --rerank int8
    <dataset>-bm25-cosine_float.txt     --rerank float

Fetch an index first with:
    python examples/dl.py dataset nfcorpus --assets index qrels
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

from scrydb import Index

# Paths are anchored on this file, not the cwd, so the defaults resolve the
# same way `dl.py` writes them however the script is invoked.
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INDEX_DIR = REPO_ROOT / "data" / "indices" / "beir"
DEFAULT_RUNS_DIR = REPO_ROOT / "data" / "runs" / "beir"

# Run filename suffix per second stage, matching the names
# scripts/evaluation/effectiveness.py globs for (its METHODS table). The
# spelling is the v0.1.x one -- "hamming" for binary, "cosine_*" for the
# cosine precisions -- because that is how the published runs are named.
RUN_SUFFIX = {
    None: "-bm25.txt",
    "binary": "-bm25-hamming.txt",
    "int8": "-bm25-cosine_int8.txt",
    "float": "-bm25-cosine_float.txt",
}

# Score fields in the same priority order as scrydb's Run._SCORE_FIELDS: a
# reranked result keeps its first-stage BM25 "score" alongside the rerank
# field, and the rerank field is the one that reflects the final ranking
# (and the one written to the TREC SCORE column).
SCORE_FIELDS = ("cosine_similarity", "int8_similarity", "hamming_distance", "score")

# Hits shown per previewed query.
PREVIEW_HITS = 5


def log(message: str) -> None:
    print(f"scrydb-lexical: {message}", file=sys.stderr)


def preview(index: Index, run, n_queries: int, n_hits: int = PREVIEW_HITS) -> None:
    """Print the top hits of the first *n_queries* queries, so the run is
    inspectable instead of an opaque file of ids."""
    for query_id in list(run)[:n_queries]:
        query_text = (index.queries.get(query_id) or {}).get("text", "")
        print(f"\n[{query_id}] {query_text}")
        hits = run[query_id][:n_hits]
        if not hits:
            # BM25 returns nothing when no query term is in the corpus
            # vocabulary -- a legitimate empty result, not an error.
            print("  (no matches)")
            continue
        for rank, hit in enumerate(hits, start=1):
            field = next((f for f in SCORE_FIELDS if f in hit), None)
            score = f"{field}={hit[field]:.4f}" if field else ""
            text = " ".join(hit["document"].get("text", "").split())
            print(f"  {rank:>2}. [{hit['id']:<12}] {score:<28} {text[:90]}")


def main(argv: "list[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(
        description="Produce a TREC run with lexical (BM25) search over a scrydb index."
    )
    parser.add_argument(
        "dataset",
        help="Dataset name, e.g. nfcorpus -- names the index under "
             f"{DEFAULT_INDEX_DIR} and the run written to {DEFAULT_RUNS_DIR}",
    )
    parser.add_argument("--index", type=Path, help="Index path (default: <dataset>.db under the index dir)")
    parser.add_argument("--output", type=Path, help="Run path (default: the effectiveness.py name for this method)")
    parser.add_argument("--tag", default="run", help="TREC run tag, the 6th column (default: %(default)s)")
    parser.add_argument(
        "--top-k", type=int, default=1000, help="Results per query (default: %(default)s)"
    )
    parser.add_argument(
        "--rerank",
        choices=("binary", "int8", "float"),
        help="Rerank the BM25 candidates by vector search at this precision "
             "-- turns BM25 into a first stage and needs embeddings in the index",
    )
    parser.add_argument(
        "--rerank-depth",
        type=int,
        default=1000,
        help="How many BM25 candidates the rerank stage sees (default: %(default)s)",
    )
    # FTS5's own bm25() defaults are b=0.75, k1=1.2; scrydb defaults to
    # 0.6/0.9, the values Anserini uses for its BEIR baselines, so runs made
    # here are comparable with the published ones.
    parser.add_argument("--bm25-b", type=float, default=0.6, help="BM25 b (default: %(default)s)")
    parser.add_argument("--bm25-k1", type=float, default=0.9, help="BM25 k1 (default: %(default)s)")
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Pass query text to FTS5 MATCH unmodified (advanced FTS5 syntax) "
             "instead of sanitizing and OR-joining its terms",
    )
    parser.add_argument("--limit", type=int, help="Search only the first N stored queries (smoke test)")
    parser.add_argument(
        "--preview",
        type=int,
        default=1,
        metavar="N",
        help=f"Print the top {PREVIEW_HITS} hits of N queries (default: %(default)s)",
    )
    args = parser.parse_args(argv)

    index_path = args.index or DEFAULT_INDEX_DIR / f"{args.dataset}.db"
    if not index_path.is_file():
        parser.error(
            f"no index at {index_path} -- fetch one with: "
            f"python examples/dl.py dataset {args.dataset} --assets index"
        )
    suffix = RUN_SUFFIX[args.rerank]
    if args.limit:
        # A truncated run must not take the canonical name: effectiveness.py
        # would score it as if it covered the whole query set. The marker also
        # keeps it out of that script's "*-bm25.txt" dataset discovery.
        suffix = suffix.replace(".txt", f"-limit{args.limit}.txt")
    run_path = args.output or DEFAULT_RUNS_DIR / f"{args.dataset}{suffix}"
    run_path.parent.mkdir(parents=True, exist_ok=True)

    # Plain BM25 touches only FTS5, so the sqlite-vec extension is not
    # loaded at all; a rerank stage is vector search and does need it.
    with Index.open(index_path, vec_ext_path="auto" if args.rerank else None) as index:
        n_documents, n_queries = len(index.documents), len(index.queries)
        log(f"{index_path.name}: {n_documents} documents, {n_queries} queries")
        if not n_documents:
            raise SystemExit(f"scrydb-lexical: {index_path} has no documents")
        if not n_queries:
            # batch_search(queries=None) would silently return an empty Run.
            raise SystemExit(
                f"scrydb-lexical: {index_path} has no stored queries -- index them with "
                "examples/index.py --queries, or use index.search(...) for one-off text"
            )

        # queries=None is the whole point of batch_search: every stored
        # query, in one pass, keyed by query id in the returned Run.
        queries = list(index.queries)[: args.limit] if args.limit else None
        try:
            run = index.batch_search(
                queries=queries,
                mode="lexical",
                top_k=args.top_k,
                rerank=args.rerank or False,
                rerank_depth=args.rerank_depth,
                raw=args.raw,
                bm25_b=args.bm25_b,
                bm25_k1=args.bm25_k1,
            )
        except sqlite3.OperationalError as exc:
            # Only reachable with --raw: without it the query text is
            # sanitized into an OR of quoted terms, which FTS5 always
            # parses. A stored query ending in "?" is enough to hit this.
            if not args.raw:
                raise
            raise SystemExit(f"scrydb-lexical: --raw: {exc} -- these queries are not FTS5 syntax; drop --raw")

        # QID Q0 DOCID RANK SCORE TAG -- what trec_eval and friends read.
        run.write_trec(run_path, tag=args.tag)

        if args.preview:
            preview(index, run, args.preview)

    retrieved = sum(len(hits) for hits in run.values())
    empty = sum(1 for hits in run.values() if not hits)
    log(f"{len(run)} queries, {retrieved} results ({empty} queries with no match)")
    log(f"wrote {run_path}")
    if not args.limit:
        log(f"score it with: python scripts/evaluation/effectiveness.py --datasets {args.dataset}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
