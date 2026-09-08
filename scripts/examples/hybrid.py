#!/usr/bin/env python3
"""Produce a TREC run with hybrid (RRF) search.

The third of the trio next to ``examples/lexical.py`` and
``examples/semantic.py``: open a prebuilt scrydb index, rank the corpus for
every stored query by fusing a BM25 ranking with a vector ranking, and write
the result as a TREC run file that trec_eval/ir_measures -- and this repo's
``scripts/evaluation/effectiveness.py`` -- can score.

Hybrid runs both of the other two and merges their rankings with Reciprocal
Rank Fusion: each document scores ``1/(k + rank)`` in every list it appears
in, summed. Only ranks go into that sum, never the raw scores, which is the
point -- BM25 scores and cosine similarities are not on a common scale, and
RRF never has to put them on one. A document ranked well by both stages
beats one ranked brilliantly by a single stage, so fusion rewards agreement.

Because both stages run, this example needs everything the other two need:
stored query ``text`` for the BM25 side, and document/query vectors for the
semantic side. --precision picks the vector representation the semantic side
ranks with, exactly as in ``examples/semantic.py``, and --rerank re-scores
that side at a finer precision *before* fusion.

The knob that matters here is --candidate-limit: it sets how deep *each*
stage retrieves before fusion, so the fused pool is at most twice it. It is
not --top-k, and leaving it at scrydb's own default of 50 while asking for
1000 results is the classic way to get a run that stops at 100. This script
defaults it to 1000, matching the published RRF runs.

One caveat specific to fusion: documents found by a single stage at the
same rank get identical RRF scores, and ``_reciprocal_rank_fusion`` orders
its output from a ``set`` union, whose iteration order follows Python's
per-process string hashing. Tied documents therefore come out in a
different order -- and, right at the --top-k cutoff, a different selection
-- from one process to the next. Ranking measures are unaffected (eval
tools re-sort by the SCORE column), but the run file is only byte-identical
across processes with PYTHONHASHSEED fixed. The lexical and semantic
examples have no such wrinkle: their order comes from SQLite's ORDER BY.

Run with:
    python examples/hybrid.py nfcorpus
    python examples/hybrid.py nfcorpus --limit 5 --preview 5          # smoke test
    python examples/hybrid.py nfcorpus --precision float
    python examples/hybrid.py nfcorpus --precision binary --rerank float

Results are cut at 1000 per query, the TREC convention and the depth the
published runs use, and the output filename follows the convention
``effectiveness.py`` globs for, so a finished run is picked up without
configuration:

    <dataset>-rrf-hamming.txt          --precision binary
    <dataset>-rrf-cosine_int8.txt      --precision int8
    <dataset>-rrf-cosine_float.txt     --precision float

Fetch an index first with:
    python examples/dl.py dataset nfcorpus --assets index qrels
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from scrydb import Index

# Paths are anchored on this file, not the cwd, so the defaults resolve the
# same way `dl.py` writes them however the script is invoked.
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INDEX_DIR = REPO_ROOT / "data" / "indices" / "beir"
DEFAULT_RUNS_DIR = REPO_ROOT / "data" / "runs" / "beir"

# How each precision is spelled in a run filename, matching the names
# scripts/evaluation/effectiveness.py globs for (its METHODS table): the
# fused runs are "-rrf-" plus the semantic side's precision.
PRECISION_NAME = {"binary": "hamming", "int8": "cosine_int8", "float": "cosine_float"}

# Hits shown per previewed query.
PREVIEW_HITS = 5


def log(message: str) -> None:
    print(f"scrydb-hybrid: {message}", file=sys.stderr)


def embeddings(index: Index, collection: str, precision: str):
    """The ``{id: vector}`` view for one collection/precision. Empty (not an
    error) when nothing was indexed at that precision."""
    tables = {
        ("documents", "binary"): index.document_embeddings_binary,
        ("documents", "int8"): index.document_embeddings_int8,
        ("documents", "float"): index.document_embeddings,
        ("queries", "binary"): index.query_embeddings_binary,
        ("queries", "int8"): index.query_embeddings_int8,
        ("queries", "float"): index.query_embeddings,
    }
    return tables[(collection, precision)]


def check_available(index: Index, precision: str, role: str) -> None:
    """Fail loudly when the semantic side cannot actually run at *precision*.

    A missing vec0 table is not an error inside scrydb -- ``_vec_search``
    returns no rows for it -- and in hybrid mode that failure is quieter
    still: fusion of a full BM25 list with an empty vector list is just the
    BM25 list, wearing rrf_score. A "hybrid" run that is silently lexical is
    the worst thing this script could hand to an evaluation.
    """
    n_documents = len(embeddings(index, "documents", precision))
    if not n_documents:
        raise SystemExit(
            f"scrydb-hybrid: no {precision!r} document embeddings in this index, so the "
            f"{role} stage would contribute nothing and the fusion would collapse to "
            "plain BM25 -- re-index with that precision, or pick one this index has"
        )

    # The query side has a fallback the document side does not: a vector at
    # the exact precision is used verbatim, and failing that a stored float
    # vector is quantized down to it on the fly.
    n_exact = len(embeddings(index, "queries", precision))
    n_float = len(embeddings(index, "queries", "float"))
    if n_exact:
        source = f"{n_exact} stored {precision} query vectors"
    elif n_float:
        source = f"{n_float} stored float query vectors, quantized to {precision} per query"
    else:
        raise SystemExit(
            f"scrydb-hybrid: no {precision!r} or float query embeddings in this index -- "
            "index the queries with embeddings (examples/index.py --queries), or attach a "
            "model to encode them"
        )
    log(f"{role} stage: {precision} over {n_documents} documents, using {source}")


def preview(index: Index, run, n_queries: int, n_hits: int = PREVIEW_HITS) -> None:
    """Print the top hits of the first *n_queries* queries, each with the
    rank it held in the two fused lists -- which is the whole story of why a
    document ended up where it did."""
    for query_id in list(run)[:n_queries]:
        query_text = (index.queries.get(query_id) or {}).get("text", "")
        print(f"\n[{query_id}] {query_text}")
        hits = run[query_id][:n_hits]
        if not hits:
            print("  (no matches)")
            continue
        for rank, hit in enumerate(hits, start=1):
            # A document retrieved by only one stage still fuses in, with
            # the other rank left as None -- that asymmetry is what the
            # contribution counts below summarize.
            bm25 = hit.get("lexical_rank")
            vec = hit.get("semantic_rank")
            stages = f"bm25 #{bm25 or '--':<4} vec #{vec or '--':<4}"
            text = " ".join(hit["document"].get("text", "").split())
            print(f"  {rank:>2}. [{hit['id']:<12}] rrf={hit['rrf_score']:.5f}  {stages} {text[:70]}")


def contributions(run) -> "tuple[int, int, int]":
    """How many retrieved documents each stage accounts for: found by both,
    by BM25 alone, by the vector side alone."""
    both = lexical_only = semantic_only = 0
    for hits in run.values():
        for hit in hits:
            in_lexical = hit.get("lexical_rank") is not None
            in_semantic = hit.get("semantic_rank") is not None
            if in_lexical and in_semantic:
                both += 1
            elif in_lexical:
                lexical_only += 1
            elif in_semantic:
                semantic_only += 1
    return both, lexical_only, semantic_only


def main(argv: "list[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(
        description="Produce a TREC run with hybrid (RRF) search over a scrydb index."
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
        "--precision",
        choices=("binary", "int8", "float"),
        default="binary",
        help="Vector representation the semantic side of the fusion ranks with: "
             "binary (1 bit/dim, Hamming), int8 (1 byte/dim, cosine), or float "
             "(full precision, cosine). Default: %(default)s",
    )
    parser.add_argument(
        "--top-k", type=int, default=1000, help="Results per query, after fusion (default: %(default)s)"
    )
    parser.add_argument(
        "--candidate-limit",
        type=int,
        default=1000,
        help="How deep each stage retrieves before fusion, so the fused pool is at "
             "most twice this (default: %(default)s; scrydb's own default is 50)",
    )
    parser.add_argument(
        "--rrf-k",
        type=int,
        default=60,
        help="RRF constant k in 1/(k + rank): larger flattens the weight of top "
             "ranks, so agreement between the stages counts for relatively more "
             "(default: %(default)s)",
    )
    parser.add_argument(
        "--rerank",
        choices=("binary", "int8", "float"),
        help="Re-score the semantic side at this precision before fusion -- must "
             "differ from --precision, e.g. --precision binary --rerank float",
    )
    parser.add_argument(
        "--rerank-depth",
        type=int,
        default=1000,
        help="How many of the semantic side's hits the rerank stage sees (default: %(default)s)",
    )
    # These reach FTS5's bm25() as its trailing arguments, which are per-column
    # weights -- not BM25's b and k1. FTS5 fixes those at compile time
    # (k1=1.2, b=0.75) and offers no way to set them, so despite the names
    # neither flag retunes BM25. documents_fts is fts5(id UNINDEXED, text):
    # the first weight lands on the UNINDEXED column and provably changes
    # nothing, the second reweights the text column and does move scores. The
    # defaults are scrydb's own, so a run made here matches the published ones.
    parser.add_argument(
        "--bm25-b",
        type=float,
        default=0.6,
        help="FTS5 bm25() weight for the UNINDEXED id column -- inert, kept only to "
             "mirror batch_search's signature (default: %(default)s)",
    )
    parser.add_argument(
        "--bm25-k1",
        type=float,
        default=0.9,
        help="FTS5 bm25() weight for the text column -- rescales and can reorder "
             "results, but it is not BM25's k1 (default: %(default)s)",
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

    if args.rerank == args.precision:
        parser.error(
            f"--rerank {args.rerank} is a no-op on a {args.precision}-ranked semantic side "
            "-- rerank at a finer precision, e.g. --precision binary --rerank float"
        )
    if args.candidate_limit < args.top_k:
        # Fusion can only rank what the two stages retrieved: at most
        # 2 * candidate_limit documents, and far fewer where they agree.
        log(f"--candidate-limit {args.candidate_limit} is below --top-k {args.top_k}; "
            f"at most {2 * args.candidate_limit} documents can be fused per query")

    index_path = args.index or DEFAULT_INDEX_DIR / f"{args.dataset}.db"
    if not index_path.is_file():
        parser.error(
            f"no index at {index_path} -- fetch one with: "
            f"python examples/dl.py dataset {args.dataset} --assets index"
        )

    stages = f"rrf-{PRECISION_NAME[args.precision]}"
    if args.rerank:
        stages += f"-{PRECISION_NAME[args.rerank]}"
    suffix = f"-{stages}.txt"
    if args.limit:
        # A truncated run must not take the canonical name: effectiveness.py
        # would score it as if it covered the whole query set.
        suffix = suffix.replace(".txt", f"-limit{args.limit}.txt")
    run_path = args.output or DEFAULT_RUNS_DIR / f"{args.dataset}{suffix}"
    run_path.parent.mkdir(parents=True, exist_ok=True)

    with Index.open(index_path) as index:
        n_documents, n_queries = len(index.documents), len(index.queries)
        log(f"{index_path.name}: {n_documents} documents, {n_queries} queries")
        if not n_queries:
            # batch_search(queries=None) would silently return an empty Run.
            raise SystemExit(
                f"scrydb-hybrid: {index_path} has no stored queries -- index them with "
                "examples/index.py --queries, or use index.search(...) for one-off text"
            )

        # The lexical stage reads each query's stored text, and raises per
        # query if it is missing; checking the first one fails fast instead,
        # since queries are almost always indexed the same way.
        first_id = next(iter(index.queries))
        if not (index.queries.get(first_id) or {}).get("text"):
            raise SystemExit(
                f"scrydb-hybrid: query {first_id!r} has no stored 'text', which the BM25 "
                "side of the fusion needs -- re-index the queries with a text field, or "
                "use examples/semantic.py for a vectors-only run"
            )
        check_available(index, args.precision, role="semantic")
        if args.rerank:
            check_available(index, args.rerank, role="rerank")

        # queries=None is the whole point of batch_search: every stored
        # query, in one pass, keyed by query id in the returned Run.
        # Note there is no raw= here: hybrid always sanitizes the query text
        # for its BM25 stage, unlike mode="lexical", which can opt out.
        queries = list(index.queries)[: args.limit] if args.limit else None
        run = index.batch_search(
            queries=queries,
            mode="hybrid",
            precision=args.precision,
            top_k=args.top_k,
            candidate_limit=args.candidate_limit,
            rrf_k=args.rrf_k,
            rerank=args.rerank or False,
            rerank_depth=args.rerank_depth,
            bm25_b=args.bm25_b,
            bm25_k1=args.bm25_k1,
        )

        # QID Q0 DOCID RANK SCORE TAG -- what trec_eval and friends read.
        # rrf_score outranks every constituent stage's field in
        # Run._SCORE_FIELDS, so the SCORE column is the fused score. Rows
        # sharing a score may be ordered differently by another process
        # (see the tie-breaking note in this module's docstring).
        run.write_trec(run_path, tag=args.tag)

        if args.preview:
            preview(index, run, args.preview)

    retrieved = sum(len(hits) for hits in run.values())
    empty = sum(1 for hits in run.values() if not hits)
    both, lexical_only, semantic_only = contributions(run)
    log(f"{len(run)} queries, {retrieved} results ({empty} queries with no match)")
    log(f"retrieved by both stages: {both}, by BM25 only: {lexical_only}, "
        f"by {args.rerank or args.precision} only: {semantic_only}")
    log(f"wrote {run_path}")
    if not args.limit:
        log(f"score it with: python scripts/evaluation/effectiveness.py --datasets {args.dataset}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
