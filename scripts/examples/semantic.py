#!/usr/bin/env python3
"""Produce a TREC run with semantic (vector) search.

The dense counterpart to ``examples/lexical.py``: open a prebuilt scrydb
index, rank the corpus for every stored query by vector distance, and
write the ranking as a TREC run file that trec_eval/ir_measures -- and
this repo's ``scripts/evaluation/effectiveness.py`` -- can score.

No model is loaded here either, and that is the point. ``batch_search``
prefers a query's *stored* embedding over re-encoding its text, so a run
is reproduced from the exact vectors it was built with -- no download, no
GPU, no drift from a model version that moved underneath the experiment.
Attaching a model (``index.add_model(...)``, see ``examples/repl.py``) is
only needed to search text that was never indexed.

What does have to be there is the sqlite-vec extension, which is where all
three precisions live:

    --precision binary   1 bit/dim,  Hamming distance   (the default)
    --precision int8     1 byte/dim, cosine
    --precision float    full,       cosine

Same index, same queries, three answers at three costs -- the comparison
this repo exists to make. Documents are stored at every precision that was
requested at index time, so switching between them is a flag, not a rebuild.

--rerank adds a second stage: the top --top-k hits from the cheap first
stage are re-scored at a finer precision, the "retrieve coarse, rerank
precise" pattern (e.g. --precision binary --rerank float). It has to differ
from --precision, since re-scoring at the precision that already ranked
would be a no-op.

Run with:
    python examples/semantic.py nfcorpus
    python examples/semantic.py nfcorpus --limit 5 --preview 5        # smoke test
    python examples/semantic.py nfcorpus --precision int8
    python examples/semantic.py nfcorpus --precision binary --rerank float

Results are cut at 1000 per query, the TREC convention and the depth the
published runs use, and the output filename follows the convention
``effectiveness.py`` globs for, so a finished run is picked up without
configuration:

    <dataset>-hamming.txt                    --precision binary
    <dataset>-cosine_int8.txt                --precision int8
    <dataset>-cosine_float.txt               --precision float
    <dataset>-hamming-cosine_float.txt       --precision binary --rerank float

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
# scripts/evaluation/effectiveness.py globs for (its METHODS table). These
# are the v0.1.x spellings -- "hamming" for binary, "cosine_*" for the
# cosine precisions -- because that is how the published runs are named.
PRECISION_NAME = {"binary": "hamming", "int8": "cosine_int8", "float": "cosine_float"}

# The result field each precision reports its score under (scrydb's
# _SCORE_FIELD), and whether lower is better. Binary search reports raw
# Hamming distance; write_trec negates it, because the TREC SCORE column is
# always higher-is-better.
SCORE_FIELD = {
    "binary": ("hamming_distance", True),
    "int8": ("int8_similarity", False),
    "float": ("cosine_similarity", False),
}

# Hits shown per previewed query.
PREVIEW_HITS = 5


def log(message: str) -> None:
    print(f"scrydb-semantic: {message}", file=sys.stderr)


def embeddings(index: Index, collection: str, precision: str):
    """The ``{id: vector}`` view for one collection/precision. Empty (not
    an error) when nothing was indexed at that precision -- which is the
    case worth checking before searching, see `check_available`."""
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
    """Fail loudly when *precision* cannot actually be searched.

    A missing vec0 table is not an error inside scrydb -- ``_vec_search``
    returns no rows for it -- so an index built with, say,
    --no-full-embeddings answers a float search with a perfectly
    well-formed, perfectly empty run. Better to stop here than to score
    that silence as a result.
    """
    n_documents = len(embeddings(index, "documents", precision))
    if not n_documents:
        raise SystemExit(
            f"scrydb-semantic: no {precision!r} document embeddings in this index, so the "
            f"{role} stage would match nothing -- re-index with that precision, or pick "
            "one this index has"
        )

    # The query side has a fallback the document side does not: a vector at
    # the exact precision is used verbatim, and failing that a stored float
    # vector is quantized down to it on the fly (_query_expr).
    n_exact = len(embeddings(index, "queries", precision))
    n_float = len(embeddings(index, "queries", "float"))
    if n_exact:
        source = f"{n_exact} stored {precision} query vectors"
    elif n_float:
        source = f"{n_float} stored float query vectors, quantized to {precision} per query"
    else:
        raise SystemExit(
            f"scrydb-semantic: no {precision!r} or float query embeddings in this index -- "
            "index the queries with embeddings (examples/index.py --queries), or attach a "
            "model to encode them"
        )
    log(f"{role} stage: {precision} over {n_documents} documents, using {source}")


def preview(index: Index, run, n_queries: int, n_hits: int = PREVIEW_HITS) -> None:
    """Print the top hits of the first *n_queries* queries, so the run is
    inspectable instead of an opaque file of ids."""
    # Same priority order as scrydb's Run._SCORE_FIELDS: a reranked result
    # keeps the first stage's field alongside the rerank's, and the rerank
    # field is the one that decided the final ranking.
    fields = ("cosine_similarity", "int8_similarity", "hamming_distance")
    for query_id in list(run)[:n_queries]:
        query_text = (index.queries.get(query_id) or {}).get("text", "")
        print(f"\n[{query_id}] {query_text}")
        hits = run[query_id][:n_hits]
        if not hits:
            print("  (no matches)")
            continue
        for rank, hit in enumerate(hits, start=1):
            field = next((f for f in fields if f in hit), None)
            # Hamming distances count downward-ranked bits, so they climb
            # as the list gets worse; the cosine fields fall.
            score = f"{field}={hit[field]:.4f}" if field else ""
            text = " ".join(hit["document"].get("text", "").split())
            print(f"  {rank:>2}. [{hit['id']:<12}] {score:<28} {text[:90]}")


def main(argv: "list[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(
        description="Produce a TREC run with semantic (vector) search over a scrydb index."
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
        help="Vector representation the ranking runs on: binary (1 bit/dim, Hamming), "
             "int8 (1 byte/dim, cosine), or float (full precision, cosine). "
             "Default: %(default)s",
    )
    parser.add_argument(
        "--top-k", type=int, default=1000, help="Results per query (default: %(default)s)"
    )
    parser.add_argument(
        "--rerank",
        choices=("binary", "int8", "float"),
        help="Re-score the first stage's hits at this precision -- must differ from "
             "--precision, e.g. --precision binary --rerank float",
    )
    parser.add_argument(
        "--rerank-depth",
        type=int,
        default=1000,
        help="How many first-stage hits the rerank stage sees (default: %(default)s)",
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
        # scrydb raises this too; catching it here costs nothing and says
        # which flag to change.
        parser.error(
            f"--rerank {args.rerank} is a no-op on a {args.precision}-ranked search "
            "-- rerank at a finer precision, e.g. --precision binary --rerank float"
        )
    # Unlike lexical mode, the first stage retrieves --top-k candidates and
    # --rerank-depth only trims that shortlist, so a depth above top_k gives
    # the rerank stage nothing extra to work with.
    if args.rerank and args.rerank_depth > args.top_k:
        log(f"--rerank-depth {args.rerank_depth} exceeds --top-k {args.top_k}; "
            f"the rerank stage will see {args.top_k} candidates")

    index_path = args.index or DEFAULT_INDEX_DIR / f"{args.dataset}.db"
    if not index_path.is_file():
        parser.error(
            f"no index at {index_path} -- fetch one with: "
            f"python examples/dl.py dataset {args.dataset} --assets index"
        )

    stages = PRECISION_NAME[args.precision]
    if args.rerank:
        stages += f"-{PRECISION_NAME[args.rerank]}"
    suffix = f"-{stages}.txt"
    if args.limit:
        # A truncated run must not take the canonical name: effectiveness.py
        # would score it as if it covered the whole query set.
        suffix = suffix.replace(".txt", f"-limit{args.limit}.txt")
    run_path = args.output or DEFAULT_RUNS_DIR / f"{args.dataset}{suffix}"
    run_path.parent.mkdir(parents=True, exist_ok=True)

    # vec_ext_path="auto" (the default) loads sqlite-vec from the installed
    # pip package; every precision below is its virtual table.
    with Index.open(index_path) as index:
        n_documents, n_queries = len(index.documents), len(index.queries)
        log(f"{index_path.name}: {n_documents} documents, {n_queries} queries")
        if not n_queries:
            # batch_search(queries=None) would silently return an empty Run.
            raise SystemExit(
                f"scrydb-semantic: {index_path} has no stored queries -- index them with "
                "examples/index.py --queries, or use index.search(...) for one-off text"
            )
        check_available(index, args.precision, role="ranking")
        if args.rerank:
            check_available(index, args.rerank, role="rerank")

        # queries=None is the whole point of batch_search: every stored
        # query, in one pass, keyed by query id in the returned Run.
        queries = list(index.queries)[: args.limit] if args.limit else None
        run = index.batch_search(
            queries=queries,
            mode="semantic",
            precision=args.precision,
            top_k=args.top_k,
            rerank=args.rerank or False,
            rerank_depth=args.rerank_depth,
        )

        # QID Q0 DOCID RANK SCORE TAG -- what trec_eval and friends read.
        # The SCORE column carries the *last* stage's field, negated for
        # binary so higher stays better.
        run.write_trec(run_path, tag=args.tag)

        if args.preview:
            preview(index, run, args.preview)

    field, lower_is_better = SCORE_FIELD[args.rerank or args.precision]
    retrieved = sum(len(hits) for hits in run.values())
    empty = sum(1 for hits in run.values() if not hits)
    log(f"{len(run)} queries, {retrieved} results ({empty} queries with no match)")
    log(f"ranked by {field}{' (negated in the run file)' if lower_is_better else ''}")
    log(f"wrote {run_path}")
    if not args.limit:
        log(f"score it with: python scripts/evaluation/effectiveness.py --datasets {args.dataset}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
