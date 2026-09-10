#!/usr/bin/env python3
"""Index a corpus, search it every way scrydb can, write TREC runs.

The end-to-end example: where ``examples/index.py`` builds an index and
``examples/{lexical,semantic,hybrid}.py`` each produce one run from an
existing one, this script is the whole experiment in a single process --
JSONL in, a directory of scored-and-ready TREC runs out.

    corpus-embeddings.jsonl  --.
                               |--> <dataset>.db --> batch_search x N --> <dataset>-<method>.txt
    queries-embeddings.jsonl --'                                          ...

The reason to run it this way rather than calling the other three in a
loop is that all the methods share one open index. Every method below is
the *same* corpus, the *same* stored query vectors and the *same* SQLite
file answering a different question -- which is exactly the comparison
this repo exists to make, and it stays honest only when nothing is rebuilt
in between.

Methods
-------
--methods selects from the 13 configurations in this repo's published
table, keyed and named exactly as ``scripts/evaluation/effectiveness.py``
expects them, so a finished run is picked up for scoring without any
configuration:

  bm25                       -bm25.txt                      BM25 over FTS5
  bm25_hamming               -bm25-hamming.txt              BM25 >> binary
  bm25_cosine_int8           -bm25-cosine_int8.txt          BM25 >> int8
  bm25_cosine_float          -bm25-cosine_float.txt         BM25 >> float
  hamming                    -hamming.txt                   binary
  hamming_cosine_int8        -hamming-cosine_int8.txt       binary >> int8
  hamming_cosine_float       -hamming-cosine_float.txt      binary >> float
  cosine_int8                -cosine_int8.txt               int8
  cosine_int8_cosine_float   -cosine_int8-cosine_float.txt  int8 >> float
  cosine_float               -cosine_float.txt              float
  hybrid_hamming             -rrf-hamming.txt               RRF(BM25, binary)
  hybrid_cosine_int8         -rrf-cosine_int8.txt           RRF(BM25, int8)
  hybrid_cosine_float        -rrf-cosine_float.txt          RRF(BM25, float)

``all`` (the default), ``lexical``, ``semantic`` and ``hybrid`` are group
shorthands for the rows above, and the two can be mixed:
``--methods lexical cosine_float``.

The method selection also drives *indexing*: int8 vectors are stored only
when a selected method actually ranks or reranks at that precision, since
scrydb does not store them by default. That is the one place where asking
for a different method changes the index and not just the search.

Resumability
------------
Both halves skip work that is already done, so an interrupted 13-method
sweep is restarted by running the same command again:

  * an index that already holds documents and queries is reused as-is
    (--reindex forces the upsert, e.g. after the JSONL changed);
  * a run file that already exists is left alone (--overwrite replaces it).

Run with:
    python examples/pipeline.py nfcorpus --limit 20        # smoke test, minutes
    python examples/pipeline.py nfcorpus
    python examples/pipeline.py nfcorpus --methods bm25 cosine_float
    python examples/pipeline.py nfcorpus --methods all --overwrite

--limit caps both the rows indexed and the queries searched, and moves the
whole experiment onto its own ``<dataset>-limit<N>`` index and run names,
so a smoke test can never be mistaken for -- or overwrite -- a real one.

Inputs are ``{"docid": ..., "text": ..., "embedding": [...]}`` JSONL, as
produced by ``scripts/datasets/beir/<dataset>/embed_<dataset>_docs.py``;
--*-field renames any of those keys. To skip embedding a corpus yourself,
fetch a prebuilt index and score against the published runs instead:

    python examples/dl.py dataset nfcorpus --assets index qrels
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import OrderedDict
from pathlib import Path

from scrydb import Index

# Paths are anchored on this file, not the cwd, so the defaults resolve the
# same way `dl.py` writes them however the script is invoked.
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_DIR = REPO_ROOT / "data" / "datasets" / "beir"
DEFAULT_INDEX_DIR = REPO_ROOT / "data" / "indices" / "beir"
DEFAULT_RUNS_DIR = REPO_ROOT / "data" / "runs" / "beir"

# The published method matrix: run-file suffix + the batch_search keywords
# that produce it. The suffixes are the ones effectiveness.py's METHODS
# table globs for, and the spellings there are the v0.1.x ones ("hamming"
# for binary, "cosine_*" for the cosine precisions), which is why the file
# names and the scrydb keywords do not read alike.
#
# Only the knobs that define a method live here; --top-k, --rerank-depth
# and --candidate-limit are passed to every method alike, so a sweep varies
# one thing at a time.
METHODS = OrderedDict(
    [
        ("bm25", ("-bm25.txt", {"mode": "lexical"})),
        ("bm25_hamming", ("-bm25-hamming.txt", {"mode": "lexical", "rerank": "binary"})),
        ("bm25_cosine_int8", ("-bm25-cosine_int8.txt", {"mode": "lexical", "rerank": "int8"})),
        ("bm25_cosine_float", ("-bm25-cosine_float.txt", {"mode": "lexical", "rerank": "float"})),
        ("hamming", ("-hamming.txt", {"mode": "semantic", "precision": "binary"})),
        (
            "hamming_cosine_int8",
            ("-hamming-cosine_int8.txt", {"mode": "semantic", "precision": "binary", "rerank": "int8"}),
        ),
        (
            "hamming_cosine_float",
            ("-hamming-cosine_float.txt", {"mode": "semantic", "precision": "binary", "rerank": "float"}),
        ),
        ("cosine_int8", ("-cosine_int8.txt", {"mode": "semantic", "precision": "int8"})),
        (
            "cosine_int8_cosine_float",
            ("-cosine_int8-cosine_float.txt", {"mode": "semantic", "precision": "int8", "rerank": "float"}),
        ),
        ("cosine_float", ("-cosine_float.txt", {"mode": "semantic", "precision": "float"})),
        ("hybrid_hamming", ("-rrf-hamming.txt", {"mode": "hybrid", "precision": "binary"})),
        ("hybrid_cosine_int8", ("-rrf-cosine_int8.txt", {"mode": "hybrid", "precision": "int8"})),
        ("hybrid_cosine_float", ("-rrf-cosine_float.txt", {"mode": "hybrid", "precision": "float"})),
    ]
)

# Group shorthands, derived from the table rather than restated, so a new
# method row joins its group by itself.
GROUPS = {
    "all": list(METHODS),
    "lexical": [k for k, (_, kw) in METHODS.items() if kw["mode"] == "lexical"],
    "semantic": [k for k, (_, kw) in METHODS.items() if kw["mode"] == "semantic"],
    "hybrid": [k for k, (_, kw) in METHODS.items() if kw["mode"] == "hybrid"],
}


def log(message: str) -> None:
    print(f"scrydb-pipeline: {message}", file=sys.stderr)


def precisions_used(kwargs: dict) -> "set[str]":
    """Which vector precisions a method needs in the index.

    Read off the search keywords instead of tabulated a second time: the
    ranking precision counts only for the vector modes (BM25 ranks without
    a vector at all), and a rerank stage counts wherever it appears.
    """
    needed = set()
    if kwargs["mode"] in ("semantic", "hybrid"):
        needed.add(kwargs.get("precision", "binary"))
    if kwargs.get("rerank"):
        needed.add(kwargs["rerank"])
    return needed


def resolve_methods(selection: "list[str]") -> "list[str]":
    """Expand group names, drop duplicates, keep METHODS' order."""
    chosen = set()
    for name in selection:
        chosen.update(GROUPS.get(name, [name]))
    return [key for key in METHODS if key in chosen]


def embeddings(index: Index, collection: str, precision: str):
    """The ``{id: vector}`` view for one collection/precision."""
    tables = {
        ("documents", "binary"): index.document_embeddings_binary,
        ("documents", "int8"): index.document_embeddings_int8,
        ("documents", "float"): index.document_embeddings,
        ("queries", "binary"): index.query_embeddings_binary,
        ("queries", "int8"): index.query_embeddings_int8,
        ("queries", "float"): index.query_embeddings,
    }
    return tables[(collection, precision)]


def check_index(index: Index, methods: "list[str]") -> None:
    """Fail before the first search if any selected method cannot run.

    A missing vec0 table is not an error inside scrydb -- ``_vec_search``
    simply returns no rows for it -- so a float search against an index
    built without float vectors yields a well-formed, entirely empty run
    that effectiveness.py would happily score as zero. Checking all
    methods up front also means a 13-method sweep does not fail on the
    twelfth after an hour of work.
    """
    if not len(index.documents):
        raise SystemExit("scrydb-pipeline: the index has no documents")
    if not len(index.queries):
        # batch_search(queries=None) would silently return an empty Run.
        raise SystemExit(
            "scrydb-pipeline: the index has no stored queries -- pass --queries, or use "
            "index.search(...) for one-off query text"
        )

    # Every check below reads a vec0 table, which only exists when the
    # sqlite-vec extension was loaded -- and a BM25-only selection
    # deliberately opens the index without it.
    vector_methods = [k for k in methods if precisions_used(METHODS[k][1])]
    if not vector_methods:
        return

    unavailable = {}
    for key in vector_methods:
        for precision in sorted(precisions_used(METHODS[key][1])):
            if len(embeddings(index, "documents", precision)):
                continue
            unavailable.setdefault(precision, []).append(key)
    if unavailable:
        detail = "; ".join(
            f"no {precision} document vectors, needed by {', '.join(keys)}"
            for precision, keys in sorted(unavailable.items())
        )
        raise SystemExit(
            f"scrydb-pipeline: {detail} -- re-index with --reindex, which stores whatever "
            "precisions this --methods selection needs, or drop those methods"
        )

    # The query side has a fallback the document side does not: a stored
    # vector at the exact precision is used verbatim, and failing that a
    # stored float vector is quantized down to it per query (_query_expr).
    # Neither existing means every vector search re-encodes the text -- and
    # with no model attached, matches nothing.
    if not len(index.query_embeddings_binary) and not len(index.query_embeddings):
        raise SystemExit(
            "scrydb-pipeline: the queries were indexed without embeddings, so "
            f"{', '.join(vector_methods)} would match nothing -- re-index --queries from a "
            "JSONL carrying an --embedding-field, or select --methods bm25"
        )


def build_index(index: Index, args, store_int8: bool) -> None:
    """Index whatever JSONL sources were given, unless the index has them."""
    sources = [(args.documents, False), (args.queries, True)]
    populated = len(index.documents) and len(index.queries)
    if populated and not args.reindex:
        log(f"index already holds {len(index.documents)} documents and {len(index.queries)} "
            "queries -- reusing it (--reindex to rebuild from the JSONL)")
        return

    for path, is_query in sources:
        if path is None:
            continue
        kind = "queries" if is_query else "documents"
        method = index.index_queries if is_query else index.index_documents
        log(f"indexing {kind} from {path}")
        try:
            method(
                source=path,
                id_field=args.query_id_field if is_query else args.doc_id_field,
                text_field=args.text_field,
                embedding_field=args.embedding_field,
                batch_size=args.batch_size,
                limit=args.limit,
                # Binary is always stored; float is what every cosine rerank
                # needs, and int8 costs a byte per dimension, so it is stored
                # only when a selected method actually ranks at it.
                store_full_embeddings=True,
                store_int8_embeddings=store_int8,
            )
        except KeyError as exc:
            # _index_rows raises a bare KeyError naming the missing field.
            # Only the id and text fields raise: a wrong --embedding-field
            # is not an error there, it just indexes no vectors at all --
            # which check_index() catches before any search runs.
            flag = "--query-id-field" if is_query else "--doc-id-field"
            raise SystemExit(
                f"scrydb-pipeline: {kind}: {exc.args[0]} -- check {flag}/--text-field "
                f"against the first line of {path}"
            )


def summarize(index: Index, db_path: Path, vec_loaded: bool) -> None:
    if not vec_loaded:
        # Counting vectors means reading a vec0 table, which a BM25-only
        # run has deliberately not loaded the extension for.
        vectors = " (vectors not counted -- sqlite-vec not loaded)"
    else:
        stored = []
        for label, table in (
            ("binary", index.document_embeddings_binary),
            ("int8", index.document_embeddings_int8),
            ("float", index.document_embeddings),
        ):
            if len(table):
                stored.append(f"{len(table)} {label}")
        vectors = f" ({', '.join(stored)} document vectors)" if stored else " (no vectors -- BM25 only)"
    size_mb = db_path.stat().st_size / (1024 * 1024)
    log(f"{db_path.name}: {len(index.documents)} documents, {len(index.queries)} queries"
        f"{vectors}, {size_mb:.1f} MiB")


def main(argv: "list[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(
        description="Index JSONL, run every scrydb search method over it, write TREC runs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="methods: " + ", ".join(list(GROUPS) + list(METHODS)),
    )
    parser.add_argument(
        "dataset",
        help="Dataset name, e.g. nfcorpus -- names the JSONL sources, the index and the runs",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["all"],
        metavar="METHOD",
        help="Methods or groups to run (default: all). Groups: " + ", ".join(GROUPS),
    )
    parser.add_argument("--documents", type=Path, help="Documents JSONL (default: <dataset>/corpus-embeddings.jsonl)")
    parser.add_argument("--queries", type=Path, help="Queries JSONL (default: <dataset>/queries-embeddings.jsonl)")
    parser.add_argument("--db", type=Path, help="Index path (default: <dataset>.db under the index dir)")
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR, help="Where runs are written (default: %(default)s)")
    parser.add_argument("--doc-id-field", default="docid", help="Document id key (default: %(default)s)")
    parser.add_argument("--query-id-field", default="qid", help="Query id key (default: %(default)s)")
    parser.add_argument("--text-field", default="text", help="Text key in both files (default: %(default)s)")
    parser.add_argument(
        "--embedding-field", default="embedding", help="Embedding key in both files (default: %(default)s)"
    )
    parser.add_argument("--batch-size", type=int, default=1000, help="Rows per write batch (default: %(default)s)")
    parser.add_argument(
        "--int8-embeddings",
        action="store_true",
        help="Store int8 vectors even when no selected method needs them",
    )
    parser.add_argument("--reindex", action="store_true", help="Re-index from the JSONL even if the index is populated")
    parser.add_argument("--overwrite", action="store_true", help="Rewrite run files that already exist")
    parser.add_argument(
        "--top-k", type=int, default=1000, help="Results per query (default: %(default)s, the published depth)"
    )
    parser.add_argument(
        "--rerank-depth", type=int, default=1000, help="Candidates a rerank stage sees (default: %(default)s)"
    )
    parser.add_argument(
        "--candidate-limit",
        type=int,
        default=1000,
        help="How deep each stage retrieves before RRF fusion, hybrid only. scrydb's own "
             "default of 50 would cap a hybrid run at 100 results (default: %(default)s)",
    )
    parser.add_argument(
        "--tag",
        help="TREC run tag, the 6th column (default: the method key, so runs are self-identifying)",
    )
    parser.add_argument("--limit", type=int, help="Index and search only the first N rows/queries (smoke test)")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan and exit without touching anything")
    args = parser.parse_args(argv)

    unknown = [m for m in args.methods if m not in GROUPS and m not in METHODS]
    if unknown:
        parser.error(f"unknown method(s): {', '.join(unknown)} -- pick from {', '.join(list(GROUPS) + list(METHODS))}")
    methods = resolve_methods(args.methods)

    dataset_dir = DEFAULT_DATASET_DIR / args.dataset
    if args.documents is None and (dataset_dir / "corpus-embeddings.jsonl").is_file():
        args.documents = dataset_dir / "corpus-embeddings.jsonl"
    if args.queries is None and (dataset_dir / "queries-embeddings.jsonl").is_file():
        args.queries = dataset_dir / "queries-embeddings.jsonl"
    for path in (args.documents, args.queries):
        if path is not None and not path.is_file():
            parser.error(f"no such file: {path}")

    # A truncated experiment gets its own index and its own run names. Both
    # matter: a --limit index reused later would silently be a 20-document
    # corpus, and a truncated run under the canonical name would be scored
    # by effectiveness.py as if it covered the whole query set.
    stem = f"{args.dataset}-limit{args.limit}" if args.limit else args.dataset
    db_path = args.db or DEFAULT_INDEX_DIR / f"{stem}.db"
    run_paths = OrderedDict((key, args.runs_dir / f"{stem}{METHODS[key][0]}") for key in methods)

    if not db_path.is_file() and args.documents is None:
        hint = (
            # --limit deliberately does not touch the full index, so it always
            # has to build its own -- which needs the JSONL, not a prebuilt .db.
            f"--limit builds its own {db_path.name}, so it needs --documents/--queries JSONL"
            if args.limit
            else f"fetch a prebuilt index with: python examples/dl.py dataset {args.dataset} "
                 "--assets index qrels"
        )
        parser.error(f"no index at {db_path} and no --documents to build one from -- {hint}")

    # int8 is the one precision scrydb does not store by default, so the
    # method selection has to opt in for it at index time.
    store_int8 = args.int8_embeddings or any("int8" in precisions_used(METHODS[k][1]) for k in methods)
    todo = [k for k in methods if args.overwrite or not run_paths[k].is_file()]
    done = [k for k in methods if k not in todo]

    log(f"{len(methods)} method(s): {', '.join(methods)}")
    if done:
        log(f"{len(done)} run(s) already written, skipping: {', '.join(done)} (--overwrite to redo)")
    if args.dry_run:
        log(f"index: {db_path}{' (int8 storage on)' if store_int8 else ''}")
        for key in todo:
            log(f"would write {run_paths[key]}")
        return 0

    db_path.parent.mkdir(parents=True, exist_ok=True)
    args.runs_dir.mkdir(parents=True, exist_ok=True)

    # BM25 alone needs neither the sqlite-vec extension nor a build of
    # Python that can load extensions at all; anything else does, and so
    # does storing a vector at index time.
    needs_vec = args.documents is not None or any(precisions_used(METHODS[k][1]) for k in methods)
    started = time.perf_counter()
    results = []

    with Index.open(db_path, vec_ext_path="auto" if needs_vec else None) as index:
        build_index(index, args, store_int8)
        summarize(index, db_path, needs_vec)
        check_index(index, methods)

        # queries=None is the whole point of batch_search: every stored
        # query, in one pass, keyed by query id in the returned Run.
        queries = list(index.queries)[: args.limit] if args.limit else None

        for position, key in enumerate(todo, start=1):
            _, method_kwargs = METHODS[key]
            log(f"[{position}/{len(todo)}] {key}: {method_kwargs}")
            elapsed = time.perf_counter()
            run = index.batch_search(
                queries=queries,
                top_k=args.top_k,
                rerank_depth=args.rerank_depth,
                candidate_limit=args.candidate_limit,
                **method_kwargs,
            )
            elapsed = time.perf_counter() - elapsed

            # QID Q0 DOCID RANK SCORE TAG -- what trec_eval and friends
            # read. The SCORE column carries the last stage's field, so a
            # reranked run is scored by its rerank, not its first stage.
            run.write_trec(run_paths[key], tag=args.tag or key)
            retrieved = sum(len(hits) for hits in run.values())
            empty = sum(1 for hits in run.values() if not hits)
            results.append((key, len(run), retrieved, empty, elapsed))

    if results:
        log("")
        log(f"{'method':<26} {'queries':>8} {'results':>10} {'empty':>6} {'seconds':>9}  run")
        for key, n_queries, retrieved, empty, elapsed in results:
            log(f"{key:<26} {n_queries:>8} {retrieved:>10} {empty:>6} {elapsed:>9.1f}  {run_paths[key].name}")
    log(f"{len(results)} run(s) written to {args.runs_dir} in {time.perf_counter() - started:.1f}s")
    if args.limit:
        log("this was a --limit smoke test; drop --limit for a scoreable run")
    else:
        log(f"score them with: python scripts/evaluation/effectiveness.py --datasets {args.dataset}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
