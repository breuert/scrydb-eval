"""Build a scrydb SQLite index from JSONL documents and queries.

The inverse of ``examples/dump.py``: each input line is one JSON object,
and ``index_documents``/``index_queries`` split it into a JSON ``payload``
(every key except the embedding one) plus the ``vec0`` embedding tables.
Nothing about the JSONL schema is fixed -- the ``--*-field`` flags name
which keys carry the id, the text and the vector, so a corpus with
``{"docid": ..., "text": ..., "embedding": [...]}`` rows and one with
``{"_id": ..., "contents": ..., "vector": [...]}`` rows both index without
touching the files.

Rows carrying an embedding are indexed densely with that vector verbatim
-- no model is loaded and nothing is re-encoded, which is what makes a
benchmark reproduction exact. Rows without one still get their BM25/FTS5
entry, so a corpus with no vectors at all indexes fine and stays lexically
searchable.

Documents and queries are independent: pass either, or both. Re-running
over the same database upserts, so an interrupted run can be repeated and
a second corpus can be added to an existing index.

Run with:
    python examples/index.py idx.db --documents documents.jsonl --queries queries.jsonl

Start with --limit to check the field names against a few rows before
committing to a multi-GB file.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from scrydb import Index


def _peek_fields(path: Path) -> "list[str] | None":
    """Keys of the first JSON row in *path*, for error messages only."""
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    return list(json.loads(line).keys())
    except (OSError, ValueError):
        return None
    return None


def _index(
    index: Index,
    path: Path,
    *,
    is_query: bool,
    id_field: str,
    text_field: str,
    embedding_field: str,
    batch_size: int,
    limit: "int | None",
    store_full_embeddings: bool,
    store_int8_embeddings: bool,
) -> None:
    kind = "queries" if is_query else "documents"
    print(f"scrydb-index: indexing {kind} from {path}", file=sys.stderr)
    method = index.index_queries if is_query else index.index_documents
    try:
        method(
            source=path,
            id_field=id_field,
            text_field=text_field,
            embedding_field=embedding_field,
            batch_size=batch_size,
            limit=limit,
            store_full_embeddings=store_full_embeddings,
            store_int8_embeddings=store_int8_embeddings,
        )
    except KeyError as exc:
        # _index_rows raises a bare KeyError naming the missing field; the
        # keys the file actually has are what makes that actionable.
        found = _peek_fields(path)
        detail = f" -- first row of {path} has: {', '.join(found)}" if found else ""
        raise SystemExit(f"scrydb-index: {kind}: {exc.args[0]}{detail}")


def _report(index: Index, is_query: bool) -> None:
    kind = "queries" if is_query else "documents"
    payloads = index.queries if is_query else index.documents
    binary = index.query_embeddings_binary if is_query else index.document_embeddings_binary
    int8 = index.query_embeddings_int8 if is_query else index.document_embeddings_int8
    full = index.query_embeddings if is_query else index.document_embeddings
    n = len(payloads)
    if not n:
        return
    # Every vector is stored binary-quantized, so that count is also the
    # number of rows that carried an embedding at all; a zero there means
    # nothing matched --embedding-field and only BM25 is available.
    if not len(binary):
        print(
            f"scrydb-index: {n} {kind} in index, none with an embedding "
            "-- lexical (BM25) search only",
            file=sys.stderr,
        )
        return
    stored = [f"{len(binary)} binary"]
    if len(int8):
        stored.append(f"{len(int8)} int8")
    if len(full):
        stored.append(f"{len(full)} float")
    print(f"scrydb-index: {n} {kind} in index ({', '.join(stored)})", file=sys.stderr)


def main(argv: "list[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a scrydb SQLite index from JSONL documents and queries."
    )
    parser.add_argument("db", help="Path to the scrydb SQLite index to create or extend")
    parser.add_argument("--documents", help="Path to a JSONL documents file")
    parser.add_argument("--queries", help="Path to a JSONL queries file")
    parser.add_argument(
        "--doc-id-field", default="docid", help="Document id key (default: %(default)s)"
    )
    parser.add_argument(
        "--query-id-field", default="qid", help="Query id key (default: %(default)s)"
    )
    parser.add_argument(
        "--text-field", default="text", help="Text key in both files (default: %(default)s)"
    )
    parser.add_argument(
        "--embedding-field",
        default="embedding",
        help="Embedding key in both files (default: %(default)s)",
    )
    parser.add_argument(
        "--limit", type=int, help="Index at most this many rows per file (default: all)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=1000, help="Rows per write batch (default: %(default)s)"
    )
    parser.add_argument(
        "--no-full-embeddings",
        action="store_true",
        help="Skip the full-precision copy -- smaller index, but cosine rerank "
             "and exact dump round-trips are no longer possible",
    )
    parser.add_argument(
        "--int8-embeddings",
        action="store_true",
        help="Additionally store int8-quantized vectors (1 byte/dim)",
    )
    args = parser.parse_args(argv)

    if not args.documents and not args.queries:
        parser.error("nothing to do -- pass --documents and/or --queries")

    sources = [(Path(p), q) for p, q in ((args.documents, False), (args.queries, True)) if p]
    for path, _ in sources:
        if not path.is_file():
            parser.error(f"no such file: {path}")

    with Index.open(args.db) as index:
        for path, is_query in sources:
            _index(
                index,
                path,
                is_query=is_query,
                id_field=args.query_id_field if is_query else args.doc_id_field,
                text_field=args.text_field,
                embedding_field=args.embedding_field,
                batch_size=args.batch_size,
                limit=args.limit,
                store_full_embeddings=not args.no_full_embeddings,
                store_int8_embeddings=args.int8_embeddings,
            )
        for is_query in (False, True):
            _report(index, is_query)

    size_mb = Path(args.db).stat().st_size / (1024 * 1024)
    print(f"scrydb-index: {args.db} is {size_mb:.1f} MiB", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
