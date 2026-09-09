#!/usr/bin/env python3
"""Look documents up by id, the way you would in a dict.

``index.documents`` is a read-only ``collections.abc.Mapping`` over the
index's ``documents`` table -- ``{doc_id: document_dict}`` -- so the whole
dict vocabulary works on a corpus that never leaves SQLite:

    index.documents["MED-10"]           # the payload dict, or KeyError
    index.documents.get(doc_id, {})     # ... or a default
    doc_id in index.documents           # one indexed lookup
    len(index.documents)                # SELECT COUNT(*)
    for doc_id in index.documents: ...  # ids, streamed from a cursor

Nothing here is cached or preloaded. Every operation is a SQL statement
against the open connection, so a lookup on the 11 GB quora index costs the
same as one on nfcorpus, and a script can hold an "in-memory corpus" it
never had the memory for. The flip side is that each operation really is a
query: the difference between iterating ids and iterating ``.items()`` is
the difference between one statement and one per document.

The payload is whatever was indexed, minus the embedding: ``index.py``
stores every key of the input JSONL row except the embedding field, so a
BEIR corpus indexed with ``--id-field docid`` comes back as
``{"docid": ..., "text": ..., "title": ...}``. There is no fixed schema to
rely on -- ``--field`` prints one key, ``--json`` prints the row as stored.

``index.queries`` and the three ``*_embeddings`` mappings implement the
same protocol, so everything below transfers to them unchanged; the
embedding tables are the one exception that needs the sqlite-vec extension
loaded, since their rows live in ``vec0`` virtual tables.

Run with:
    python examples/documents.py idx.db                    # the guided tour
    python examples/documents.py idx.db MED-10 MED-14      # look up two ids
    python examples/documents.py idx.db --ids - --json     # ids on stdin
    python examples/documents.py idx.db MED-10 --field text

Fetch an index first with:
    python examples/dl.py dataset nfcorpus --assets index
"""

from __future__ import annotations

import argparse
import itertools
import json
import sqlite3
import sys
from pathlib import Path

from scrydb import Index

# How many rows the tour prints per section, and how much of a text field
# fits on one terminal line.
SAMPLE = 3
WIDTH = 88


def log(message: str) -> None:
    print(f"scrydb-documents: {message}", file=sys.stderr)


def section(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


def one_line(value, width: int = WIDTH) -> str:
    """Collapse a payload value onto a single line for printing."""
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    text = " ".join(text.split())
    return text if len(text) <= width else text[: width - 1] + "…"


def show(doc_id: str, document: dict, field: "str | None", as_json: bool) -> None:
    """Print one looked-up document: the whole row, one field, or raw JSON."""
    if as_json:
        print(json.dumps({"id": doc_id, **document}, ensure_ascii=False))
        return
    if field is not None:
        # .get, not [], because the payload has no guaranteed schema -- a
        # corpus without titles is not an error.
        print(f"[{doc_id}] {field}={one_line(document.get(field, ''))}")
        return
    print(f"[{doc_id}]")
    for key, value in document.items():
        print(f"    {key:<12} {one_line(value, WIDTH - 12)}")


# ---------------------------------------------------------------------------
# Direct lookups -- the reason the mapping exists
# ---------------------------------------------------------------------------

def lookup(index: Index, doc_ids: "list[str]", field: "str | None", as_json: bool) -> int:
    """Resolve *doc_ids* against the corpus, reporting the misses.

    This is the shape most callers want: an id list from somewhere else --
    a TREC run, a qrels file, a sample of hard negatives -- joined back to
    the documents it names. ``.get`` keeps a single unknown id from
    aborting the whole join, which matters when the ids come from a file
    that was built against a different version of the corpus.
    """
    missing = []
    for doc_id in doc_ids:
        document = index.documents.get(doc_id)
        if document is None:
            missing.append(doc_id)
            continue
        show(doc_id, document, field, as_json)
    if missing:
        log(f"{len(missing)} of {len(doc_ids)} ids are not in the corpus: {', '.join(missing[:10])}")
    return 1 if missing and len(missing) == len(doc_ids) else 0


# ---------------------------------------------------------------------------
# The guided tour -- each section is one part of the Mapping protocol
# ---------------------------------------------------------------------------

def tour(index: Index, field: "str | None", as_json: bool) -> int:
    documents = index.documents

    section("len() and repr()")
    # COUNT(*) on an indexed table: one statement, no payloads read.
    print(f"len(index.documents) -> {len(documents)}")
    print(f"repr(index.documents) -> {documents!r}")
    if not len(documents):
        raise SystemExit("scrydb-documents: this index has no documents -- build one with examples/index.py")

    section("iteration yields ids, lazily")
    # __iter__ is a bare `SELECT id`, streamed row by row: safe to open on
    # a corpus far larger than memory, as long as you stop pulling from it.
    # islice is what makes "the first few" cheap -- the cursor is abandoned
    # after three rows instead of walking the table.
    sample = list(itertools.islice(documents, SAMPLE))
    print(f"list(islice(index.documents, {SAMPLE})) -> {sample}")

    section("index.documents[doc_id]")
    first = sample[0]
    document = documents[first]
    print(f"index.documents[{first!r}] ->")
    show(first, document, field, as_json)
    print(f"\npayload keys: {list(document)}")
    print("(the indexed row minus its embedding -- whatever index.py was given)")

    section("a missing id raises KeyError")
    try:
        documents["no-such-document"]
    except KeyError as exc:
        print(f"index.documents['no-such-document'] -> KeyError({exc.args[0]!r})")
    # .get is the same lookup with the exception swallowed.
    print(f"index.documents.get('no-such-document') -> {documents.get('no-such-document')}")
    print(f"index.documents.get('no-such-document', {{}}) -> {documents.get('no-such-document', {})}")

    section("membership")
    # `in` goes through __getitem__ -- an indexed WHERE id = ?, not a scan.
    print(f"{first!r} in index.documents -> {first in documents}")
    print(f"'no-such-document' in index.documents -> {'no-such-document' in documents}")

    section("keys() / values() / items()")
    # The Mapping mixins are built on __iter__ + __getitem__, so values()
    # and items() cost one query per document -- fine for a slice, wrong
    # for a corpus. dict(index.documents) is that same N+1 with the whole
    # corpus pulled into memory at the end of it; stream instead.
    for doc_id, doc in itertools.islice(documents.items(), SAMPLE):
        print(f"    {doc_id:<14} {one_line(doc.get('text', ''), WIDTH - 14)}")
    print(f"\n(one SELECT per id: cheap over islice, {len(documents)} statements over the corpus)")
    print("dict(index.documents) would do exactly that, then hold every payload in memory")

    section("the same protocol on queries and embeddings")
    print(f"index.queries -> {index.queries!r}")
    for query_id, query in itertools.islice(index.queries.items(), SAMPLE):
        print(f"    {query_id:<14} {one_line(query.get('text', ''), WIDTH - 14)}")
    for name in ("document_embeddings", "document_embeddings_int8", "document_embeddings_binary"):
        table = getattr(index, name)
        try:
            count = len(table)
        except sqlite3.OperationalError as exc:
            # `no such module: vec0` -- the mapping is over a virtual table,
            # so it needs the extension that Index.open(vec_ext_path=None)
            # deliberately skips. The payload mappings never do.
            print(f"    index.{name:<26} unavailable ({exc})")
            continue
        if not count:
            # Not an error: an index built without --store-int8-embeddings
            # simply has no such table, and the mapping reads as empty.
            print(f"    index.{name:<26} empty (not stored at this precision)")
            continue
        vector = table[first]
        print(f"    index.{name:<26} {count} items, [{first}] -> {vector.dtype} {vector.shape}")

    section("search results already carry the payload")
    # _materialize attaches each hit's document as it ranks, so a hit needs
    # no second lookup -- the mapping is for ids that did not come from a
    # search in the first place.
    hits = index.search(one_line(documents[first].get("text", "") or first, 60), mode="lexical", top_k=SAMPLE)
    for rank, hit in enumerate(hits, start=1):
        print(f"    {rank}. [{hit['id']:<14}] score={hit['score']:.4f} document={list(hit['document'])}")
    print("\n(hit['document'] is the same dict index.documents[hit['id']] returns)")
    return 0


def read_ids(values: "list[str]") -> "list[str]":
    """Expand a "-" among *values* into the ids read from stdin."""
    ids = []
    for value in values:
        if value == "-":
            ids.extend(line.strip() for line in sys.stdin if line.strip())
        else:
            ids.append(value)
    return ids


def main(argv: "list[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(
        description="Mapping-style document lookups against a scrydb SQLite index."
    )
    parser.add_argument("db", type=Path, help="Path to the scrydb SQLite index file")
    parser.add_argument(
        "ids",
        nargs="*",
        help="Document ids to look up; '-' reads ids from stdin, one per line. "
             "With no ids, print a guided tour of the mapping API instead.",
    )
    parser.add_argument(
        "--ids",
        dest="id_flag",
        action="append",
        default=[],
        metavar="ID",
        help="Another document id to look up (repeatable; '-' reads stdin)",
    )
    parser.add_argument("--field", help="Print only this payload key (e.g. text) instead of the whole row")
    parser.add_argument("--json", action="store_true", help="Print each document as one JSON line")
    parser.add_argument(
        "--vec-ext",
        default="auto",
        help="sqlite-vec extension path, or 'none' to skip loading it -- payload "
             "lookups never need it, the embedding mappings do (default: %(default)s)",
    )
    args = parser.parse_args(argv)

    if not args.db.is_file():
        parser.error(
            f"no index at {args.db} -- fetch one with: "
            "python examples/dl.py dataset nfcorpus --assets index"
        )

    doc_ids = read_ids(args.ids + args.id_flag)
    vec_ext = None if args.vec_ext.lower() in ("none", "null") else args.vec_ext

    with Index.open(args.db, vec_ext_path=vec_ext) as index:
        if doc_ids:
            return lookup(index, doc_ids, args.field, args.json)
        log(f"{args.db.name}: no ids given, showing the mapping API")
        return tour(index, args.field, args.json)


if __name__ == "__main__":
    raise SystemExit(main())
