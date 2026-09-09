#!/usr/bin/env python3
"""Look queries up by id, the way you would in a dict.

``index.queries`` is the same read-only ``collections.abc.Mapping`` that
``examples/documents.py`` tours over the corpus -- ``{query_id: query_dict}``,
every operation a SQL statement, nothing cached:

    index.queries["PLAIN-1"]            # the payload dict, or KeyError
    index.queries.get(query_id, {})     # ... or a default
    query_id in index.queries           # one indexed lookup
    len(index.queries)                  # SELECT COUNT(*)
    for query_id in index.queries: ...  # ids, streamed from a cursor

What makes the query side worth its own script is that a query id is an
*address*, not just a key. ``batch_search(queries=None)`` resolves to
``list(index.queries)`` -- iterating this mapping is literally what drives
a full run -- and passing ids instead of text makes scrydb prefer each
query's stored embedding over re-encoding its text. That is the whole
reproducibility story: with ``index_queries`` having written the vectors,
a semantic run replays off the database with no model loaded and no
sentence-transformers import, bit-for-bit as first published.

The other half is the join. A ``Run`` is keyed by query id, and so are
TREC qrels and every run file in ``data/runs/``, so turning any of them
back into readable text is one lookup per id -- exactly what
``lexical.py``'s ``preview`` does.

The payload is whatever was indexed: a BEIR query set comes back as
``{"qid": ..., "text": ...}``. Only ``text`` carries meaning to scrydb
(``batch_search`` reads it for the lexical and hybrid stages); everything
else is passed through untouched, and a query with no ``text`` at all is
still searchable semantically off its stored vector.

Run with:
    python examples/queries.py idx.db                     # the guided tour
    python examples/queries.py idx.db PLAIN-1 PLAIN-10    # look up two ids
    python examples/queries.py idx.db --ids - --json      # ids on stdin
    python examples/queries.py idx.db PLAIN-1 --search semantic

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

# How many rows the tour prints per section, how many hits --search shows,
# and how much of a text field fits on one terminal line.
SAMPLE = 3
HITS = 3
WIDTH = 88

# Score fields in the same priority order as scrydb's Run._SCORE_FIELDS --
# whichever is present reflects the ranking that produced the hit.
SCORE_FIELDS = ("cosine_similarity", "int8_similarity", "hamming_distance", "score")


def log(message: str) -> None:
    print(f"scrydb-queries: {message}", file=sys.stderr)


def section(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


def one_line(value, width: int = WIDTH) -> str:
    """Collapse a payload value onto a single line for printing."""
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    text = " ".join(text.split())
    return text if len(text) <= width else text[: width - 1] + "…"


def show(query_id: str, query: dict, field: "str | None", as_json: bool) -> None:
    """Print one looked-up query: the whole row, one field, or raw JSON."""
    if as_json:
        print(json.dumps({"id": query_id, **query}, ensure_ascii=False))
        return
    if field is not None:
        # .get, not [], because the payload has no guaranteed schema.
        print(f"[{query_id}] {field}={one_line(query.get(field, ''))}")
        return
    print(f"[{query_id}] {one_line(query.get('text', ''), WIDTH - len(query_id) - 3)}")
    for key, value in query.items():
        if key != "text":
            print(f"    {key:<12} {one_line(value, WIDTH - 12)}")


def show_hits(hits, indent: str = "    ") -> None:
    """Print the top of one query's ranking, with whichever score field the
    mode that produced it attached."""
    if not hits:
        # A legitimate empty result: BM25 returns nothing when no query
        # term is in the corpus vocabulary.
        print(f"{indent}(no matches)")
        return
    for rank, hit in enumerate(hits, start=1):
        name = next((f for f in SCORE_FIELDS if f in hit), None)
        score = f"{name}={hit[name]:.4f}" if name else ""
        text = one_line(hit["document"].get("text", ""), WIDTH - 34)
        print(f"{indent}{rank}. [{hit['id']:<12}] {score:<26} {text}")


# ---------------------------------------------------------------------------
# Direct lookups -- the reason the mapping exists
# ---------------------------------------------------------------------------

def lookup(
    index: Index,
    query_ids: "list[str]",
    field: "str | None",
    as_json: bool,
    search_mode: "str | None",
) -> int:
    """Resolve *query_ids* against the stored query set, reporting misses.

    This is the join that turns a file keyed by query id -- a TREC run, a
    qrels file, a list of failures worth eyeballing -- back into text.
    ``.get`` keeps one unknown id from aborting the whole join, which
    matters when the ids come from a run built against a different version
    of the query set.
    """
    missing = []
    for query_id in query_ids:
        query = index.queries.get(query_id)
        if query is None:
            missing.append(query_id)
            continue
        show(query_id, query, field, as_json)
        if search_mode:
            # The id, not the text, is what gets searched: batch_search
            # reads the stored text for the lexical stage and the stored
            # embedding for the semantic one.
            run = index.batch_search(queries=[query_id], mode=search_mode, top_k=HITS)
            show_hits(run[query_id])
    if missing:
        log(f"{len(missing)} of {len(query_ids)} ids are not in the query set: {', '.join(missing[:10])}")
    return 1 if missing and len(missing) == len(query_ids) else 0


# ---------------------------------------------------------------------------
# The guided tour -- each section is one part of the Mapping protocol
# ---------------------------------------------------------------------------

def tour(index: Index, field: "str | None", as_json: bool) -> int:
    queries = index.queries

    section("len() and repr()")
    # COUNT(*) on an indexed table: one statement, no payloads read.
    print(f"len(index.queries) -> {len(queries)}")
    print(f"repr(index.queries) -> {queries!r}")
    if not len(queries):
        raise SystemExit(
            "scrydb-queries: this index has no stored queries -- add some with "
            "examples/index.py --queries, or search one-off text with index.search(...)"
        )

    section("iteration yields ids, lazily")
    # __iter__ is a bare `SELECT id`, streamed row by row. This is not just
    # a convenience: _resolve_query_ids(None) *is* list(index.queries), so
    # batch_search(queries=None) evaluates exactly what this iterates, in
    # this order.
    sample = list(itertools.islice(queries, SAMPLE))
    print(f"list(islice(index.queries, {SAMPLE})) -> {sample}")
    print("batch_search(queries=None) evaluates list(index.queries) -- this same set, in this order")

    section("index.queries[query_id]")
    first = sample[0]
    query = queries[first]
    print(f"index.queries[{first!r}] ->")
    show(first, query, field, as_json)
    print(f"\npayload keys: {list(query)}")
    print("(only 'text' means anything to scrydb; the rest is passed through as indexed)")

    section("a missing id raises KeyError")
    try:
        queries["no-such-query"]
    except KeyError as exc:
        print(f"index.queries['no-such-query'] -> KeyError({exc.args[0]!r})")
    # .get is the same lookup with the exception swallowed.
    print(f"index.queries.get('no-such-query') -> {queries.get('no-such-query')}")
    print(f"index.queries.get('no-such-query', {{}}) -> {queries.get('no-such-query', {})}")

    section("membership -- worth checking before you search")
    # `in` goes through __getitem__: an indexed WHERE id = ?, not a scan.
    print(f"{first!r} in index.queries -> {first in queries}")
    print(f"'no-such-query' in index.queries -> {'no-such-query' in queries}")
    # batch_search does not validate its ids up front -- _resolve_query_ids
    # passes them straight through, and the miss only surfaces once the
    # search stage finds no text (or no embedding) behind the id. Filtering
    # with `in` first turns a confusing late failure into an obvious one.
    try:
        index.batch_search(queries=["no-such-query"], mode="lexical", top_k=1)
    except ValueError as exc:
        print(f"\nbatch_search(queries=['no-such-query']) -> ValueError: {exc}")
    print("(unknown ids fail inside the search stage, not at the call -- filter them with `in`)")

    section("keys() / values() / items()")
    # Built on __iter__ + __getitem__, so values()/items() cost one query
    # per row -- fine over a slice, wrong over the whole set, and
    # dict(index.queries) is that N+1 with every payload kept in memory.
    for query_id, q in itertools.islice(queries.items(), SAMPLE):
        print(f"    {query_id:<14} {one_line(q.get('text', ''), WIDTH - 14)}")
    print(f"\n(one SELECT per id: {len(queries)} statements over the whole query set)")

    section("query ids address the stored embeddings too")
    # Same protocol, same keys, different value type: these mappings are
    # what make a run replayable without a model.
    stored = {}
    for name in ("query_embeddings", "query_embeddings_int8", "query_embeddings_binary"):
        table = getattr(index, name)
        try:
            count = len(table)
        except sqlite3.OperationalError as exc:
            # `no such module: vec0` -- these live in virtual tables, so
            # they need the extension that vec_ext_path=None skips. The
            # payload mapping never does.
            print(f"    index.{name:<24} unavailable ({exc})")
            continue
        if not count:
            # Not an error: an index built without this precision simply
            # has no such table, and the mapping reads as empty.
            print(f"    index.{name:<24} empty (not stored at this precision)")
            continue
        vector = table[first]
        stored[name] = vector
        print(f"    index.{name:<24} {count} items, [{first}] -> {vector.dtype} {vector.shape}")

    section("searching by id vs. by text")
    # search() takes text and must encode it; batch_search() takes ids and
    # prefers the stored vector, so it reproduces the published run exactly
    # -- and, as below, runs at all on a process with no model attached.
    text = query.get("text", "")
    print(f"index.search({one_line(text, 46)!r}, mode='lexical')  -- encodes/parses the text you pass")
    show_hits(index.search(text, mode="lexical", top_k=HITS) if text else [])
    print(f"\nindex.batch_search(queries=[{first!r}], mode='lexical')  -- reads that text from the mapping")
    show_hits(index.batch_search(queries=[first], mode="lexical", top_k=HITS)[first])

    if stored:
        print(f"\nindex.batch_search(queries=[{first!r}], mode='semantic')  -- uses the stored vector")
        try:
            run = index.batch_search(queries=[first], mode="semantic", top_k=HITS, precision="binary")
        except (RuntimeError, ValueError, sqlite3.OperationalError) as exc:
            print(f"    unavailable: {exc}")
        else:
            show_hits(run[first])
            # No add_model() call anywhere in this script -- the vectors
            # were written at index time and are simply read back.
            print("    (no model was ever attached: the query embedding came out of the index)")

    section("a Run is keyed by the same ids")
    # Which is why joining a run -- or a qrels file, or anything else keyed
    # by query id -- back to readable text is just this mapping again.
    run = index.batch_search(queries=sample, mode="lexical", top_k=1)
    print(f"list(run) -> {list(run)}")
    for query_id in run:
        print(f"    {query_id:<14} {one_line(index.queries.get(query_id, {}).get('text', ''), WIDTH - 14)}")
    print("\n(the same lookup examples/lexical.py's preview does over a full run)")
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
        description="Mapping-style query lookups against a scrydb SQLite index."
    )
    parser.add_argument("db", type=Path, help="Path to the scrydb SQLite index file")
    parser.add_argument(
        "ids",
        nargs="*",
        help="Query ids to look up; '-' reads ids from stdin, one per line. "
             "With no ids, print a guided tour of the mapping API instead.",
    )
    parser.add_argument(
        "--ids",
        dest="id_flag",
        action="append",
        default=[],
        metavar="ID",
        help="Another query id to look up (repeatable; '-' reads stdin)",
    )
    parser.add_argument("--field", help="Print only this payload key (e.g. text) instead of the whole row")
    parser.add_argument("--json", action="store_true", help="Print each query as one JSON line")
    parser.add_argument(
        "--search",
        choices=("lexical", "semantic", "hybrid"),
        help=f"Also run each looked-up id through batch_search and show its top {HITS} hits "
             "-- semantic and hybrid need embeddings in the index",
    )
    parser.add_argument(
        "--vec-ext",
        default="auto",
        help="sqlite-vec extension path, or 'none' to skip loading it -- payload "
             "lookups never need it, embeddings and semantic search do (default: %(default)s)",
    )
    args = parser.parse_args(argv)

    if not args.db.is_file():
        parser.error(
            f"no index at {args.db} -- fetch one with: "
            "python examples/dl.py dataset nfcorpus --assets index"
        )

    query_ids = read_ids(args.ids + args.id_flag)
    vec_ext = None if args.vec_ext.lower() in ("none", "null") else args.vec_ext
    if args.search in ("semantic", "hybrid") and vec_ext is None:
        parser.error(f"--search {args.search} needs the sqlite-vec extension; drop --vec-ext none")

    with Index.open(args.db, vec_ext_path=vec_ext) as index:
        if query_ids:
            return lookup(index, query_ids, args.field, args.json, args.search)
        log(f"{args.db.name}: no ids given, showing the mapping API")
        return tour(index, args.field, args.json)


if __name__ == "__main__":
    raise SystemExit(main())
