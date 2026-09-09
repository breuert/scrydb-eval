#!/usr/bin/env python3
"""Look embeddings up by id, the way you would in a dict.

The vector counterpart to ``examples/documents.py``. Where that one reads
payloads, this one reads the vectors those payloads were indexed with --
through the same read-only ``collections.abc.Mapping`` protocol, one
mapping per collection and precision:

    index.document_embeddings          float32   full precision
    index.document_embeddings_int8     int8      1 byte/dim
    index.document_embeddings_binary   uint8     1 bit/dim, packed 8 to a byte
    index.query_embeddings             float32   ... and the same three
    index.query_embeddings_int8        int8          on the query side
    index.query_embeddings_binary      uint8

Each is ``{item_id: np.ndarray}``, so the dict vocabulary applies:

    index.document_embeddings["MED-10"]        # np.ndarray, or KeyError
    index.document_embeddings.get(doc_id)      # ... or None
    doc_id in index.document_embeddings        # one indexed lookup
    len(index.document_embeddings)             # SELECT COUNT(*)
    for doc_id in index.document_embeddings:   # ids, streamed from a cursor

Three things make these different from the payload mappings:

*They live in vec0 virtual tables.* Reading one needs the sqlite-vec
extension loaded, which ``Index.open(vec_ext_path=None)`` deliberately
skips; without it every operation here raises ``no such module: vec0``.

*A precision that was never indexed reads as empty, not as an error.* An
index built without ``--store-int8-embeddings`` still exposes
``document_embeddings_int8`` -- it just has nothing in it, and every
lookup is a ``KeyError``. Check ``len()`` before concluding a vector is
missing for one particular id. The same goes for a corpus indexed without
embeddings at all: ids live in ``index.documents`` that are absent here.

*The arrays are views on the SQLite blob.* ``np.frombuffer`` does not
copy, so what you get back is read-only; call ``.copy()`` before doing
anything in place. Binary vectors are also *packed*, 8 dimensions per
byte, so ``len(vector)`` is ``dim / 8`` -- ``np.unpackbits(v,
bitorder="little")`` recovers one 0/1 per dimension, which is the sign bit
of the corresponding float component.

Nothing is cached. Every lookup is a SELECT joining the vec0 table back to
its parent table by rowid, so pulling a single vector out of an 11 GB
index costs one query rather than a load of the matrix -- and iterating
``.items()`` costs one query per row, which is the wrong way to read a
whole corpus but the right way to read a shortlist.

Run with:
    python examples/embeddings.py idx.db                        # the guided tour
    python examples/embeddings.py idx.db MED-10 MED-14          # look up two ids
    python examples/embeddings.py idx.db MED-10 --precision int8
    python examples/embeddings.py idx.db PLAIN-2 --collection queries
    python examples/embeddings.py idx.db --ids - --npy vecs.npy # stdin -> matrix
    python examples/embeddings.py idx.db MED-10 --precision binary --unpack --json

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

import numpy as np

from scrydb import Index

# How many rows the tour prints per section, and how many vector components
# fit on one terminal line.
SAMPLE = 3
COMPONENTS = 6

# What each precision stores, for the tour's size arithmetic. bits_per_dim
# is what the vec0 column costs; the dtype is what np.frombuffer decodes the
# blob back into (scrydb's _NUMPY_DTYPE).
PRECISIONS = {
    "float": ("float32", 32, "full precision, cosine"),
    "int8": ("int8", 8, "quantized to unit range, cosine"),
    "binary": ("uint8", 1, "sign bit per dimension, packed, Hamming"),
}

# The six mappings, keyed the way the CLI names them.
ATTRIBUTE = {
    ("documents", "float"): "document_embeddings",
    ("documents", "int8"): "document_embeddings_int8",
    ("documents", "binary"): "document_embeddings_binary",
    ("queries", "float"): "query_embeddings",
    ("queries", "int8"): "query_embeddings_int8",
    ("queries", "binary"): "query_embeddings_binary",
}


def log(message: str) -> None:
    print(f"scrydb-embeddings: {message}", file=sys.stderr)


def section(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


def table(index: Index, collection: str, precision: str):
    """The mapping for one collection/precision pair."""
    return getattr(index, ATTRIBUTE[(collection, precision)])


def dimensions(vector: np.ndarray, precision: str) -> int:
    """Model dimensions behind *vector*.

    Only binary needs the correction: its 4096-dim vector arrives as 512
    packed bytes, so len() undercounts by a factor of 8.
    """
    return len(vector) * 8 if precision == "binary" else len(vector)


def unpack(vector: np.ndarray, precision: str) -> np.ndarray:
    """One value per dimension, whatever the storage.

    bitorder="little" is not cosmetic: sqlite-vec's vec_quantize_binary
    fills each byte LSB-first, so unpacking big-endian returns the right
    bits of every byte in the wrong order. It happens to leave Hamming
    distances intact -- a permutation does not change how many bits differ
    -- which is exactly why the mistake survives a smoke test and then
    quietly ruins any comparison against the float vector.
    """
    if precision != "binary":
        return vector
    return np.unpackbits(vector, bitorder="little")


def preview(vector: np.ndarray, n: int = COMPONENTS) -> str:
    """The first few components, printed like a truncated repr."""
    head = ", ".join(
        f"{value:.4f}" if np.issubdtype(vector.dtype, np.floating) else str(value)
        for value in vector[:n]
    )
    return f"[{head}{', …' if len(vector) > n else ''}]"


def show(item_id: str, vector: np.ndarray, precision: str, as_json: bool, unpacked: bool) -> None:
    """Print one looked-up vector: a summary line, or the whole thing as JSON."""
    values = unpack(vector, precision) if unpacked else vector
    if as_json:
        print(json.dumps({
            "id": item_id,
            "precision": precision,
            "dim": dimensions(vector, precision),
            # tolist() converts numpy scalars to Python ones; json cannot
            # serialize np.float32/np.int8 on its own.
            "embedding": values.tolist(),
        }))
        return
    packed = " packed" if precision == "binary" and not unpacked else ""
    print(
        f"[{item_id}] {values.dtype}{packed} shape={values.shape} "
        f"dim={dimensions(vector, precision)} bytes={vector.nbytes} {preview(values)}"
    )


# ---------------------------------------------------------------------------
# Direct lookups -- the reason the mapping exists
# ---------------------------------------------------------------------------

def lookup(
    index: Index,
    item_ids: "list[str]",
    collection: str,
    precision: str,
    as_json: bool,
    unpacked: bool,
    npy: "Path | None",
) -> int:
    """Resolve *item_ids* against one embedding mapping, reporting the misses.

    The shape most callers want: an id list from somewhere else -- a TREC
    run's top-k, a qrels file, a sample of hard negatives -- turned into
    the vectors those ids were indexed with. ``.get`` keeps one unknown id
    from aborting the join, which matters when the ids came from a run
    built against a different version of the corpus.
    """
    vectors = table(index, collection, precision)
    if not len(vectors):
        # An empty mapping and a corpus of unknown ids are indistinguishable
        # id by id, so say which one this is before printing N misses.
        log(
            f"index.{ATTRIBUTE[(collection, precision)]} is empty -- this index has no "
            f"{precision} {collection[:-1]} embeddings, so every lookup below will miss"
        )

    found, missing = [], []
    for item_id in item_ids:
        vector = vectors.get(item_id)
        if vector is None:
            missing.append(item_id)
            continue
        found.append(vector)
        if npy is None:
            show(item_id, vector, precision, as_json, unpacked)
    if missing:
        log(
            f"{len(missing)} of {len(item_ids)} ids have no {precision} vector: "
            f"{', '.join(missing[:10])}"
        )

    if npy is not None:
        if not found:
            raise SystemExit(f"scrydb-embeddings: nothing to write to {npy}")
        # np.stack copies, which is what makes the result writable -- the
        # rows themselves are read-only views on the SQLite blobs. Every
        # vector in one collection/precision has the same length, so this
        # is safe as long as the ids all resolved.
        matrix = np.stack([unpack(v, precision) if unpacked else v for v in found])
        np.save(npy, matrix)
        log(f"wrote {npy}: {matrix.shape} {matrix.dtype} ({matrix.nbytes} bytes)")

    return 1 if missing and len(missing) == len(item_ids) else 0


# ---------------------------------------------------------------------------
# The guided tour -- each section is one part of the Mapping protocol
# ---------------------------------------------------------------------------

def tour(index: Index, collection: str, precision: str) -> int:
    vectors = table(index, collection, precision)
    attribute = ATTRIBUTE[(collection, precision)]
    payloads = index.documents if collection == "documents" else index.queries

    section("the six mappings")
    # One per collection per precision, all the same protocol. A precision
    # that was never indexed is empty rather than absent, so this table is
    # the quickest way to see what an index actually contains.
    for (coll, prec), name in ATTRIBUTE.items():
        dtype, bits, note = PRECISIONS[prec]
        count = len(table(index, coll, prec))
        state = f"{count} item{'s' * (count != 1)}" if count else "empty (not stored at this precision)"
        print(f"    index.{name:<28} {dtype:<8} {bits:>2} bit/dim  {state}")
    print("\n" + "  |  ".join(f"{p}: {PRECISIONS[p][2]}" for p in PRECISIONS))

    if not len(vectors):
        raise SystemExit(
            f"scrydb-embeddings: index.{attribute} is empty -- pick a precision this "
            "index has, or re-index with it (examples/index.py)"
        )

    section("len() and repr()")
    print(f"len(index.{attribute}) -> {len(vectors)}")
    print(f"repr(index.{attribute}) -> {vectors!r}")
    # Every id in the parent table with no vector of its own: a corpus can
    # be indexed for lexical search alone, or lose rows to an encoder that
    # skipped them.
    gap = len(payloads) - len(vectors)
    print(f"len(index.{collection}) -> {len(payloads)}   ({gap} with no {precision} vector)")

    section("iteration yields ids, lazily")
    # __iter__ is a bare SELECT over the rowid join, streamed row by row --
    # no blobs are read. islice is what keeps "the first few" cheap: the
    # cursor is abandoned after three rows instead of walking the table.
    sample = list(itertools.islice(vectors, SAMPLE))
    print(f"list(islice(index.{attribute}, {SAMPLE})) -> {sample}")

    section(f"index.{attribute}[item_id]")
    first = sample[0]
    vector = vectors[first]
    show(first, vector, precision, as_json=False, unpacked=False)
    dim = dimensions(vector, precision)
    print(f"\nnp.frombuffer over the stored blob: {vector.nbytes} bytes for {dim} dimensions")
    print(f"writeable={vector.flags.writeable} -- it is a view on the blob, .copy() to modify")
    for prec in PRECISIONS:
        other = table(index, collection, prec)
        if len(other):
            print(f"    {prec:<6} {other[first].nbytes:>6} bytes/vector"
                  f"  ->  {other[first].nbytes * len(other) / 1e6:.1f} MB over {len(other)} items")

    if precision == "binary":
        section("binary vectors are packed, 8 dimensions per byte")
        bits = unpack(vector, precision)
        print(f"index.{attribute}[{first!r}]         -> {vector.dtype} {vector.shape} {preview(vector)}")
        print(f"np.unpackbits(v, bitorder='little') -> {bits.dtype} {bits.shape} {preview(bits)}")
        floats = index.document_embeddings if collection == "documents" else index.query_embeddings
        if first in floats:
            # The check that proves the bit order: binary quantization is
            # just the sign of each float component, so the unpacked bits
            # must equal (float > 0) dimension for dimension.
            signs = (floats[first] > 0).astype(np.uint8)
            print(f"(index.{'document' if collection == 'documents' else 'query'}_embeddings[{first!r}] > 0) "
                  f"-> {preview(signs)}")
            print(f"\nidentical: {np.array_equal(bits, signs)}"
                  f"   (with bitorder='big': {np.array_equal(np.unpackbits(vector), signs)})")

    section("a missing id raises KeyError")
    try:
        vectors["no-such-id"]
    except KeyError as exc:
        print(f"index.{attribute}['no-such-id'] -> KeyError({exc.args[0]!r})")
    # .get is the same lookup with the exception swallowed. Note the default
    # has to be spelled out: `or` is unusable on an array, and a bare
    # `if vectors.get(x)` raises on the ambiguous truth value of one.
    print(f"index.{attribute}.get('no-such-id') -> {vectors.get('no-such-id')}")
    print("(test the result against None -- `if array` raises on an ndarray)")

    section("membership")
    # `in` goes through __getitem__: an indexed WHERE id = ?, not a scan.
    print(f"{first!r} in index.{attribute} -> {first in vectors}")
    print(f"'no-such-id' in index.{attribute} -> {'no-such-id' in vectors}")

    section("keys() / values() / items()")
    # The Mapping mixins are built on __iter__ + __getitem__, so values()
    # and items() cost one query per item -- fine for a shortlist, wrong
    # for a corpus.
    for item_id, item_vector in itertools.islice(vectors.items(), SAMPLE):
        print(f"    {item_id:<14} {preview(item_vector, 4)}")
    ids = list(itertools.islice(vectors, SAMPLE))
    matrix = np.stack([vectors[i] for i in ids])
    print(f"\nnp.stack([index.{attribute}[i] for i in ids]) -> {matrix.dtype} {matrix.shape}")
    print(f"({len(ids)} SELECTs here, {len(vectors)} for the whole collection -- "
          "stack a shortlist, not a corpus)")

    section("these are the vectors search ranks with")
    demo(index, collection, precision, first)
    return 0


def demo(index: Index, collection: str, precision: str, fallback_id: str) -> None:
    """Recompute a search score by hand from the looked-up vectors.

    The point of the section: the mapping is not a parallel copy of the
    index, it is the index. Scoring the stored vectors in numpy reproduces
    what sqlite-vec reported, so a mapping lookup is the way to audit a
    ranking, probe a distance, or reuse an embedding somewhere else --
    without re-encoding anything.
    """
    query_ids = list(itertools.islice(index.queries, 1))
    if not query_ids:
        print("(this index has no stored queries, so there is nothing to rank)")
        return
    query_id = query_ids[0]
    query_vectors = table(index, "queries", precision)
    query_vector = query_vectors.get(query_id)
    if query_vector is None:
        print(f"(no {precision} vector stored for query {query_id})")
        return

    hits = index.batch_search(
        queries=[query_id], mode="semantic", precision=precision, top_k=SAMPLE
    )[query_id]
    text = " ".join((index.queries.get(query_id) or {}).get("text", "").split())
    print(f"[{query_id}] {text[:80]}\n")

    documents = table(index, "documents", precision)
    for rank, hit in enumerate(hits, start=1):
        document_vector = documents[hit["id"]]
        if precision == "binary":
            # Hamming distance: differing bits. Bit order does not matter
            # here -- both sides are permuted the same way -- so the
            # default unpackbits would agree too.
            reported = hit["hamming_distance"]
            computed = float(np.count_nonzero(unpack(query_vector, precision) != unpack(document_vector, precision)))
            name = "hamming_distance"
        else:
            # Cosine similarity, which is what sqlite-vec reports as
            # 1 - cosine_distance. int8 vectors were normalized before
            # quantization, so the cosine of the stored integers is the
            # cosine the search ran on, up to the quantization itself.
            reported = hit["cosine_similarity" if precision == "float" else "int8_similarity"]
            a, b = query_vector.astype(np.float64), document_vector.astype(np.float64)
            computed = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
            name = "cosine_similarity" if precision == "float" else "int8_similarity"
        print(f"    {rank}. [{hit['id']:<14}] {name}={reported:<12.4f} recomputed={computed:.4f}")
    print("\n(search reports the score; the mapping hands you the operands it used)")


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
        description="Mapping-style embedding lookups against a scrydb SQLite index."
    )
    parser.add_argument("db", type=Path, help="Path to the scrydb SQLite index file")
    parser.add_argument(
        "ids",
        nargs="*",
        help="Ids to look up; '-' reads ids from stdin, one per line. With no ids, "
             "print a guided tour of the mapping API instead.",
    )
    parser.add_argument(
        "--ids",
        dest="id_flag",
        action="append",
        default=[],
        metavar="ID",
        help="Another id to look up (repeatable; '-' reads stdin)",
    )
    parser.add_argument(
        "--collection",
        choices=("documents", "queries"),
        default="documents",
        help="Which side to read: the corpus or the stored queries (default: %(default)s)",
    )
    parser.add_argument(
        "--precision",
        choices=("float", "int8", "binary"),
        default="float",
        help="Which stored representation to read: float (float32), int8, or binary "
             "(packed bits). Default: %(default)s",
    )
    parser.add_argument(
        "--unpack",
        action="store_true",
        help="Expand binary vectors to one 0/1 per dimension (no-op for float/int8)",
    )
    parser.add_argument("--json", action="store_true", help="Print each vector as one JSON line")
    parser.add_argument(
        "--npy",
        type=Path,
        metavar="PATH",
        help="Stack the looked-up vectors into a matrix and np.save it there instead of printing",
    )
    parser.add_argument(
        "--vec-ext",
        default="auto",
        help="sqlite-vec extension path, or 'none' to skip loading it -- every embedding "
             "mapping reads a vec0 virtual table, so 'none' makes them all fail (default: "
             "%(default)s)",
    )
    args = parser.parse_args(argv)

    if not args.db.is_file():
        parser.error(
            f"no index at {args.db} -- fetch one with: "
            "python examples/dl.py dataset nfcorpus --assets index"
        )

    item_ids = read_ids(args.ids + args.id_flag)
    vec_ext = None if args.vec_ext.lower() in ("none", "null") else args.vec_ext

    with Index.open(args.db, vec_ext_path=vec_ext) as index:
        try:
            if item_ids:
                return lookup(
                    index, item_ids, args.collection, args.precision,
                    args.json, args.unpack, args.npy,
                )
            log(f"{args.db.name}: no ids given, showing the mapping API")
            return tour(index, args.collection, args.precision)
        except sqlite3.OperationalError as exc:
            # `no such module: vec0` -- the mappings are views over virtual
            # tables, so they need the extension the payload mappings do
            # without. Nothing here works around it.
            if "vec0" not in str(exc):
                raise
            raise SystemExit(
                f"scrydb-embeddings: {exc} -- the embedding mappings need the sqlite-vec "
                "extension; install it (pip install sqlite-vec) or pass --vec-ext with a "
                "path to the built extension"
            )


if __name__ == "__main__":
    raise SystemExit(main())
