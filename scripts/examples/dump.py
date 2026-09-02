"""Dump JSONL files for documents and queries from a scrydb SQLite index.

``index_documents``/``index_queries`` split each input row into a JSON
``payload`` (everything except the embedding field) plus separate embedding
tables, so dumping is the inverse: read each id's payload out of
``index.documents``/``index.queries``, and -- if a full-precision vector was
stored for that id (``store_full_embeddings=True``, the default) -- splice
it back in under its original ``embedding_field`` key.

Only the full-precision embedding tables (``embeddings_full`` /
``query_embeddings_full``) round-trip exactly. The ``ubinary``-quantized
tables (``embeddings`` / ``query_embeddings``) are a lossy 1-bit-per-dimension
sketch used for Hamming search and cannot be inverted back into the original
float vector, so rows with only a binary embedding are written without an
embedding field and counted separately.

Run with:
    python examples/dump.py idx.db --documents-out documents.jsonl --queries-out queries.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tqdm import tqdm

from scrydb import Index
from scrydb.core import _EmbeddingTable, _PayloadTable


def _dump_table(
    payloads: _PayloadTable,
    full_embeddings: _EmbeddingTable,
    binary_embeddings: _EmbeddingTable,
    embedding_field: str,
    out_path: Path,
    desc: str,
) -> None:
    n_rows = 0
    n_with_embedding = 0
    n_binary_only = 0
    with out_path.open("w", encoding="utf-8") as f:
        for item_id in tqdm(payloads, desc=desc, unit="rows", unit_scale=True, total=len(payloads)):
            row = dict(payloads[item_id])
            try:
                row[embedding_field] = full_embeddings[item_id].tolist()
                n_with_embedding += 1
            except KeyError:
                if item_id in binary_embeddings:
                    n_binary_only += 1
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_rows += 1
    print(
        f"scrydb-dump: wrote {n_rows} rows to {out_path} "
        f"({n_with_embedding} with a full-precision '{embedding_field}', "
        f"{n_rows - n_with_embedding} without)",
        file=sys.stderr,
    )
    if n_binary_only:
        print(
            f"scrydb-dump: {n_binary_only} rows had only a ubinary-quantized "
            "embedding (no store_full_embeddings copy) and cannot be exactly recovered",
            file=sys.stderr,
        )


def main(argv: "list[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(
        description="Dump JSONL files for documents and queries from a scrydb SQLite index."
    )
    parser.add_argument("db", help="Path to the scrydb SQLite index file")
    parser.add_argument(
        "--documents-out", default="documents.jsonl", help="Output path for dumped documents"
    )
    parser.add_argument(
        "--queries-out", default="queries.jsonl", help="Output path for dumped queries"
    )
    parser.add_argument(
        "--embedding-field", default="embedding", help="Key to restore the embedding under (default: %(default)s)"
    )
    args = parser.parse_args(argv)

    with Index.open(args.db) as index:
        if len(index.documents) > 0:
            _dump_table(
                index.documents,
                index.document_embeddings,
                index.document_embeddings_binary,
                args.embedding_field,
                Path(args.documents_out),
                "Dumping documents",
            )
        else:
            print("scrydb-dump: no documents in index, skipping", file=sys.stderr)

        if len(index.queries) > 0:
            _dump_table(
                index.queries,
                index.query_embeddings,
                index.query_embeddings_binary,
                args.embedding_field,
                Path(args.queries_out),
                "Dumping queries",
            )
        else:
            print("scrydb-dump: no queries in index, skipping", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
