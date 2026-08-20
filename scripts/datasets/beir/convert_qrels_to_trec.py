#!/usr/bin/env python3
"""
Convert a BEIR qrels TSV file to TREC-compatible qrels format.

Input format (BEIR qrels, tab-separated, with header row):
query-id    corpus-id    score

Output format (TREC qrels, whitespace-separated, no header):
query-id    0    corpus-id    score
"""

import csv
import sys
from pathlib import Path


def convert_qrels_to_trec(input_path: str, output_path: str) -> int:
    """
    Convert a BEIR qrels TSV file to TREC-compatible qrels format.

    Parameters
    ----------
    input_path : str
        Path to the input BEIR qrels TSV file (header: query-id, corpus-id, score).
    output_path : str
        Path to the output TREC-format qrels file.

    Returns
    -------
    int
        Number of qrels rows written.
    """
    input_file = Path(input_path)
    output_file = Path(output_path)

    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    count = 0
    with input_file.open("r", encoding="utf-8", newline="") as infile, \
            output_file.open("w", encoding="utf-8") as outfile:
        reader = csv.DictReader(infile, delimiter="\t")
        for row in reader:
            query_id = row["query-id"]
            corpus_id = row["corpus-id"]
            score = row["score"]
            outfile.write(f"{query_id}\t0\t{corpus_id}\t{score}\n")
            count += 1

    return count


def main():
    """Main entry point for command-line usage."""
    if len(sys.argv) != 3:
        print("Usage: python convert_qrels_to_trec.py <input_qrels.tsv> <output_qrels.txt>")
        print()
        print("Example:")
        print("  python convert_qrels_to_trec.py trec-covid/qrels/test.tsv trec-covid/qrels/test.trec.txt")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    try:
        count = convert_qrels_to_trec(input_path, output_path)
        print(f"Wrote {count} qrels to {output_path}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
