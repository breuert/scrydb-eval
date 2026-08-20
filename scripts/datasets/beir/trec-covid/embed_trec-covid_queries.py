#!/usr/bin/env python3
"""
Generate embeddings for TrecCovid queries using the Qwen3 embedding API.

Input format (BEIR queries.jsonl, one JSON object per line):
{"_id": "...", "text": "...", "metadata": {...}}

Output format: {"qid": "...", "text": "...", "embedding": [...]}

Query text is embedded with instruction formatting for better retrieval performance.
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import List, Set
from dotenv import load_dotenv
import requests


def get_detailed_instruct(task_description: str, query: str) -> str:
    """
    Format query with instruction for better embedding performance.

    Parameters
    ----------
    task_description : str
        Description of the retrieval task.
    query : str
        The original query text.

    Returns
    -------
    str
        Formatted query with instruction.
    """
    return f'Instruct: {task_description}\nQuery:{query}'


class QueryEmbeddingGenerator:
    """Generate embeddings for queries using the Qwen3 embedding API."""

    def __init__(
        self,
        api_url: str = "https://chat.kiconnect.nrw/api/v1/embeddings",
        model: str = "qwen-qwen3-embedding-8b",
        api_key: str | None = None,
        timeout: int = 60,
        batch_size: int = 10,
        delay_between_batches: float = 0.5,
        task_description: str = "Given a web search query, retrieve relevant passages that answer the query",
    ):
        """
        Initialize the query embedding generator.

        Parameters
        ----------
        api_url : str
            The API endpoint URL.
        model : str
            The model name to use for embeddings.
        api_key : str | None
            API key for authentication. If None, reads from THKI_API_KEY env var.
        timeout : int
            Request timeout in seconds.
        batch_size : int
            Number of texts to embed in a single API call.
        delay_between_batches : float
            Delay in seconds between batch requests to avoid rate limiting.
        task_description : str
            Task description for instruction formatting.
        """
        self.api_url = api_url
        self.model = model
        self.timeout = timeout
        self.batch_size = batch_size
        self.delay_between_batches = delay_between_batches
        self.task_description = task_description

        if api_key is None:
            api_key = os.getenv('THKI_API_KEY')

        if not api_key:
            raise ValueError(
                "API key must be provided either as parameter or "
                "via THKI_API_KEY environment variable"
            )

        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts.

        Parameters
        ----------
        texts : List[str]
            List of texts to embed.

        Returns
        -------
        List[List[float]]
            List of embedding vectors.
        """
        payload = {
            "input": texts,
            "model": self.model,
        }

        try:
            r = requests.post(
                self.api_url,
                headers=self.headers,
                json=payload,
                timeout=self.timeout,
            )
            r.raise_for_status()
            data = r.json()

            embeddings = [item["embedding"] for item in data["data"]]
            return embeddings

        except requests.exceptions.RequestException as e:
            print(f"Error during API request: {e}", file=sys.stderr)
            raise

    def _load_blacklist(self, blacklist_path: str | None) -> Set[str]:
        """
        Load blacklisted query IDs from a file.

        Parameters
        ----------
        blacklist_path : str | None
            Path to the blacklist file. If None, returns empty set.

        Returns
        -------
        Set[str]
            Set of blacklisted query IDs.
        """
        if blacklist_path is None:
            return set()

        blacklist_file = Path(blacklist_path)
        if not blacklist_file.exists():
            print(f"Warning: Blacklist file not found: {blacklist_path}", file=sys.stderr)
            return set()

        blacklisted = set()
        with blacklist_file.open('r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    blacklisted.add(line)

        return blacklisted

    def _get_processed_qids(self, output_path: str) -> Set[str]:
        """
        Get set of query IDs that have already been processed.

        Parameters
        ----------
        output_path : str
            Path to the output JSONL file.

        Returns
        -------
        Set[str]
            Set of query IDs that have already been processed.
        """
        output_file = Path(output_path)
        processed_qids = set()

        if output_file.exists():
            with output_file.open('r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        doc = json.loads(line)
                        if "qid" in doc:
                            processed_qids.add(doc["qid"])
                    except json.JSONDecodeError:
                        continue

        return processed_qids

    def embed_queries(
        self,
        input_path: str,
        output_path: str,
        blacklist_path: str | None = None,
        progress: bool = True,
    ) -> None:
        """
        Generate embeddings for all queries in a BEIR queries.jsonl file.

        This function is resumable - it will skip queries that have already
        been processed and append new results to the output file.

        Parameters
        ----------
        input_path : str
            Path to the input queries.jsonl file with format:
            {"_id": "...", "text": "...", "metadata": {...}}.
        output_path : str
            Path to the output JSONL file with embeddings.
        blacklist_path : str | None
            Path to file containing blacklisted query IDs (one per line).
        progress : bool
            Whether to show progress information.
        """
        input_file = Path(input_path)
        output_file = Path(output_path)

        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        # Load blacklist
        blacklisted = self._load_blacklist(blacklist_path)
        if blacklisted and progress:
            print(f"Loaded {len(blacklisted)} blacklisted query IDs")

        # Get already processed query IDs
        processed_qids = self._get_processed_qids(output_path)

        if processed_qids and progress:
            print(f"Found {len(processed_qids)} already processed queries, will skip them...")

        # Read all queries and filter out blacklisted and already processed ones
        queries = []
        with input_file.open('r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    raw = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse line {line_num}: {e}", file=sys.stderr)
                    continue

                qid = raw.get("_id")
                query_text = raw.get("text")

                if not qid or query_text is None:
                    print(f"Warning: Line {line_num} missing '_id' or 'text', skipping", file=sys.stderr)
                    continue

                # Skip blacklisted queries
                if qid in blacklisted:
                    if progress:
                        print(f"Skipping blacklisted query: {qid}")
                    continue

                # Skip already processed queries
                if qid in processed_qids:
                    if progress:
                        print(f"Skipping already processed query: {qid}")
                    continue

                queries.append({
                    "qid": qid,
                    "text": query_text,
                })

        if not queries:
            print("No new queries to process.")
            return

        # Process in batches
        total_queries = len(queries)
        processed_count = 0

        if progress:
            print(f"Processing {total_queries} new queries in batches of {self.batch_size}...")

        # Open output file in append mode
        with output_file.open('a', encoding='utf-8') as f:
            for i in range(0, total_queries, self.batch_size):
                batch_queries = queries[i:i + self.batch_size]

                # Format queries with instruction
                formatted_queries = [
                    get_detailed_instruct(self.task_description, q["text"])
                    for q in batch_queries
                ]

                if progress:
                    print(f"Processing batch {i // self.batch_size + 1}/{(total_queries + self.batch_size - 1) // self.batch_size} "
                          f"(queries {i + 1}-{min(i + self.batch_size, total_queries)})...")

                try:
                    batch_embeddings = self.embed_batch(formatted_queries)

                    # Write batch results immediately
                    for query, embedding in zip(batch_queries, batch_embeddings):
                        output_query = {
                            "qid": query["qid"],
                            "text": query["text"],
                            "embedding": embedding,
                        }
                        f.write(json.dumps(output_query, ensure_ascii=False) + '\n')
                        f.flush()  # Ensure data is written to disk
                        processed_count += 1

                    # Add delay between batches to avoid rate limiting
                    if i + self.batch_size < total_queries:
                        time.sleep(self.delay_between_batches)

                except Exception as e:
                    print(f"Error processing batch {i // self.batch_size + 1}: {e}", file=sys.stderr)
                    print(f"Processed {processed_count} queries before error. You can restart to continue.")
                    raise

        if progress:
            print(f"Successfully generated embeddings for {processed_count}/{total_queries} new queries")
            print(f"Output saved to: {output_path}")


def main():
    """Main entry point for command-line usage."""
    if len(sys.argv) < 3:
        print("Usage: python embed_trec-covid_queries.py <input_file> <output_file> [options]")
        print()
        print("Options:")
        print("  --blacklist FILE         File containing blacklisted query IDs (one per line)")
        print("  --model MODEL_NAME        Model to use (default: qwen-qwen3-embedding-8b)")
        print("  --batch-size SIZE         Batch size for API calls (default: 1000)")
        print("  --delay SECONDS           Delay between batches (default: 0.5)")
        print("  --api-url URL             API endpoint URL")
        print("  --api-key KEY             API key (overrides THKI_API_KEY env var)")
        print("  --task TASK               Task description for instruction formatting")
        print()
        print("Example:")
        print("  python embed_trec-covid_queries.py queries.jsonl trec-covid-queries-embeddings.jsonl")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    load_dotenv(".env.keys")

    # Parse optional arguments
    blacklist_path = None
    model = "qwen-qwen3-embedding-8b"
    batch_size = 1000
    delay = 0.5
    api_url = "https://chat.kiconnect.nrw/api/v1/embeddings"
    api_key = None
    task_description = "Given a web search query, retrieve relevant passages that answer the query"

    i = 3
    while i < len(sys.argv):
        arg = sys.argv[i]

        if arg == "--blacklist" and i + 1 < len(sys.argv):
            blacklist_path = sys.argv[i + 1]
            i += 2
        elif arg == "--model" and i + 1 < len(sys.argv):
            model = sys.argv[i + 1]
            i += 2
        elif arg == "--batch-size" and i + 1 < len(sys.argv):
            batch_size = int(sys.argv[i + 1])
            i += 2
        elif arg == "--delay" and i + 1 < len(sys.argv):
            delay = float(sys.argv[i + 1])
            i += 2
        elif arg == "--api-url" and i + 1 < len(sys.argv):
            api_url = sys.argv[i + 1]
            i += 2
        elif arg == "--api-key" and i + 1 < len(sys.argv):
            api_key = sys.argv[i + 1]
            i += 2
        elif arg == "--task" and i + 1 < len(sys.argv):
            task_description = sys.argv[i + 1]
            i += 2
        else:
            print(f"Unknown argument: {arg}", file=sys.stderr)
            sys.exit(1)

    try:
        generator = QueryEmbeddingGenerator(
            api_url=api_url,
            model=model,
            api_key=api_key,
            batch_size=batch_size,
            delay_between_batches=delay,
            task_description=task_description,
        )

        generator.embed_queries(input_path, output_path, blacklist_path=blacklist_path)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
