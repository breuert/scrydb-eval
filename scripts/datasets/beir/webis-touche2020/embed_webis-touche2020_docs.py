#!/usr/bin/env python3
"""
Generate embeddings for WebisTouche2020 corpus documents using the Qwen3 embedding API.

Input format (BEIR corpus.jsonl, one JSON object per line):
{"_id": "...", "title": "...", "text": "...", "metadata": {...}}

The text that gets embedded is the title and text concatenated
("<title>\n\n<text>"). Documents with no title use the text as-is.

Output format: {"docid": "...", "text": "...", "embedding": [...]}
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import List
from dotenv import load_dotenv
import requests


class EmbeddingAPIError(Exception):
    """Raised for a non-2xx response from the embedding API, carrying the
    actual response body so the real reason (e.g. a token-limit violation)
    is visible instead of being swallowed by a generic requests exception."""

    # Substrings (checked case-insensitively) that indicate the API rejected
    # a request for being too long, as opposed to some other class of error
    # (auth, bad model name, outage, ...) that truncation can't fix and
    # that should abort loudly rather than be silently retried away.
    _LENGTH_ERROR_MARKERS = ("context length", "input_tokens", "too long", "maximum context")

    def __init__(self, status_code: int, body):
        self.status_code = status_code
        self.body = body
        super().__init__(f"API returned {status_code}: {body}")

    @property
    def is_length_error(self) -> bool:
        text = json.dumps(self.body) if isinstance(self.body, (dict, list)) else str(self.body)
        text = text.lower()
        return any(marker in text for marker in self._LENGTH_ERROR_MARKERS)


class EmbeddingGenerator:
    """Generate embeddings using the Qwen3 embedding API."""

    def __init__(
        self,
        api_url: str = "https://chat.kiconnect.nrw/api/v1/embeddings",
        model: str = "qwen-qwen3-embedding-8b",
        api_key: str | None = None,
        timeout: int = 60,
        batch_size: int = 10,
        delay_between_batches: float = 0.5,
    ):
        """
        Initialize the embedding generator.

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
        """
        self.api_url = api_url
        self.model = model
        self.timeout = timeout
        self.batch_size = batch_size
        self.delay_between_batches = delay_between_batches

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

        Raises
        ------
        EmbeddingAPIError
            If the API returns a non-2xx response. Carries the parsed
            response body (e.g. the API's own "maximum context length ..."
            message), so callers can distinguish a too-long-input error
            from other failure classes.
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
        except requests.exceptions.RequestException as e:
            print(f"Error during API request: {e}", file=sys.stderr)
            raise

        if r.status_code != 200:
            try:
                body = r.json()
            except ValueError:
                body = r.text
            raise EmbeddingAPIError(r.status_code, body)

        data = r.json()
        return [item["embedding"] for item in data["data"]]

    def _embed_batch_resilient(self, batch_docs: List[dict], skipped_docids: List[str]):
        """Embed *batch_docs* (``[{"docid", "text"}, ...]``), recovering
        from a too-long-input 400 instead of aborting the whole batch.

        On a length-related 400, the batch is bisected and each half
        retried; once a failing sub-batch is down to a single document,
        its text is progressively halved and retried until it fits (the
        API's error is content-dependent, not a fixed character cutoff, so
        halving-and-retrying is used instead of guessing a static length
        cap). A document that still doesn't fit after repeated halving is
        skipped and recorded in *skipped_docids*.

        Any 400 whose message does NOT look length-related (auth failure,
        bad model name, outage, ...) is re-raised immediately rather than
        being bisected away, since truncation can't fix it and silently
        skipping the whole batch would mask a systemic problem.

        Returns
        -------
        list[tuple[dict, list[float]]]
            ``(doc, embedding)`` pairs for every document that succeeded.
            ``doc["text"]`` reflects whatever text was actually embedded
            (i.e. truncated, if truncation was needed), so the output JSONL
            never pairs an embedding with text that doesn't match it.
        """
        if not batch_docs:
            return []

        texts = [d["text"] for d in batch_docs]
        try:
            embeddings = self.embed_batch(texts)
            return list(zip(batch_docs, embeddings))
        except EmbeddingAPIError as e:
            if e.status_code != 400 or not e.is_length_error:
                raise
            if len(batch_docs) == 1:
                return self._embed_single_with_truncation(batch_docs[0], skipped_docids)
            mid = len(batch_docs) // 2
            left = self._embed_batch_resilient(batch_docs[:mid], skipped_docids)
            right = self._embed_batch_resilient(batch_docs[mid:], skipped_docids)
            return left + right

    def _embed_single_with_truncation(
        self, doc: dict, skipped_docids: List[str], max_attempts: int = 8
    ):
        """Halve *doc*'s text up to *max_attempts* times, retrying after
        each cut, until it embeds successfully. Falls back to skipping
        (recorded in *skipped_docids*) if it still doesn't fit."""
        text = doc["text"]
        for attempt in range(max_attempts):
            try:
                embedding = self.embed_batch([text])[0]
                if attempt > 0:
                    print(
                        f"NOTE: {doc['docid']} needed truncation to {len(text)}/{len(doc['text'])} "
                        f"chars to fit the model's context window.",
                        file=sys.stderr,
                    )
                return [({**doc, "text": text}, embedding)]
            except EmbeddingAPIError as e:
                if e.status_code != 400 or not e.is_length_error:
                    raise
                text = text[: len(text) // 2]
                if not text:
                    break

        print(
            f"WARNING: skipping {doc['docid']} — still rejected as too long after "
            f"{max_attempts} rounds of truncation.",
            file=sys.stderr,
        )
        skipped_docids.append(doc["docid"])
        return []

    @staticmethod
    def _build_text(title: str, text: str) -> str:
        """
        Combine a document's title and text into a single embeddable text.

        Parameters
        ----------
        title : str
            The document title.
        text : str
            The document text.

        Returns
        -------
        str
            Combined text with title and text separated by a blank line.
            If either part is empty, only the non-empty part is returned.
        """
        title = (title or "").strip()
        text = (text or "").strip()
        if title and text:
            return f"{title}\n\n{text}"
        return title or text

    def _get_processed_docids(self, output_path: str) -> set:
        """
        Get set of document IDs that have already been processed.

        Parameters
        ----------
        output_path : str
            Path to the output JSONL file.

        Returns
        -------
        set
            Set of document IDs that have already been processed.
        """
        output_file = Path(output_path)
        processed_docids = set()

        if output_file.exists():
            with output_file.open('r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        doc = json.loads(line)
                        if "docid" in doc:
                            processed_docids.add(doc["docid"])
                    except json.JSONDecodeError:
                        continue

        return processed_docids

    def embed_documents(
        self,
        input_path: str,
        output_path: str,
        progress: bool = True,
    ) -> None:
        """
        Generate embeddings for all documents in a BEIR corpus.jsonl file.

        This function is resumable - it will skip documents that have already
        been processed and append new results to the output file.

        Parameters
        ----------
        input_path : str
            Path to the input corpus.jsonl file with format:
            {"_id": "...", "title": "...", "text": "...", "metadata": {...}}.
        output_path : str
            Path to the output JSONL file with embeddings.
        progress : bool
            Whether to show progress information.
        """
        input_file = Path(input_path)
        output_file = Path(output_path)

        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        # Get already processed document IDs
        processed_docids = self._get_processed_docids(output_path)

        if processed_docids and progress:
            print(f"Found {len(processed_docids)} already processed documents, will skip them...")

        # Read all documents and filter out already processed ones
        documents = []
        skipped_no_text = 0
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

                docid = raw.get("_id")
                if not docid:
                    print(f"Warning: Line {line_num} missing '_id', skipping", file=sys.stderr)
                    continue

                if docid in processed_docids:
                    if progress:
                        print(f"Skipping already processed document: {docid}")
                    continue

                text = self._build_text(raw.get("title", ""), raw.get("text", ""))
                if not text:
                    skipped_no_text += 1
                    continue

                documents.append({"docid": docid, "text": text})

        if skipped_no_text and progress:
            print(f"Skipped {skipped_no_text} documents with no title or text.")

        if not documents:
            print("No new documents to process.")
            return

        # Process in batches
        total_docs = len(documents)
        processed_count = 0
        skipped_docids: List[str] = []

        if progress:
            print(f"Processing {total_docs} new documents in batches of {self.batch_size}...")

        # Open output file in append mode
        with output_file.open('a', encoding='utf-8') as f:
            for i in range(0, total_docs, self.batch_size):
                batch_docs = documents[i:i + self.batch_size]

                if progress:
                    print(f"Processing batch {i // self.batch_size + 1}/{(total_docs + self.batch_size - 1) // self.batch_size} "
                          f"(documents {i + 1}-{min(i + self.batch_size, total_docs)})...")

                try:
                    # Resilient to individual oversized documents: on a
                    # too-long-input 400 it bisects the batch and truncates
                    # (or, as a last resort, skips) just the offending
                    # document(s) instead of aborting the whole batch — see
                    # _embed_batch_resilient's docstring for why a fixed
                    # character cutoff can't be used here.
                    results = self._embed_batch_resilient(batch_docs, skipped_docids)

                    # Write batch results immediately
                    for doc, embedding in results:
                        output_doc = {
                            "docid": doc["docid"],
                            "text": doc["text"],
                            "embedding": embedding,
                        }
                        f.write(json.dumps(output_doc, ensure_ascii=False) + '\n')
                        f.flush()  # Ensure data is written to disk
                        processed_count += 1

                    # Add delay between batches to avoid rate limiting
                    if i + self.batch_size < total_docs:
                        time.sleep(self.delay_between_batches)

                except Exception as e:
                    print(f"Error processing batch {i // self.batch_size + 1}: {e}", file=sys.stderr)
                    print(f"Processed {processed_count} documents before error. You can restart to continue.")
                    raise

        if skipped_docids:
            skipped_path = output_file.with_name(output_file.stem + ".skipped.txt")
            with skipped_path.open('a', encoding='utf-8') as f:
                f.write("\n".join(skipped_docids) + "\n")
            if progress:
                print(
                    f"Skipped {len(skipped_docids)} document(s) that stayed too long for the "
                    f"model's context window even after truncation; ids appended to {skipped_path}"
                )

        if progress:
            print(f"Successfully generated embeddings for {processed_count}/{total_docs} new documents")
            print(f"Output saved to: {output_path}")


def main():
    """Main entry point for command-line usage."""
    if len(sys.argv) < 3:
        print("Usage: python embed_webis-touche2020_docs.py <input_file> <output_file> [options]")
        print()
        print("Options:")
        print("  --model MODEL_NAME        Model to use (default: qwen-qwen3-embedding-8b)")
        print("  --batch-size SIZE         Batch size for API calls (default: 1000)")
        print("  --delay SECONDS           Delay between batches (default: 0.5)")
        print("  --api-url URL             API endpoint URL")
        print("  --api-key KEY             API key (overrides THKI_API_KEY env var)")
        print()
        print("Example:")
        print("  python embed_webis-touche2020_docs.py corpus.jsonl webis-touche2020_docs_with_embeddings.jsonl")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    load_dotenv(".keys")

    # Parse optional arguments
    model = "qwen-qwen3-embedding-8b"
    batch_size = 1000
    delay = 0.5
    api_url = "https://chat.kiconnect.nrw/api/v1/embeddings"
    api_key = None

    i = 3
    while i < len(sys.argv):
        arg = sys.argv[i]

        if arg == "--model" and i + 1 < len(sys.argv):
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
        else:
            print(f"Unknown argument: {arg}", file=sys.stderr)
            sys.exit(1)

    try:
        generator = EmbeddingGenerator(
            api_url=api_url,
            model=model,
            api_key=api_key,
            batch_size=batch_size,
            delay_between_batches=delay,
        )

        generator.embed_documents(input_path, output_path)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
