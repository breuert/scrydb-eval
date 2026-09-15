# Rebuilding the input data

> [!NOTE]  
> The necessary scripts are located in `scripts/datasets/beir/`. Only needed to rebuild a corpus from BEIR's own files (embeddings, qrels conversion); with a prebuilt index from `dl.py`, skip the steps below entirely.

## `<dataset>/embed_<dataset>_{docs,queries}.py`

One pair per dataset (`arguana`, `fiqa`, `nfcorpus`, `quora`, `scidocs`, `scifact`, `trec-covid`, `webis-touche2020`). They embed BEIR's `corpus.jsonl` and `queries.jsonl` with Qwen3-Embedding-8B over an HTTP embedding API and write the JSONL that `index.py` and `pipeline.py` consume:

```bash
python scripts/datasets/beir/nfcorpus/embed_nfcorpus_docs.py \
    corpus.jsonl corpus-embeddings.jsonl --batch-size 1000 --api-key <API_KEY>

python scripts/datasets/beir/nfcorpus/embed_nfcorpus_queries.py \
    queries.jsonl queries-embeddings.jsonl --batch-size 1000 --api-key <API_KEY>
```

`{"_id", "title", "text"}` in, `{"docid", "text", "embedding"}` out for documents; `{"_id", "text"}` in, `{"qid", "text", "embedding"}` out for queries. Documents are embedded as `"<title>\n\n<text>"` (text alone where there is no title), which is also exactly what is stored in the payload and indexed by FTS5, so the lexical and semantic sides of an index see identical text. Queries are wrapped in the `Instruct: <task>\nQuery:<query>` instruction format, with a per-dataset task description — e.g. *"Given a claim, find documents that refute the claim"* for ArguAna, *"Given a financial question, retrieve user replies that best answer the question"* for FiQA — overridable with `--task`.

Each script takes positional `<input_file> <output_file>` plus `--model`, `--batch-size`, `--delay` (between batches, for rate limits), `--api-url` and `--api-key`; the query scripts add `--task` and a `--blacklist` file of query ids to skip. Argument parsing is hand-rolled rather than argparse, so the two paths must come first and flags after them; running with fewer than two positionals prints the usage summary.

The API key is read from `--api-key`, else the `THKI_API_KEY` environment variable, else a dotenv file in the working directory — and that filename is inconsistent: `.keys` for the document scripts, `.env.keys` for the query scripts (and for `embed_fiqa_docs.py`). Exporting `THKI_API_KEY` or passing `--api-key` sidesteps it.

Every script is resumable: ids already present in the output file are read back and skipped, and new rows are appended, so an interrupted run loses no completed batch. The eight pairs are near-identical copies differing only in labels and the query task description, so a ninth dataset is a copy-and-edit away.

The int8 and binary representations are **not** produced here: `sqlite-vec` derives them inside SQLite at index time from the float32 vectors, so the stored quantization is exactly what a re-index of the same vectors would produce.

## `convert_qrels_to_trec.py`

BEIR ships qrels as a TSV with a `query-id / corpus-id / score` header. TREC tools want the constant iteration column:

```bash
python scripts/datasets/beir/convert_qrels_to_trec.py trec-covid/qrels/test.tsv \
    data/datasets/beir/trec-covid/qrels/test.trec
```

Relevance grades are passed through unchanged (binary for most datasets, graded 0–2 for NFCorpus, TREC-COVID and Touché).