# Examples of working with an index

> [!NOTE]  
> The example scripts can be found in `scripts/examples/`. 

| Script | What it does |
| :----- | :----------- |
| [`dl.py`](scripts/examples/dl.py) | Download indices, qrels, runs and the MTEB baselines |
| [`index.py`](scripts/examples/index.py) | Build an index from JSONL documents and queries |
| [`dump.py`](scripts/examples/dump.py) | The inverse — dump an index back to JSONL |
| [`lexical.py`](scripts/examples/lexical.py) | Produce a TREC run with BM25 over FTS5 |
| [`semantic.py`](scripts/examples/semantic.py) | Produce a TREC run with vector search |
| [`hybrid.py`](scripts/examples/hybrid.py) | Produce a TREC run with RRF fusion of the two |
| [`pipeline.py`](scripts/examples/pipeline.py) | End-to-end: JSONL in, all thirteen runs out, one open index |
| [`documents.py`](scripts/examples/documents.py) | Look documents up by id, dict-style |
| [`queries.py`](scripts/examples/queries.py) | Look queries up by id, dict-style |
| [`embeddings.py`](scripts/examples/embeddings.py) | Look vectors up by id, at any precision |
| [`repl.py`](scripts/examples/repl.py) | Interactive search prompt, over a toy corpus or a real index |

Each file's module docstring is the long-form version of what follows; every script
also has `--help`.

## `dl.py` — get the data

Downloads the published assets from [`breuert/scrydb-eval`](https://huggingface.co/datasets/breuert/scrydb-eval),
plus the MTEB baseline scores the effectiveness table compares against, which live in
MTEB's own [results repository](https://github.com/embeddings-benchmark/results) on
GitHub rather than on the Hub. Each file is one of four kinds and belongs to exactly
one dataset:

| Kind | Source | Path | Count | Size |
| :--- | :----- | :--- | ----: | ---: |
| `index` | Hub | `indices/beir/<dataset>.db` | 8 | ~29 GB |
| `qrels` | Hub | `qrels/<dataset>/<split>.trec` | 14 | ~8 MB |
| `runs` | Hub | `runs/beir/<dataset>-<method>.txt` | 104 | ~17 GB |
| `baseline` | GitHub | `results/<model>/<revision>/<Task>.json` | 8 | ~48 KB |

```bash
python scripts/examples/dl.py list                                    # inventory + sizes
python scripts/examples/dl.py list --datasets nfcorpus                # …for one dataset
python scripts/examples/dl.py all                                     # everything (~46 GB)
python scripts/examples/dl.py dataset nfcorpus                        # one dataset, all assets
python scripts/examples/dl.py dataset nfcorpus scifact --assets index qrels
python scripts/examples/dl.py dataset nfcorpus --assets baseline      # just the MTEB numbers
python scripts/examples/dl.py file indices/beir/nfcorpus.db           # one explicit path
python scripts/examples/dl.py file "runs/beir/nfcorpus-*"             # glob over paths
```

Downloads land under `--dest` (default `data/`) and are **resumable and incremental**:
re-running fetches only what is missing or changed, so an interrupted 46 GB pull is
restarted by repeating the command. `--force` re-fetches regardless, `--dry-run` prints
the selection without downloading, and anything above 1 GB asks for confirmation on a
terminal unless `-y` is given.

`--layout eval` (the default) additionally copies the qrels to
`data/datasets/beir/<dataset>/qrels/<split>.trec` and writes the baselines to
`data/mteb/results/results/<model>/<revision>/<Task>.json`, which is where
`effectiveness.py` looks for them; `--layout mirror` keeps the repository paths
verbatim and nothing else. Other options: `--revision`, `--token` (or `HF_TOKEN`; the
dataset is public, so this is only for rate limits or authenticated mirrors),
`--workers`, and `--mteb-revision` for the commit of MTEB's results repository the
baselines come from — it defaults to `48e49b5`, the one behind the reported numbers,
so leave it alone to reproduce the paper. The baselines themselves need no credentials;
if GitHub's API rate-limits the size lookup they are listed as `0 B` and download
normally anyway.

> All of those are options of the *top-level* parser, so they go **before** the mode,
> not after it:
>
> ```bash
> python scripts/examples/dl.py --dest ./scratch file "runs/beir/nfcorpus-*"   # correct
> python scripts/examples/dl.py file "runs/beir/nfcorpus-*" --dest ./scratch   # rejected
> ```
>
> Only `--datasets`/`--assets` belong to a mode: `list` takes both, `dataset` takes
> `--assets`.

## `index.py` — JSONL → index

One input line is one JSON object. `index_documents`/`index_queries` split each row into
a JSON payload (every key except the embedding one) plus the `vec0` embedding tables.

```bash
python scripts/examples/index.py idx.db --documents documents.jsonl --queries queries.jsonl
python scripts/examples/index.py idx.db --documents documents.jsonl --limit 100   # check fields first
```

Nothing about the JSONL schema is fixed: `--doc-id-field` (default `docid`),
`--query-id-field` (`qid`), `--text-field` (`text`) and `--embedding-field`
(`embedding`) name the keys that carry meaning, so both
`{"docid", "text", "embedding"}` and `{"_id", "contents", "vector"}` corpora index
without editing the files. Rows carrying a vector are indexed with it **verbatim** —
no model is loaded and nothing is re-encoded, which is what makes a benchmark
reproduction exact. Rows without one still get their BM25/FTS5 entry, so a corpus with
no vectors at all indexes fine and stays lexically searchable.

Documents and queries are independent (pass either or both), and re-running over the
same database upserts, so an interrupted run can simply be repeated and a second corpus
added to an existing index. `--no-full-embeddings` skips the float32 copy (smaller
index, but no cosine rerank and no exact dump round-trip); `--int8-embeddings` adds the
int8 tables, which are not stored by default. `--batch-size` sets rows per write batch.

## `dump.py` — index → JSONL

```bash
python scripts/examples/dump.py idx.db --documents-out documents.jsonl --queries-out queries.jsonl
```

Reads each id's payload back out and, where a full-precision vector was stored, splices
it in under `--embedding-field`. Only the float32 tables round-trip exactly: the
binary-quantized sketch is a lossy 1-bit-per-dimension representation and cannot be
inverted, so rows that have only a binary embedding are written without an embedding
field and reported separately on stderr.

## `lexical.py`, `semantic.py`, `hybrid.py` — one run each

The three single-method scripts. All take a dataset name, resolve the index to
`data/indices/beir/<dataset>.db` (override with `--index`), replay **every query stored
in the index** through `batch_search`, and write a TREC run to `data/runs/beir/`
(override with `--output`). Results are cut at `--top-k 1000`, the TREC convention and
the depth the published runs use, and the filename follows the convention
`effectiveness.py` globs for, so a finished run is picked up without configuration.

Common flags: `--tag` (the TREC run's 6th column), `--rerank`/`--rerank-depth`,
`--limit N` (search only the first N queries as a smoke test — this also renames the
output to `…-limitN.txt`, so a truncated run can never be mistaken for a real one) and
`--preview N` (print the top 5 hits of N queries, so a run is inspectable rather than an
opaque file of ids).

**`lexical.py`** — BM25 over FTS5. The one mode that needs nothing but SQLite: no model,
no query encoding, and the sqlite-vec extension is not even loaded unless `--rerank`
asks for a vector second stage.

```bash
python scripts/examples/lexical.py nfcorpus                                  # -bm25.txt
python scripts/examples/lexical.py nfcorpus --rerank float --rerank-depth 1000
python scripts/examples/lexical.py nfcorpus --limit 5 --preview 5            # smoke test
```

Query text is sanitized and OR-joined before it reaches FTS5's `MATCH`, so BEIR's
natural-language queries — question marks and all — do not trip the FTS5 parser; pass
`--raw` only for hand-written FTS5 expressions. `--bm25-b` and `--bm25-k1` are, despite
their names, FTS5 `bm25()` *per-column weights*, not BM25's b and k1 (FTS5 fixes those
at compile time). The defaults are scrydb's own, so a run made here matches the
published ones.

**`semantic.py`** — vector search at a chosen precision. `batch_search` prefers a
query's *stored* embedding over re-encoding its text, so a run is reproduced from the
exact vectors it was built with.

```bash
python scripts/examples/semantic.py nfcorpus                                 # binary, -hamming.txt
python scripts/examples/semantic.py nfcorpus --precision int8                # -cosine_int8.txt
python scripts/examples/semantic.py nfcorpus --precision binary --rerank float
```

`--precision binary|int8|float` picks the representation the ranking runs on (Hamming
distance for binary, cosine for the other two); `--rerank` re-scores the first stage's
hits at a finer precision and must differ from `--precision`. Unlike lexical mode, the
first stage here retrieves `--top-k` candidates and `--rerank-depth` only trims that
shortlist, so a depth above `--top-k` gives the rerank stage nothing extra (the script
says so). Both stages are checked
up front: a precision that was never indexed answers with a perfectly well-formed,
perfectly *empty* run inside scrydb, so the script fails loudly instead of letting that
silence be scored.

**`hybrid.py`** — Reciprocal Rank Fusion of the two. Each document scores `1/(k + rank)`
in every list it appears in, summed; only ranks enter the sum, never the raw scores,
which is what lets BM25 scores and cosine similarities be combined without putting them
on a common scale.

```bash
python scripts/examples/hybrid.py nfcorpus                                   # -rrf-hamming.txt
python scripts/examples/hybrid.py nfcorpus --precision float                 # -rrf-cosine_float.txt
```

The knob that matters here is `--candidate-limit` (default 1000, matching the published
runs): it sets how deep *each* stage retrieves before fusion, so the fused pool is at
most twice it. It is not `--top-k` — leaving it at scrydb's own default of 50 while
asking for 1000 results is the classic way to get a run that stops at 100.
`--rrf-k` (default 60) is the RRF constant. The preview shows each hit's rank in both
fused lists, and the summary counts how many results each stage accounted for.

> Fusion has one reproducibility wrinkle the other two modes do not: documents found by
> a single stage at the same rank get identical RRF scores, and ties come out of a `set`
> union whose order follows Python's per-process string hashing. Ranking measures are
> unaffected (eval tools re-sort by the score column), but the run file is only
> byte-identical across processes with `PYTHONHASHSEED` fixed.

## `pipeline.py` — the whole experiment in one process

Where the three scripts above each produce one run from an existing index, this one goes
from JSONL to a directory of scored-and-ready runs:

```
corpus-embeddings.jsonl  --.
                           |--> <dataset>.db --> batch_search × N --> <dataset>-<method>.txt
queries-embeddings.jsonl --'                                          …
```

```bash
python scripts/examples/pipeline.py nfcorpus --limit 20          # smoke test, minutes
python scripts/examples/pipeline.py nfcorpus                     # all thirteen methods
python scripts/examples/pipeline.py nfcorpus --methods bm25 cosine_float
python scripts/examples/pipeline.py nfcorpus --methods lexical --overwrite
python scripts/examples/pipeline.py nfcorpus --dry-run           # print the plan, touch nothing
```

The reason to run it this way rather than looping over the other three is that every
method shares one open index — the same corpus, the same stored query vectors and the
same SQLite file answering a different question, with nothing rebuilt in between.

`--methods` takes any mix of the thirteen method keys and the group shorthands `all`
(the default), `lexical`, `semantic` and `hybrid`. The selection also drives *indexing*:
int8 vectors are stored only when a selected method actually ranks or reranks at that
precision (`--int8-embeddings` forces them). Both halves are resumable — an index that
already holds documents and queries is reused as-is (`--reindex` forces the upsert), and
an existing run file is left alone (`--overwrite` replaces it) — so an interrupted
thirteen-method sweep is restarted by repeating the command.

Inputs default to `data/datasets/beir/<dataset>/{corpus,queries}-embeddings.jsonl`, as
produced by `scripts/datasets/beir/<dataset>/embed_<dataset>_*.py`; `--documents`,
`--queries`, `--db`, `--runs-dir` and the `--*-field` flags override the paths and
schema. `--limit N` caps both rows indexed and queries searched *and* moves the whole
experiment onto its own `<dataset>-limitN` index and run names, so a smoke test can
never overwrite a real one.

## `documents.py`, `queries.py`, `embeddings.py` — inspecting an index

`index.documents`, `index.queries` and the six `*_embeddings*` mappings are read-only
`collections.abc.Mapping`s, so the whole dict vocabulary works on a corpus that never
leaves SQLite — and nothing is cached or preloaded, so a lookup in the 11 GB Quora index
costs the same as one in NFCorpus. Run any of the three with no ids for a guided tour of
that API against a real index:

```bash
python scripts/examples/documents.py data/indices/beir/nfcorpus.db              # the tour
python scripts/examples/documents.py data/indices/beir/nfcorpus.db MED-10 MED-14
python scripts/examples/documents.py data/indices/beir/nfcorpus.db --ids - --json   # ids on stdin
python scripts/examples/documents.py data/indices/beir/nfcorpus.db MED-10 --field text

python scripts/examples/queries.py data/indices/beir/nfcorpus.db PLAIN-1 --search semantic

python scripts/examples/embeddings.py data/indices/beir/nfcorpus.db MED-10 --precision int8
python scripts/examples/embeddings.py data/indices/beir/nfcorpus.db PLAIN-2 --collection queries
python scripts/examples/embeddings.py data/indices/beir/nfcorpus.db --ids - --npy vecs.npy
python scripts/examples/embeddings.py data/indices/beir/nfcorpus.db MED-10 --precision binary --unpack --json
```

Ids come from positional arguments, from repeatable `--ids` flags, or from stdin one per
line (`-`). `--field` prints a single payload key, `--json` prints one JSON line per row.
`queries.py --search lexical|semantic|hybrid` additionally runs each looked-up id through
`batch_search` and shows its top hits — the join that turns a run file or a qrels file
back into readable text. `embeddings.py` selects `--collection documents|queries` and
`--precision float|int8|binary`, expands packed bit vectors with `--unpack`, and can
stack a shortlist into a matrix with `--npy PATH`.

Payload lookups never need sqlite-vec (`--vec-ext none` skips loading it); every
embedding mapping reads a `vec0` virtual table and does.

## `repl.py` — an interactive prompt

```bash
python scripts/examples/repl.py                                        # toy corpus, BM25
python scripts/examples/repl.py --index data/indices/beir/scifact.db   # a real index
python scripts/examples/repl.py --semantic                             # toy corpus, dense
```

By default it indexes a small built-in corpus for BM25 only — no embeddings, no model —
and drops you at a prompt. `--semantic` attaches a real sentence-transformers model via
`add_model(...)`, so what you type is encoded on the fly and matched against document
vectors; that mode needs `pip install "scrydb[dense]"` and a one-time model download
(~90 MB for the default MiniLM). Ask the toy corpus "how do I keep my code from
breaking" and the dense side returns the mypy/test-suite/type-hint documents while BM25
leads with asyncio and the GIL — word overlap is all it has to go on.

Prompt commands: `:k <n>`, `:help`, `:quit`, plus `:precision binary|int8|float` and
`:rerank off|binary|int8|float` in semantic mode. Flags: `--top-k`, `--text-field`,
`--max-chars`, and in semantic mode `--model`, `--query-prompt`, `--precision`,
`--rerank`.
