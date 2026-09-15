# scrydb-eval

Evaluation, reproduction scripts, and results for
**[SQLite is Enough. Lexical, Semantic, and Hybrid Search with scrydb](https://arxiv.org/abs/2608.24060)**.

Everything here is built around one idea: a retrieval experiment can live entirely
inside a single SQLite file. Each `<dataset>.db` holds the document and query text,
an FTS5 lexical index, and every embedding at three precisions as [`sqlite-vec`](https://github.com/asg017/sqlite-vec)
`vec0` tables: binary (1 bit/dim),
int8 (1 byte/dim) and float32. All thirteen retrieval configurations in the paper therefore run
against the *same* file, replaying stored query vectors: no model download, no GPU,
no separate vector store.

| | |
| :-- | :-- |
| Library | [`scrydb`](https://github.com/breuert/scrydb) · `pip install scrydb` |
| Prebuilt indices, qrels and runs | [`breuert/scrydb-eval`](https://huggingface.co/datasets/breuert/scrydb-eval) on the Hugging Face Hub (~46 GB) |
| Embedding model | [Qwen3-Embedding-8B](https://huggingface.co/Qwen/Qwen3-Embedding-8B) |
| Datasets | Extended BEIR variants of ArguAna, FiQA-2018, NFCorpus, Quora, SciDocs, SciFact, TREC-COVID, Touché-2020 |

## Repository layout

```
scrydb-eval/
├── docs/                # the documentation, one file per part of the repository
├── scripts/
│   ├── examples/        # working with an index: download, build, search, inspect, dump
│   ├── evaluation/      # the two experiments, plus the table/plot renderers
│   ├── datasets/beir/   # rebuilding the inputs from scratch: embeddings, qrels
│   └── requirements.txt # everything the scripts need — `pip install -r`
├── results/             # the numbers behind the paper's tables and figures
│   ├── effectiveness/   # scores.csv, table.tex, table.md, heatmap_ndcg10.pdf
│   └── efficiency/      # raw_timings.csv, …, table.tex, bar_*.pdf
└── data/                # working directory, not versioned — scripts/examples/dl.py fills it
    ├── indices/beir/<dataset>.db
    ├── runs/beir/<dataset>-<method>.txt
    ├── datasets/beir/<dataset>/qrels/<split>.trec
    └── mteb/results/results/Qwen__Qwen3-Embedding-8B/<revision>/*.json
```

## Documentation

| File | What it covers |
| :--- | :------------- |
| [`docs/EXAMPLES.md`](docs/EXAMPLES.md) | `scripts/examples/` — downloading the data, building and dumping an index, producing runs, inspecting documents, queries and vectors, the interactive prompt. Start here. |
| [`docs/RUNS.md`](docs/RUNS.md) | The thirteen retrieval configurations: run-file suffix, `batch_search(...)` arguments and method key for each. |
| [`docs/EVALUATION.md`](docs/EVALUATION.md) | `scripts/evaluation/` — the effectiveness and efficiency experiments, and re-rendering their tables and plots. |
| [`docs/RESULTS.md`](docs/RESULTS.md) | The results themselves: the effectiveness and latency tables, and what `results/` holds. |
| [`docs/DATA.md`](docs/DATA.md) | `scripts/datasets/beir/` — rebuilding the inputs from BEIR's own files: re-embedding a corpus, converting qrels. Only needed without a prebuilt index. |

## Citation

```bibtex
@misc{scrydb2026,
      title={SQLite is Enough. Lexical, Semantic, and Hybrid Search with scrydb}, 
      author={Timo Breuer},
      year={2026},
      eprint={2608.24060},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2608.24060}
}
```

The artifacts here are derived from the [BEIR benchmark](https://github.com/beir-cellar/beir); the underlying corpora remain under their original per-dataset licenses and terms of use. The `scrydb` library itself is MIT-licensed.
