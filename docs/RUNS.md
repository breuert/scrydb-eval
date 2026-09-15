# Runs of the thirteen retrieval configurations under evaluation

> [!NOTE]  
> Each maps one-to-one onto scrydb's `mode=` / `precision=` / `rerank=` vocabulary. The run-file suffix is what `effectiveness.py` globs for and what `lexical.py`/`semantic.py`/`hybrid.py`/`pipeline.py` write. 

| Run suffix | Method | `batch_search(...)` | Method key |
| :--------- | :----- | :------------------ | :--------- |
| `-bm25` | BM25 | `mode="lexical"` | `bm25` |
| `-bm25-hamming` | BM25 + Hamming | `mode="lexical", rerank="binary"` | `bm25_hamming` |
| `-bm25-cosine_int8` | BM25 + cos_int8 | `mode="lexical", rerank="int8"` | `bm25_cosine_int8` |
| `-bm25-cosine_float` | BM25 + cos_float | `mode="lexical", rerank="float"` | `bm25_cosine_float` |
| `-hamming` | Hamming | `mode="semantic", precision="binary"` | `hamming` |
| `-hamming-cosine_int8` | Hamming + cos_int8 | `mode="semantic", precision="binary", rerank="int8"` | `hamming_cosine_int8` |
| `-hamming-cosine_float` | Hamming + cos_float | `mode="semantic", precision="binary", rerank="float"` | `hamming_cosine_float` |
| `-cosine_int8` | cos_int8 | `mode="semantic", precision="int8"` | `cosine_int8` |
| `-cosine_int8-cosine_float` | cos_int8 + cos_float | `mode="semantic", precision="int8", rerank="float"` | `cosine_int8_float` |
| `-cosine_float` | cos_float | `mode="semantic", precision="float"` | `cosine_float` |
| `-rrf-hamming` | RRF(Hamming) | `mode="hybrid", precision="binary"` | `rrf_hamming` |
| `-rrf-cosine_int8` | RRF(cos_int8) | `mode="hybrid", precision="int8"` | `rrf_cosine_int8` |
| `-rrf-cosine_float` | RRF(cos_float) | `mode="hybrid", precision="float"` | `rrf_cosine_float` |

The published runs were all generated with `top_k=1000`, `rerank_depth=1000`, `candidate_limit=1000` and `rrf_k=60`, and hold up to 1000 documents per query. A BM25-first run can be shorter for a given query, since FTS5 only returns documents that actually match a query term.

**The score column** is always higher-is-better, but the quantity differs by method: negated FTS5 `bm25()` for BM25; *negative* Hamming distance for binary; cosine similarity in `[-1, 1]` for int8/float; the *rerank stage's* score for any `+ rerank` method; and the fused reciprocal-rank score (≤ `2/(60+1)`) for RRF.
