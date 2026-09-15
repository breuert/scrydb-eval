# Evaluating effectiveness and efficiency

> [!NOTE]  
> Two scripts in `scripts/evaluation/` do the measuring and write CSVs; six more render tables and plots from those CSVs without re-running anything. The renderers import from `effectiveness.py` / `efficiency.py`, so run them as scripts (`python scripts/evaluation/…`) rather than importing them yourself.

| Script | Reads | Writes |
| :----- | :---- | :----- |
| [`effectiveness.py`](scripts/evaluation/effectiveness.py) | runs + qrels + MTEB baseline | `scores.csv`, `table.tex` |
| [`effectiveness_table.py`](scripts/evaluation/effectiveness_table.py) | `scores.csv` | `table.tex` |
| [`effectiveness_md.py`](scripts/evaluation/effectiveness_md.py) | `scores.csv` | `table.md` |
| [`effectiveness_plot.py`](scripts/evaluation/effectiveness_plot.py) | `scores.csv` | `heatmap_<measure>.pdf` |
| [`efficiency.py`](scripts/evaluation/efficiency.py) | the `.db` indices | 4 CSVs, `table.tex`, 2 PDFs |
| [`efficiency_table.py`](scripts/evaluation/efficiency_table.py) | the efficiency CSVs | `efficiency_table.tex` |
| [`efficiency_md.py`](scripts/evaluation/efficiency_md.py) | the efficiency CSVs | `efficiency_table.md` |
| [`efficiency_plot.py`](scripts/evaluation/efficiency_plot.py) | `summary.csv`, `overall_summary.csv` | `bar_by_dataset.pdf`, `bar_overall.pdf` |

## `effectiveness.py` — AP, RR, P@10, nDCG@10

```bash
python scripts/evaluation/effectiveness.py
python scripts/evaluation/effectiveness.py --datasets trec-covid scifact --output-dir ./out
```

Evaluates every (dataset, method) run found under `data/runs/beir` against its
TREC-format qrels and compares the results against the full-precision embedding baseline
MTEB reports for `Qwen/Qwen3-Embedding-8B`. Datasets are auto-discovered from whichever
`*-bm25.txt` files are present, and methods from the filename suffixes in the table
below, so adding a dataset's runs is enough for it to appear — nothing is hardcoded.
Missing runs and missing qrels are reported and skipped, not fatal.

Because it needs only runs and qrels, effectiveness reproduces **without the ~29 GB of
indices**:

```bash
python scripts/examples/dl.py file "runs/beir/*" "qrels/*" "results/*"
python scripts/evaluation/effectiveness.py
```

Inputs and their defaults:

- `--runs-dir data/runs/beir`
- `--qrels-dir data/datasets/beir`, read as `<dataset>/qrels/<split>.trec`, with
  `--qrels-split test` (the split the paper uses)
- `--mteb-dir data/mteb/results/results/Qwen__Qwen3-Embedding-8B/4e423935c619ae4df87b646a3ce949610c66241c`,
  one `<Task>.json` per dataset in MTEB's published results layout. `dl.py` fetches
  these as the `baseline` asset kind, from commit `48e49b5` of
  [`embeddings-benchmark/results`](https://github.com/embeddings-benchmark/results),
  and puts them exactly there (`python scripts/examples/dl.py file "results/*"`, or
  `--assets baseline` for one dataset); an existing checkout of MTEB's results tree
  dropped in as `data/mteb/results/` works just as well, as does pointing `--mteb-dir`
  elsewhere. Without it the baseline column is skipped with a warning and everything
  else still runs.

**Self-matches.** BEIR draws some datasets' queries from the corpus itself, so a query's
own document can be retrieved for it. BEIR's reference implementation never scores that
document and the MTEB baseline inherits the exclusion, so every run is filtered the same
way here before it is evaluated. `--keep-self-matches` restores the unfiltered behaviour
for quantifying the artifact — not for reporting. Of the eight datasets only ArguAna is
materially affected: the self-match takes rank 1 for 91% of its queries, and excluding it
raises cos_float from 0.514 to 0.724 nDCG@10. FiQA has 9 further incidental id
collisions; the other six datasets have none.

Table styling: `--decimals`, `--no-bold-best`, `--header-angle`.

## `efficiency.py` — query latency

```bash
python scripts/evaluation/efficiency.py
python scripts/evaluation/efficiency.py --n-queries 10 --n-reps 10               # quicker
python scripts/evaluation/efficiency.py --datasets scifact nfcorpus --n-queries 30
python scripts/evaluation/efficiency.py --methods bm25 bm25_hamming rrf_hamming  # subset
```

Benchmarks wall-clock latency of all thirteen pipelines against every `<dataset>.db`
under `--indices-dir` (default `data/indices/beir`, auto-discovered; `--datasets`
narrows it). This one **does** need the indices.

Three design decisions are worth knowing before comparing numbers:

- Every timed call is driven by a *stored query id*, so it reuses the precomputed query
  embedding instead of re-encoding text. What is measured is database-side retrieval cost
  — SQL execution plus any NumPy rerank arithmetic — not model inference, which is a
  separate, batchable, GPU-dependent concern.
- For each (dataset, method, query) one untimed warm-up call populates the OS/SQLite page
  cache, then `--n-reps` timed repetitions are recorded. Per-query means are computed
  first, and all summary statistics (mean / std / 95% CI) are computed *across queries*:
  queries, not repeated calls of one query, are the independent sampling unit.
- The cross-dataset **Mean** row treats *datasets* as the sampling unit, so a dataset with
  many more queries than another does not dominate the average.

Retrieval knobs — note these differ from the published runs' settings, so latency and
effectiveness are not measured at the same depths: `--top-k 10`, `--rerank-depth 100`,
`--candidate-limit 50`, `--rrf-k 60`. Sampling: `--n-queries 50`, `--n-reps 20`,
`--seed 42`. Extension: `--vec-ext auto|none|<path>` (`none` leaves only `bm25`
runnable). Presentation: `--decimals`, `--no-bold-best`, `--header-angle`, and
`--style color|patterns|grayscale` for colorblind- and print-friendly bars.

Outputs, all in `--output-dir` (default `./results/efficiency`):

| File | Contents |
| :--- | :------- |
| `raw_timings.csv` | every timed call: dataset, method, query_id, rep, ms |
| `per_query_means.csv` | mean latency per (dataset, method, query) |
| `summary.csv` | mean/std/sem/95% CI per (dataset, method) |
| `overall_summary.csv` | mean/std/sem/95% CI per method, across datasets |
| `table.tex` | rows = datasets + a Mean row, columns = methods |
| `bar_by_dataset.pdf`, `bar_overall.pdf` | grouped and averaged bar charts |

## Re-rendering tables and plots

All six renderers read the CSVs above and never re-run an experiment, so restyling,
relabelling or subsetting is cheap:

```bash
# LaTeX
python scripts/evaluation/effectiveness_table.py --no-bold-best
python scripts/evaluation/efficiency_table.py --methods bm25 hamming rrf_hamming

# plots
python scripts/evaluation/effectiveness_plot.py --measure all
python scripts/evaluation/efficiency_plot.py --style grayscale

# Markdown
python scripts/evaluation/effectiveness_md.py --output -
python scripts/evaluation/efficiency_md.py --output -
```

`effectiveness_plot.py` draws a heatmap of one measure (`--measure nDCG@10` by default,
`all` for a 2×2 grid), with the best and second-best method per row boxed solid and
dashed. `efficiency_table.py` and `efficiency_md.py` add two context columns after
"Dataset" so latency can be read against scale and verbosity: corpus size, and mean
query length in words computed over exactly the queries that were timed for that row
(the ids in `per_query_means.csv`, with text looked up in
`data/datasets/beir/<dataset>/queries.jsonl`). That queries file is not part of this
repository, so the column degrades to `--` with a warning when it is absent;
`efficiency_md.py --no-query-length` drops it entirely.

### Markdown tables

`effectiveness_md.py` and `efficiency_md.py` render the same two tables as
GitHub-flavoured Markdown. Markdown has no `\multirow` or `\underline`, so dataset names
repeat across their measure rows and the runner-up is *italicised* (`--second-best
underline` uses a `<u>` tag instead); `cos_int8`/`cos_float` become
`cos<sub>int8</sub>`/`cos<sub>float</sub>`, the equivalent of the paper's
`cos\textsubscript{}` notation (`--labels plain` keeps them literal). Other shared flags:
`--title`, `--heading-level`, `--no-caption`, `--decimals`, `--no-bold-best`.

`effectiveness_md.py --layout per-measure --measures nDCG@10` gives the narrower
one-table-per-measure layout, which reads better in a README than the full
(dataset, measure) block layout.

Either script can splice its table straight into a Markdown file between
`<!-- BEGIN <marker> -->` / `<!-- END <marker> -->` comments — appended to the end of the
file if the markers are not there yet — so re-running it after a new evaluation refreshes
the numbers in place.

```bash
python scripts/evaluation/effectiveness_md.py --measures nDCG@10 --inject README.md
python scripts/evaluation/efficiency_md.py --inject README.md
```

`--marker` renames the comment pair (defaults: `effectiveness-table`,
`efficiency-table`). Keep `--measures nDCG@10` on the first command unless you want all
four measures. 
