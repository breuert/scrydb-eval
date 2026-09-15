#!/usr/bin/env python3
"""
dl.py
=====

Downloads the scrydb-eval resources -- the prebuilt scrydb indices, the
TREC-format qrels and the retrieval runs -- from the Hugging Face Hub:

    https://huggingface.co/datasets/breuert/scrydb-eval

plus the MTEB baseline scores the effectiveness table compares against, which
are not on the Hub but in MTEB's own results repository on GitHub:

    https://github.com/embeddings-benchmark/results

Everything is one of four asset kinds, and every file belongs to exactly one
BEIR dataset:

  index     indices/beir/<dataset>.db                 8 files,  ~28.8 GB
  qrels     qrels/<dataset>/<split>.trec             14 files,   ~7.7 MB
  runs      runs/beir/<dataset>-<method>.txt        104 files,  ~17.3 GB
  baseline  results/<model>/<revision>/<Task>.json    8 files,    ~47 KB
                                                    ------------------
                                                     everything, ~46 GB

The baselines are the per-task result files that MTEB publishes for
Qwen/Qwen3-Embedding-8B, pinned to one commit of that repository so the
numbers stay the ones the paper reports.

Modes
-----
Pick as little as you need -- the indices are large, and a single dataset's
index plus its qrels and runs is usually all an experiment requires:

  all         every file of every kind (~46 GB)
  dataset     every asset of one or more datasets, optionally narrowed with
              --assets to just the index, the qrels, the runs and/or the
              MTEB baseline
  file        one or more explicit repo paths, or glob patterns over them

`list` prints the remote inventory (with sizes, and what is already on disk)
without downloading anything; --dry-run does the same for a chosen selection.

Local layout
------------
Files land under --dest (default: <repo>/data). With the default
--layout eval the download lines up with what the evaluation scripts expect:

  data/indices/beir/<dataset>.db                  <- efficiency.py
  data/runs/beir/<dataset>-<method>.txt           <- effectiveness.py
  data/datasets/beir/<dataset>/qrels/<split>.trec <- effectiveness.py
  data/mteb/results/results/<model>/<rev>/*.json  <- effectiveness.py

Two kinds are placed elsewhere than their repository path. The qrels live at
qrels/<dataset>/<split>.trec on the Hub, which effectiveness.py's --qrels-dir
cannot express, so they are copied into the layout above after downloading
(they are a few MB in total, and the mirrored copy is left in place so re-runs
stay incremental). The baselines are written straight under data/mteb/results/,
i.e. where a checkout of MTEB's results repository would put them, which is
effectiveness.py's --mteb-dir default. Pass --layout mirror to keep the
repository paths verbatim and nothing else.

Downloads are resumable and incremental: re-running only fetches what is
missing or has changed on the Hub, so an interrupted 46 GB pull can simply be
started again. Use --force to re-fetch regardless.

Usage
-----
    python dl.py list
    python dl.py list --datasets nfcorpus
    python dl.py all
    python dl.py dataset nfcorpus
    python dl.py dataset nfcorpus scifact --assets index qrels
    python dl.py dataset nfcorpus --assets baseline
    python dl.py file indices/beir/nfcorpus.db
    python dl.py --dest ./scratch file "runs/beir/nfcorpus-*"

Run `python dl.py --help` (or `python dl.py <mode> --help`) for all options.

Requires `pip install -U huggingface_hub`. The dataset is public; set HF_TOKEN
(or pass --token) only if you are rate-limited or behind an authenticated
mirror. The baselines are fetched over plain HTTPS from raw.githubusercontent.com
and need no credentials.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import shutil
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download, snapshot_download

# ---------------------------------------------------------------------------
# Paths -- the destination is anchored on this file's location (not the cwd),
# so the script writes to the repo's data/ directory however it is invoked.
# ---------------------------------------------------------------------------
REPO_ID = "breuert/scrydb-eval"
REPO_TYPE = "dataset"

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DEST = REPO_ROOT / "data"

INDEX_PREFIX = "indices/beir/"
RUNS_PREFIX = "runs/beir/"
QRELS_PREFIX = "qrels/"

# ---------------------------------------------------------------------------
# The MTEB baselines. They are not part of the Hub dataset: they are the
# per-task result files MTEB publishes in its own GitHub repository, for the
# embedding model this evaluation uses. MTEB_COMMIT pins the results repo and
# MTEB_MODEL_REVISION the model revision inside it, so the baseline column of
# the effectiveness table stays exactly the one the paper reports; --mteb-revision
# overrides the former.
# ---------------------------------------------------------------------------
MTEB_REPO = "embeddings-benchmark/results"
MTEB_COMMIT = "48e49b5fc4a937fa43672c7107531a70c52bce49"
MTEB_MODEL = "Qwen__Qwen3-Embedding-8B"
MTEB_MODEL_REVISION = "4e423935c619ae4df87b646a3ce949610c66241c"
MTEB_PREFIX = f"results/{MTEB_MODEL}/{MTEB_MODEL_REVISION}/"
MTEB_RAW = "https://raw.githubusercontent.com"
MTEB_API = "https://api.github.com"
# Where a checkout of the results repo would sit under --dest, and therefore
# what effectiveness.py's --mteb-dir default points into.
MTEB_LOCAL_ROOT = ("mteb", "results")

# BEIR/scrydb dataset name -> MTEB result filename. The names differ (MTEB uses
# its task names), so unlike the Hub assets the baselines cannot be discovered
# by pattern; this map is the same one effectiveness.py reads them back with.
DATASET_TO_MTEB_FILE = {
    "arguana": "ArguAna.json",
    "fiqa": "FiQA2018.json",
    "nfcorpus": "NFCorpus.json",
    "quora": "QuoraRetrieval.json",
    "scidocs": "SCIDOCS.json",
    "scifact": "SciFact.json",
    "trec-covid": "TRECCOVID.json",
    "webis-touche2020": "Touche2020.json",
}

# Asset kinds selectable with --assets, in listing order. Anything else in the
# Hub repository (README.md, .gitattributes, .gitkeep) is kind "meta" and is
# only reachable by naming it explicitly in `file` mode.
ASSETS = ("index", "qrels", "runs", "baseline")

# Ask before starting a download bigger than this, unless --yes is given.
CONFIRM_BYTES = 1_000_000_000


@dataclass(frozen=True)
class RemoteFile:
    """One remote file, tagged with where it comes from, what it is and what it belongs to."""

    path: str  # path within its repository, e.g. "indices/beir/nfcorpus.db"
    size: int  # bytes, as reported by the remote (0 if unknown)
    kind: str  # "index" | "qrels" | "runs" | "baseline" | "meta"
    dataset: str | None  # BEIR dataset the file belongs to, None for "meta"
    source: str = "hf"  # "hf" (the Hub dataset) | "mteb" (the GitHub results repo)


# ---------------------------------------------------------------------------
# Remote inventory
# ---------------------------------------------------------------------------
def kind_of(path: str) -> str:
    if path.startswith(INDEX_PREFIX) and path.endswith(".db"):
        return "index"
    if path.startswith(QRELS_PREFIX) and path.endswith(".trec"):
        return "qrels"
    if path.startswith(RUNS_PREFIX) and path.endswith(".txt"):
        return "runs"
    return "meta"


def dataset_of(path: str, kind: str, datasets: list[str]) -> str | None:
    if kind == "index":
        return path[len(INDEX_PREFIX) : -len(".db")]
    if kind == "qrels":
        return path.split("/")[1]
    if kind == "runs":
        # Dataset names themselves contain hyphens ("trec-covid",
        # "webis-touche2020"), so a run file is attributed to the longest known
        # dataset name its stem starts with rather than by splitting on "-".
        stem = path[len(RUNS_PREFIX) : -len(".txt")]
        candidates = [d for d in datasets if stem.startswith(f"{d}-")]
        return max(candidates, key=len) if candidates else None
    return None


def mteb_url(name: str, revision: str) -> str:
    """Raw-content URL of one MTEB result file at the pinned revision."""
    return f"{MTEB_RAW}/{MTEB_REPO}/{revision}/{MTEB_PREFIX}{name}"


def mteb_sizes(revision: str) -> dict[str, int]:
    """File sizes in the model's results directory, via one GitHub API call.

    Best-effort only: the API is rate-limited for anonymous callers, and the
    sizes are cosmetic (they make `list` and the download plan readable). On any
    failure every baseline is reported as unknown and still downloads fine --
    raw.githubusercontent.com, which serves the files themselves, is not
    rate-limited the same way.
    """
    url = f"{MTEB_API}/repos/{MTEB_REPO}/contents/{MTEB_PREFIX.rstrip('/')}?ref={revision}"
    try:
        with urllib.request.urlopen(url, timeout=15) as response:
            entries = json.load(response)
        return {e["name"]: int(e.get("size", 0)) for e in entries if e.get("type") == "file"}
    except (OSError, ValueError, KeyError, TypeError):
        return {}


def list_mteb_files(revision: str) -> list[RemoteFile]:
    """The baseline files, one per dataset that MTEB publishes a task result for."""
    sizes = mteb_sizes(revision)
    return [
        RemoteFile(f"{MTEB_PREFIX}{name}", sizes.get(name, 0), "baseline", dataset, "mteb")
        for dataset, name in sorted(DATASET_TO_MTEB_FILE.items(), key=lambda kv: kv[1])
    ]


def list_remote_files(
    revision: str, token: str | None, mteb_revision: str
) -> tuple[list[RemoteFile], list[str], str]:
    """Return (files, dataset names, resolved Hub commit) for both sources."""
    api = HfApi(token=token)
    info = api.repo_info(REPO_ID, repo_type=REPO_TYPE, revision=revision, files_metadata=True)

    sizes = {s.rfilename: (s.size or 0) for s in info.siblings}
    # Datasets are discovered from whichever indices exist -- nothing is
    # hardcoded here, so a dataset added to the Hub repo shows up by itself.
    # A dataset that only has a baseline (no index yet) is still selectable.
    indexed = sorted(
        p[len(INDEX_PREFIX) : -len(".db")]
        for p in sizes
        if p.startswith(INDEX_PREFIX) and p.endswith(".db")
    )
    datasets = sorted(set(indexed) | set(DATASET_TO_MTEB_FILE))

    files = []
    for path in sorted(sizes):
        kind = kind_of(path)
        files.append(RemoteFile(path, sizes[path], kind, dataset_of(path, kind, indexed)))
    files.extend(list_mteb_files(mteb_revision))
    return files, datasets, info.sha


# ---------------------------------------------------------------------------
# Selection and local placement
# ---------------------------------------------------------------------------
def select(
    files: list[RemoteFile],
    datasets: list[str] | None = None,
    assets: list[str] | None = None,
    patterns: list[str] | None = None,
) -> list[RemoteFile]:
    selected = files
    if patterns is not None:
        selected = [
            f for f in selected if any(fnmatch.fnmatch(f.path, pat) for pat in patterns)
        ]
    else:
        # Without explicit paths only real assets are in play, never meta files.
        selected = [f for f in selected if f.kind in (assets or ASSETS)]
        if datasets is not None:
            selected = [f for f in selected if f.dataset in datasets]
    return selected


def target_path(f: RemoteFile, dest: Path, layout: str) -> Path:
    """Where the file should end up locally, which is not always where it downloads to."""
    if layout == "eval" and f.kind == "qrels":
        _, dataset, name = f.path.split("/")
        return dest / "datasets" / "beir" / dataset / "qrels" / name
    if layout == "eval" and f.kind == "baseline":
        return dest.joinpath(*MTEB_LOCAL_ROOT, f.path)
    return dest / f.path


def human(num_bytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if num_bytes < 1000 or unit == "TB":
            return f"{num_bytes:.0f} {unit}" if unit == "B" else f"{num_bytes:.1f} {unit}"
        num_bytes /= 1000
    raise AssertionError("unreachable")


def is_present(f: RemoteFile, path: Path) -> bool:
    """Cheap "already downloaded" check -- exact size match against the Hub."""
    return path.is_file() and (f.size == 0 or path.stat().st_size == f.size)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def print_inventory(files: list[RemoteFile], dest: Path, layout: str) -> None:
    for kind in (*ASSETS, "meta"):
        group = [f for f in files if f.kind == kind]
        if not group:
            continue
        total = sum(f.size for f in group)
        print(f"\n{kind}  ({len(group)} files, {human(total)})")
        for f in group:
            local = target_path(f, dest, layout)
            mark = "*" if is_present(f, local) else " "
            print(f"  {mark} {human(f.size):>9}  {f.path}")
    total = sum(f.size for f in files)
    print(f"\n{len(files)} files, {human(total)} total   (* = already in {dest})")


def print_plan(files: list[RemoteFile], dest: Path, layout: str) -> tuple[int, int]:
    """Print what a selection would fetch; return (missing file count, missing bytes)."""
    missing = [f for f in files if not is_present(f, target_path(f, dest, layout))]
    have = len(files) - len(missing)
    total = sum(f.size for f in files)
    todo = sum(f.size for f in missing)

    print(f"Selected {len(files)} files ({human(total)}) -> {dest}")
    if have:
        print(f"  {have} already present, {len(missing)} to download ({human(todo)})")
    for f in missing[:20]:
        print(f"    {human(f.size):>9}  {f.path}")
    if len(missing) > 20:
        print(f"    ... and {len(missing) - 20} more")
    return len(missing), todo


def confirm(todo_bytes: int, assume_yes: bool) -> bool:
    if assume_yes or todo_bytes < CONFIRM_BYTES or not sys.stdin.isatty():
        return True
    answer = input(f"Download {human(todo_bytes)}? [y/N] ").strip().lower()
    return answer in ("y", "yes")


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------
def download(
    files: list[RemoteFile],
    dest: Path,
    layout: str,
    revision: str,
    token: str | None,
    workers: int,
    force: bool,
    mteb_revision: str,
) -> list[Path]:
    """Fetch the selection into ``dest`` and return the final local paths.

    The two sources are fetched independently: the Hub assets through
    huggingface_hub, the baselines over plain HTTPS from GitHub.
    """
    dest.mkdir(parents=True, exist_ok=True)
    hub = [f for f in files if f.source == "hf"]
    baselines = [f for f in files if f.source == "mteb"]

    paths = [f.path for f in hub]
    if len(paths) == 1:
        hf_hub_download(
            REPO_ID,
            paths[0],
            repo_type=REPO_TYPE,
            revision=revision,
            token=token,
            local_dir=dest,
            force_download=force,
        )
    elif paths:
        # snapshot_download fetches the files in parallel and resumes whatever a
        # previous, interrupted invocation left behind. The exact repo paths are
        # used as allow_patterns, which fnmatch matches literally.
        snapshot_download(
            REPO_ID,
            repo_type=REPO_TYPE,
            revision=revision,
            token=token,
            local_dir=dest,
            allow_patterns=paths,
            max_workers=workers,
            force_download=force,
        )

    written = {f.path: place(f, dest, layout, force) for f in hub}
    for f in baselines:
        written[f.path] = fetch_baseline(f, dest, layout, mteb_revision, force)
    return [written[f.path] for f in files]


def fetch_baseline(
    f: RemoteFile, dest: Path, layout: str, revision: str, force: bool
) -> Path:
    """Download one MTEB result file straight to its final path.

    Unlike the Hub assets there is no cache in between, so the file is written
    to a sibling .part first and moved into place only once complete -- an
    interrupted download then leaves nothing that a re-run would mistake for a
    finished file.
    """
    target = target_path(f, dest, layout)
    if not force and is_present(f, target):
        return target

    url = mteb_url(f.path[len(MTEB_PREFIX) :], revision)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    try:
        with urllib.request.urlopen(url, timeout=60) as response, partial.open("wb") as out:
            shutil.copyfileobj(response, out)
    except urllib.error.HTTPError as exc:
        partial.unlink(missing_ok=True)
        sys.exit(f"Could not download {url}: HTTP {exc.code} {exc.reason}")
    except OSError as exc:
        partial.unlink(missing_ok=True)
        sys.exit(f"Could not download {url}: {exc}")
    partial.replace(target)
    return target


def place(f: RemoteFile, dest: Path, layout: str, force: bool = False) -> Path:
    """Copy a downloaded file to its final path when the layout moves it (qrels only)."""
    downloaded = dest / f.path
    target = target_path(f, dest, layout)
    if target != downloaded and (force or not is_present(f, target)):
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(downloaded, target)
    return target


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download scrydb-eval indices, qrels and runs from the Hugging Face Hub.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python dl.py list\n"
            "  python dl.py all\n"
            "  python dl.py dataset nfcorpus\n"
            "  python dl.py dataset nfcorpus scifact --assets index qrels\n"
            "  python dl.py dataset nfcorpus --assets baseline\n"
            "  python dl.py file indices/beir/nfcorpus.db\n"
        ),
    )
    p.add_argument("--dest", type=Path, default=DEFAULT_DEST, help="Download directory")
    p.add_argument(
        "--layout",
        choices=("eval", "mirror"),
        default="eval",
        help="'eval' also places qrels and MTEB baselines where the evaluation "
        "scripts expect them (data/datasets/beir/<dataset>/qrels/, data/mteb/results/); "
        "'mirror' keeps the repository paths only",
    )
    p.add_argument("--revision", default="main", help="Branch, tag or commit to download")
    p.add_argument(
        "--mteb-revision",
        default=MTEB_COMMIT,
        help=f"Commit, tag or branch of {MTEB_REPO} to take the baselines from "
        f"(default: {MTEB_COMMIT[:7]}, the one the paper reports)",
    )
    p.add_argument("--token", default=None, help="Hugging Face token (default: HF_TOKEN env)")
    p.add_argument("--workers", type=int, default=8, help="Parallel download workers")
    p.add_argument("--force", action="store_true", help="Re-download even if already present")
    p.add_argument("--dry-run", action="store_true", help="Print the selection, download nothing")
    p.add_argument("-y", "--yes", action="store_true", help="Skip the size confirmation prompt")

    sub = p.add_subparsers(dest="mode", required=True)

    lst = sub.add_parser("list", help="List the remote files and their sizes")
    lst.add_argument("--datasets", nargs="+", default=None, help="Restrict to these datasets")
    lst.add_argument("--assets", nargs="+", choices=ASSETS, default=None, help="Restrict to these kinds")

    sub.add_parser("all", help="Download everything (~46 GB)")

    ds = sub.add_parser("dataset", help="Download the assets of one or more datasets")
    ds.add_argument("datasets", nargs="+", help="Dataset names, e.g. nfcorpus scifact")
    ds.add_argument(
        "--assets",
        nargs="+",
        choices=ASSETS,
        default=list(ASSETS),
        help=f"Which assets to fetch (default: {' '.join(ASSETS)})",
    )

    fl = sub.add_parser("file", help="Download specific repo paths or glob patterns")
    fl.add_argument("paths", nargs="+", help='e.g. indices/beir/nfcorpus.db or "runs/beir/nfcorpus-*"')

    return p.parse_args()


def main() -> None:
    args = parse_args()
    dest = args.dest.expanduser().resolve()

    files, datasets, sha = list_remote_files(args.revision, args.token, args.mteb_revision)
    print(f"{REPO_ID} @ {args.revision} ({sha[:7]}) -- {len(datasets)} datasets")
    print(f"{MTEB_REPO} @ {args.mteb_revision[:7]} -- baselines for {MTEB_MODEL}")

    if args.mode == "list":
        if args.datasets:
            unknown = sorted(set(args.datasets) - set(datasets))
            if unknown:
                sys.exit(f"Unknown dataset(s): {', '.join(unknown)}\nAvailable: {', '.join(datasets)}")
        selected = select(files, datasets=args.datasets, assets=args.assets)
        print_inventory(selected, dest, args.layout)
        return

    if args.mode == "all":
        selected = select(files)
    elif args.mode == "dataset":
        unknown = sorted(set(args.datasets) - set(datasets))
        if unknown:
            sys.exit(f"Unknown dataset(s): {', '.join(unknown)}\nAvailable: {', '.join(datasets)}")
        selected = select(files, datasets=args.datasets, assets=args.assets)
    else:  # file
        selected = select(files, patterns=args.paths)
        if not selected:
            sys.exit(
                f"No file in {REPO_ID} matches: {', '.join(args.paths)}\n"
                "Run `python dl.py list` to see the available paths."
            )

    if not selected:
        sys.exit("Nothing selected.")

    n_missing, todo = print_plan(selected, dest, args.layout)
    if args.dry_run:
        return
    if not args.force and n_missing == 0:
        print("Everything is already downloaded. Use --force to re-fetch.")
        return
    if not confirm(todo, args.yes):
        sys.exit("Aborted.")

    written = download(
        selected,
        dest,
        args.layout,
        args.revision,
        args.token,
        args.workers,
        args.force,
        args.mteb_revision,
    )

    print(f"\nDone -- {len(written)} files in {dest}")
    for path in written:
        if path.suffix == ".db":
            print(f"  scrydb.Index.open({str(path)!r})")


if __name__ == "__main__":
    main()
