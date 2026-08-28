"""Self-contained, interactive search demo of scrydb.

By default, indexes a small toy corpus (a plain Python dict of documents)
for BM25/FTS5 search only -- no embeddings, no model, no `add_model(...)`
call -- then drops you into a prompt where you can type your own queries
and see ranked results against the toy corpus.

Pass `--semantic` for the dense counterpart: the same corpus, but indexed
with a real sentence-transformers model attached via `add_model(...)`, so
whatever you type at the prompt is encoded into a query vector and matched
against the documents' embeddings -- no shared vocabulary required. Ask for
"how do I keep my code from breaking" and the mypy/test-suite/type-hint
documents come back; BM25 on the same corpus and the same query leads with
asyncio and the GIL, because word overlap is all it has to go on.

A model is unavoidable in semantic mode. `toy_demo.py` gets away with
synthetic vectors because `batch_search` drives *stored* queries whose
embeddings were indexed up front; an interactive prompt has to embed unseen
text on the fly, and only a real model makes that meaningful. So `--semantic`
needs the dense extra and a one-time model download (~90 MB for the default
MiniLM):

    pip install "scrydb[dense]"

Pass `--index` to interactively query a prebuilt index instead (e.g. one
built by `examples/index.py`). Lexical search only touches its FTS5/BM25
side, so an index built without embeddings works just as well as one with
them; `--semantic` needs the embeddings, and `--model` has to match the one
they were built with.

Both modes accept prompt commands:

    :k <n>                         number of results to show
    :help                          show the commands
    :quit                          leave (blank line or Ctrl-D also works)

and `--semantic` adds two more, since the toy corpus is indexed at all three
precisions (see `store_int8_embeddings` below):

    :precision binary|int8|float   rank with this vector representation
    :rerank off|binary|int8|float  second-stage rerank of the candidates

with `:k` doubling, once a rerank is set, as how many candidates the first
stage hands to the second.

Run with:
    python examples/repl.py
    python examples/repl.py --index ../data/indices/beir/scifact.db
    python examples/repl.py --semantic
    python examples/repl.py --semantic --model mixedbread-ai/mxbai-embed-large-v1 \
        --query-prompt "Represent this query for searching relevant documents: {query}"
"""

from __future__ import annotations

import argparse
import importlib.util
import sqlite3
import sys
from pathlib import Path

from scrydb import Index, SentenceEmbedding

# A small model keeps the first run to a ~90 MB download; scrydb's own
# default (mixedbread-ai/mxbai-embed-large-v1, ~1.2 GB) is what the
# benchmark runs in this repo use. MiniLM was not trained with a query
# instruction, so the prompt is a bare "{query}" passthrough rather than
# SentenceEmbedding's mxbai-style default -- prepending an instruction a
# model never saw in training mostly just adds noise to the vector.
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_QUERY_PROMPT = "{query}"

# scrydb defaults to precision="binary", which is the right default at
# benchmark scale: 1 bit/dim over 1024-dim mxbai vectors keeps most of the
# ranking quality for 1/32nd of the storage. MiniLM's 384 dims leave binary
# far less to work with, and on a 100-document corpus the coarseness shows
# -- so this demo starts at float and lets `:precision binary` demonstrate
# the tradeoff rather than hiding it behind bad first results.
DEFAULT_PRECISION = "float"

# Mirrors core._SCORE_FIELD: which result field each precision reports its
# score under. binary is a Hamming *distance* (lower is better), int8 and
# float are cosine *similarities* (higher is better). Ordered coarsest
# first, which is also the order they are listed back to the user in.
SCORE_FIELD = {
    "binary": ("hamming_distance", "distance, lower is better"),
    "int8": ("int8_similarity", "similarity, higher is better"),
    "float": ("cosine_similarity", "similarity, higher is better"),
}

TOY_DOCUMENTS = {
    "d1": {"text": "Python is a popular language for scripting and data science."},
    "d2": {"text": "Type hints and virtual environments keep large Python codebases maintainable."},
    "d3": {"text": "A pour-over method extracts bright acidity from light-roast coffee beans."},
    "d4": {"text": "Grinding beans right before brewing preserves the aromatic oils."},
    "d5": {"text": "The James Webb telescope captured new images of a distant exoplanet."},
    "d6": {"text": "Astronomers tracked the comet's orbit as it passed near Jupiter."},
    "d7": {"text": "Debugging often means adding print statements and rereading stack traces."},
    "d8": {"text": "Single-origin beans highlight the terroir of a particular farm."},
    "d9": {"text": "Solar panels unfold automatically once a satellite reaches orbit."},
    "d10": {"text": "Searing meat at high heat locks in flavorful browning compounds."},
    "d11": {"text": "A pinch of salt balances sweetness in most dessert recipes."},
    "d12": {"text": "Overnight trains let budget travelers skip a night of hotel costs."},
    "d13": {"text": "Renewing a passport early avoids delays before an international trip."},
    "d14": {"text": "List comprehensions offer a concise way to build new lists from existing sequences."},
    "d15": {"text": "Context managers ensure resources like files and sockets are cleaned up properly."},
    "d16": {"text": "The Global Interpreter Lock complicates true multithreaded CPU-bound Python code."},
    "d17": {"text": "Virtual environments isolate project dependencies from the system-wide interpreter."},
    "d18": {"text": "Decorators wrap functions to add logging, caching, or timing without changing their body."},
    "d19": {"text": "Generators yield values lazily, which keeps memory usage low for large datasets."},
    "d20": {"text": "The asyncio library enables cooperative concurrency for I/O-bound workloads."},
    "d21": {"text": "Dataclasses reduce boilerplate when defining simple container classes."},
    "d22": {"text": "Unit tests written with pytest catch regressions before they reach production."},
    "d23": {"text": "Package managers like pip resolve dependencies from the Python Package Index."},
    "d24": {"text": "Static type checkers such as mypy catch type errors before runtime."},
    "d25": {"text": "Python's duck typing lets objects be used interchangeably if they share behavior."},
    "d26": {"text": "Profiling tools reveal which functions consume the most CPU time."},
    "d27": {"text": "The standard library's collections module provides specialized container datatypes."},
    "d28": {"text": "Refactoring legacy code often starts with adding a reliable test suite."},
    "d29": {"text": "Espresso extraction relies on fine grounds and nine bars of pressure."},
    "d30": {"text": "Cold brew steeps coarse grounds in room-temperature water for many hours."},
    "d31": {"text": "A burr grinder produces more consistent particle sizes than a blade grinder."},
    "d32": {"text": "Roast level changes the balance between acidity, sweetness, and bitterness."},
    "d33": {"text": "Arabica beans generally taste sweeter and less bitter than robusta."},
    "d34": {"text": "Water temperature just off boiling avoids scalding delicate coffee oils."},
    "d35": {"text": "Latte art depends on steaming milk into a smooth, glossy microfoam."},
    "d36": {"text": "Fair-trade certification aims to guarantee growers a stable minimum price."},
    "d37": {"text": "Freshly roasted beans release carbon dioxide for days after roasting."},
    "d38": {"text": "A French press uses a metal filter that lets more oils into the cup."},
    "d39": {"text": "Cupping is a standardized tasting method used to score coffee quality."},
    "d40": {"text": "High-altitude farms often produce beans with brighter, more complex flavors."},
    "d41": {"text": "Neutron stars pack more mass than the sun into a sphere the size of a city."},
    "d42": {"text": "The Hubble constant describes the rate at which the universe is expanding."},
    "d43": {"text": "A light-year measures distance, not time, despite its name."},
    "d44": {"text": "Saturn's rings are made mostly of ice particles with a bit of rocky debris."},
    "d45": {"text": "Black holes bend light so strongly that nothing can escape past the event horizon."},
    "d46": {"text": "The International Space Station orbits Earth roughly every ninety minutes."},
    "d47": {"text": "Mars rovers analyze soil samples for signs of ancient microbial life."},
    "d48": {"text": "Supernovae seed the surrounding galaxy with heavy elements forged in the explosion."},
    "d49": {"text": "Exoplanet hunters look for the faint dimming of starlight during a transit."},
    "d50": {"text": "The Moon's gravity is the dominant force driving Earth's ocean tides."},
    "d51": {"text": "Voyager 1 is now the most distant human-made object from Earth."},
    "d52": {"text": "Cosmic microwave background radiation is the afterglow of the early universe."},
    "d53": {"text": "Deglazing a pan with wine lifts browned bits into a flavorful sauce base."},
    "d54": {"text": "Resting meat after cooking lets its juices redistribute evenly."},
    "d55": {"text": "Blanching vegetables briefly preserves their color and crisp texture."},
    "d56": {"text": "A roux of butter and flour thickens sauces like bechamel and gravy."},
    "d57": {"text": "Fermentation transforms cabbage into tangy, probiotic-rich sauerkraut."},
    "d58": {"text": "Searing at high heat triggers the Maillard reaction's savory browning."},
    "d59": {"text": "Kneading dough develops gluten strands that give bread its chewy structure."},
    "d60": {"text": "Acidic marinades tenderize meat while adding brightness to the flavor."},
    "d61": {"text": "Caramelizing onions slowly over low heat coaxes out their natural sugars."},
    "d62": {"text": "A pinch of baking soda helps beans soften faster during cooking."},
    "d63": {"text": "Sous vide cooking holds food at a precise temperature in a water bath."},
    "d64": {"text": "Toasting spices in a dry pan intensifies their aromatic oils."},
    "d65": {"text": "Budget airlines often charge separately for checked bags and seat selection."},
    "d66": {"text": "A visa on arrival lets travelers skip pre-departure paperwork in some countries."},
    "d67": {"text": "Jet lag eases faster when travelers adjust to the destination's local time immediately."},
    "d68": {"text": "Rail passes can be cheaper than individual tickets for multi-country itineraries."},
    "d69": {"text": "Travel insurance can cover trip cancellations and emergency medical costs abroad."},
    "d70": {"text": "Packing cubes keep luggage organized and make unpacking at hotels faster."},
    "d71": {"text": "Shoulder-season travel often means fewer crowds and lower hotel rates."},
    "d72": {"text": "A local SIM card is usually cheaper than international roaming charges."},
    "d73": {"text": "Learning a few phrases in the local language eases everyday interactions."},
    "d74": {"text": "Overnight ferries let travelers cover long distances while they sleep."},
    "d75": {"text": "Hostels with private rooms offer a budget alternative to hotels."},
    "d76": {"text": "Travelers should photocopy passports in case the originals are lost."},
    "d77": {"text": "A metronome helps musicians practice keeping a steady, even tempo."},
    "d78": {"text": "Major and minor scales form the harmonic backbone of most Western music."},
    "d79": {"text": "Analog synthesizers generate sound through voltage-controlled oscillators."},
    "d80": {"text": "A well-mixed track balances vocals, drums, and instruments in the stereo field."},
    "d81": {"text": "Improvisation is central to jazz, where musicians respond to each other in real time."},
    "d82": {"text": "Sheet music notation lets performers reproduce a composer's intended rhythm and pitch."},
    "d83": {"text": "Interval training alternates short bursts of intense effort with recovery periods."},
    "d84": {"text": "A marathon covers just over twenty-six miles from start to finish."},
    "d85": {"text": "Proper warm-ups reduce the risk of muscle strains during exercise."},
    "d86": {"text": "Zone defense assigns players to guard areas of the court rather than individuals."},
    "d87": {"text": "Altitude training increases red blood cell count to boost endurance."},
    "d88": {"text": "Photo finishes use high-speed cameras to settle races decided by inches."},
    "d89": {"text": "Companion planting pairs crops that benefit each other's growth or pest resistance."},
    "d90": {"text": "Mulching retains soil moisture and suppresses weeds around young plants."},
    "d91": {"text": "Composting kitchen scraps returns nutrients to garden soil over time."},
    "d92": {"text": "Deadheading spent flowers encourages plants to keep producing new blooms."},
    "d93": {"text": "Raised beds warm up faster in spring and improve drainage."},
    "d94": {"text": "Crop rotation each season helps prevent soil-borne diseases from building up."},
    "d95": {"text": "The printing press dramatically lowered the cost of reproducing written texts."},
    "d96": {"text": "Ancient trade routes carried silk, spices, and ideas across continents."},
    "d97": {"text": "The Rosetta Stone allowed scholars to finally decipher Egyptian hieroglyphs."},
    "d98": {"text": "Medieval guilds regulated craftsmanship and training within a given trade."},
    "d99": {"text": "The transcontinental railroad cut cross-country travel from months to days."},
    "d100": {"text": "Archaeologists date artifacts using layers of sediment and radiocarbon analysis."},
}

# Queries whose best matches share few or no words with them -- verified
# against the default model, since which paraphrases land is a property of
# the model, not of scrydb. Only meaningful under --semantic; BM25 finds
# almost none of them, which is the point.
EXAMPLE_QUERIES = (
    "how do I keep my code from breaking",       # -> mypy, test suites, type hints
    "what happens when a star dies",             # -> supernovae, neutron stars
    "how do I make my morning cup taste better",  # -> brewing, water temperature
    "keeping plants healthy in the backyard",    # -> mulching, companion planting
)


def build_rows(documents: dict) -> list[dict]:
    return [{"id": doc_id, "text": fields["text"]} for doc_id, fields in documents.items()]


def document_text(document: dict, text_field: str, doc_id: str = "") -> str:
    """Display text for one hit's stored payload.

    A prebuilt index keeps whatever fields its source rows had, so the
    text lives under `text` only by convention (BEIR corpora, say, pair it
    with `docid` and sometimes `title`). When *text_field* is absent, fall
    back to the longest string field that isn't a repeat of the id, rather
    than printing nothing -- or the id back at the caller.
    """
    value = document.get(text_field)
    if value is None:
        candidates = [v for v in document.values() if isinstance(v, str) and v != str(doc_id)]
        value = max(candidates, key=len, default="")
    return " ".join(str(value).split())


def format_score(hit, precision: "str | None", rerank: "str | None") -> str:
    """Score column for one hit: BM25's single score when *precision* is
    None (lexical mode), otherwise the vector score(s).

    The ranking stage that actually ordered a semantic hit is the rerank,
    if there is one. A reranked result still carries the base stage's score
    too (_rerank_vec merges the rerank's fields on top of the base's), so
    both are shown -- watching binary's Hamming ordering get shuffled by a
    float rerank is the whole point of the two-stage setup.
    """
    if precision is None:
        return f"score={hit['score']:.4f}"
    final_field = SCORE_FIELD[rerank or precision][0]
    score = f"{final_field}={hit[final_field]:.4f}"
    base_field = SCORE_FIELD[precision][0] if rerank else None
    if base_field is not None and base_field in hit:
        score = f"{base_field}={hit[base_field]:.4f} >> {score}"
    return score


def print_hits(
    hits,
    text_field: str = "text",
    max_chars: int = 0,
    precision: "str | None" = None,
    rerank: "str | None" = None,
) -> None:
    if not hits:
        print("  (no matches)")
        return
    width = max(len(str(hit["id"])) for hit in hits)
    for rank, hit in enumerate(hits, start=1):
        text = document_text(hit["document"], text_field, hit["id"])
        if max_chars and len(text) > max_chars:
            text = text[:max_chars].rstrip() + "..."
        print(f"  {rank}. [{str(hit['id']):>{width}}] {format_score(hit, precision, rerank)}  {text}")
    if precision is not None:
        print(f"  ({SCORE_FIELD[rerank or precision][1]})")


def print_help(semantic: bool) -> None:
    lines = ["", "  <text>                         search the corpus"]
    if semantic:
        lines += [
            "  :precision binary|int8|float   vector representation to rank with",
            "  :rerank off|binary|int8|float  second-stage rerank of the candidates",
            "  :k <n>                         results to show (= rerank shortlist size)",
        ]
    else:
        lines.append("  :k <n>                         results to show")
    lines += [
        "  :help                          this message",
        "  :quit                          leave (blank line or Ctrl-D also works)",
        "",
    ]
    print("\n".join(lines))


def open_index(path: "str | None", model: "SentenceEmbedding | None" = None) -> Index:
    """Open the prebuilt index at *path*, or build the toy one in memory.

    `Index.open` creates the database if it doesn't exist (it mirrors
    `sqlite3.connect`), which would silently hand back an empty index on a
    typo'd path -- so an explicit *path* is checked first.

    *model* (semantic mode only) is registered either way: on a prebuilt
    index it encodes the queries typed at the prompt, and on the toy corpus
    it encodes the documents below as well.
    """
    if path is not None:
        db_path = Path(path)
        if not db_path.is_file():
            raise SystemExit(f"No such index: {db_path}")
        index = Index.open(db_path)
        return index.add_model(model) if model is not None else index

    index = Index.open(":memory:")
    if model is not None:
        # The model is what makes an interactive semantic prompt possible:
        # it encodes the documents here, and every query typed later.
        index.add_model(model)
    # store_int8_embeddings=True is off by default; the binary and
    # full-precision copies are stored anyway, so opting in here is what
    # lets :precision/:rerank reach all three representations of the same
    # vectors without reindexing. Without a model there are no vectors at
    # all and the flag is moot -- only the FTS5 index gets built.
    index.index_documents(build_rows(TOY_DOCUMENTS), store_int8_embeddings=model is not None)
    return index


def available_precisions(index: Index) -> list[str]:
    """Precisions this index actually stores document vectors at.

    The toy corpus is indexed at all three, but a prebuilt index only has
    what it was built with (int8, in particular, is opt-in), and asking for
    a missing one just returns nothing -- so the prompt offers only these.
    """
    tables = {
        "binary": index.document_embeddings_binary,
        "int8": index.document_embeddings_int8,
        "float": index.document_embeddings,
    }
    return [precision for precision in SCORE_FIELD if len(tables[precision])]


def repl(
    index: Index,
    top_k: int,
    text_field: str,
    max_chars: int,
    semantic: bool = False,
    precision: "str | None" = None,
    rerank: "str | None" = None,
    precisions: "list[str] | None" = None,
) -> None:
    precisions = precisions or []
    while True:
        label = f"semantic[{precision}{'>>' + rerank if rerank else ''}]" if semantic else "lexical"
        try:
            line = input(f"{label}> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            break

        if line.startswith(":"):
            command, _, value = line[1:].partition(" ")
            value = value.strip()
            if command in ("quit", "q", "exit"):
                break
            elif command in ("help", "h", "?"):
                print_help(semantic)
            elif command == "k":
                try:
                    top_k = max(1, int(value))
                except ValueError:
                    print("  expected: :k <n>")
            elif command == "precision" and semantic:
                if value not in precisions:
                    print(f"  expected one of: {', '.join(precisions)}")
                elif value == rerank:
                    # scrydb rejects a rerank that repeats the base
                    # precision (it would be a no-op); drop it instead
                    # of letting the next search raise.
                    precision, rerank = value, None
                    print(f"  precision={precision}, rerank dropped (would have been a no-op)")
                else:
                    precision = value
            elif command == "rerank" and semantic:
                if value in ("off", "none", "false"):
                    rerank = None
                elif value not in precisions:
                    print(f"  expected: off, {', '.join(precisions)}")
                elif value == precision:
                    print(f"  results are already {precision}-ranked; pick a different precision")
                else:
                    rerank = value
            else:
                print(f"  unknown command {command!r}; try :help")
            continue

        if semantic:
            # mode="semantic" ranks by vector search alone -- the FTS5 index
            # built alongside it is never consulted here. The query text is
            # encoded on the fly by the attached model, since an ad-hoc
            # query has no stored embedding to reuse.
            #
            # With a rerank set, the base stage retrieves top_k candidates
            # and the rerank reorders exactly those, so a rerank can only
            # fix the *order* of what the first stage already found -- raise
            # :k to give the second stage more to work with.
            try:
                hits = index.search(line, mode="semantic", precision=precision, rerank=rerank or False, top_k=top_k)
            except sqlite3.OperationalError as exc:
                # A prebuilt index's vectors come from whatever model built
                # it; encoding queries with a different one lands in another
                # space -- and usually another dimension, which sqlite-vec
                # rejects outright. Every later query would fail the same
                # way, so bail out instead of looping on it.
                raise SystemExit(f"{exc}\nDoes --model match the model this index was built with?")
        else:
            # mode="lexical" is the default: BM25 over the FTS5 index, no
            # embeddings involved. Free-form input is safe here -- raw=False
            # (the default) sanitizes the text and OR-joins its terms, so
            # stray punctuation from an interactive prompt can't raise an
            # FTS5 syntax error.
            hits = index.search(line, mode="lexical", top_k=top_k)
        print_hits(
            hits,
            text_field=text_field,
            max_chars=max_chars,
            precision=precision if semantic else None,
            rerank=rerank,
        )


def parse_args(argv: "list[str] | None" = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--semantic",
        action="store_true",
        help="Search with dense vectors instead of the default BM25/FTS5 lexical search. "
             'Needs a real embedding model: pip install "scrydb[dense]"',
    )
    parser.add_argument(
        "--index",
        metavar="PATH",
        help="Path to a prebuilt scrydb index (.db) to query. "
             "Omit to index and query the built-in toy corpus instead.",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Results per query (default: %(default)s)")
    parser.add_argument(
        "--text-field", default="text",
        help="Payload field to display per hit, for indices whose documents "
             "store their text under another name (default: %(default)s)",
    )
    parser.add_argument(
        "--max-chars", type=int, default=240,
        help="Truncate displayed text to this many characters; 0 disables "
             "truncation (default: %(default)s)",
    )

    dense = parser.add_argument_group("semantic mode (requires --semantic)")
    dense.add_argument("--model", default=DEFAULT_MODEL, help="sentence-transformers model (default: %(default)s)")
    dense.add_argument(
        "--query-prompt",
        default=DEFAULT_QUERY_PROMPT,
        help="instruction template wrapped around each query before encoding (default: %(default)r)",
    )
    dense.add_argument("--precision", default=DEFAULT_PRECISION, choices=tuple(SCORE_FIELD))
    dense.add_argument("--rerank", default="off", choices=("off", *SCORE_FIELD))

    args = parser.parse_args(argv)
    # These only mean anything once a model is attached. Ignoring them in
    # lexical mode would look like they had taken effect, so say so instead.
    if not args.semantic:
        misplaced = [
            f"--{dest.replace('_', '-')}"
            for dest in ("model", "query_prompt", "precision", "rerank")
            if getattr(args, dest) != parser.get_default(dest)
        ]
        if misplaced:
            verb = "require" if len(misplaced) > 1 else "requires"
            parser.error(f"{', '.join(misplaced)} {verb} --semantic")
    return args


def main(argv: "list[str] | None" = None) -> int:
    args = parse_args(argv)
    source = args.index or "the toy corpus"

    model = None
    if args.semantic:
        # SentenceEmbedding imports sentence-transformers lazily, so without
        # this check the missing dependency would only surface after indexing
        # has already started.
        if importlib.util.find_spec("sentence_transformers") is None:
            print(
                'repl: --semantic needs a real embedding model — pip install "scrydb[dense]"',
                file=sys.stderr,
            )
            return 1
        model = SentenceEmbedding(args.model, truncate_dim=None, query_prompt=args.query_prompt)
        if args.index is None:
            print(f"Loading {args.model} and embedding {len(TOY_DOCUMENTS)} toy documents...")

    with open_index(args.index, model) as index:
        n_documents = len(index.documents)
        if n_documents == 0:
            raise SystemExit(f"{source} holds no documents -- nothing to search.")

        precision = rerank = None
        precisions = []
        if args.semantic:
            precisions = available_precisions(index)
            if not precisions:
                raise SystemExit(
                    f"{source} holds no document embeddings -- semantic search needs an index "
                    "built with a model (see examples/index.py)."
                )
            precision = args.precision
            rerank = None if args.rerank == "off" else args.rerank
            for name, value in (("--precision", precision), ("--rerank", rerank)):
                if value is not None and value not in precisions:
                    raise SystemExit(
                        f"{source} stores no {value}-precision embeddings "
                        f"({name}); available: {', '.join(precisions)}."
                    )
            if rerank == precision:
                # scrydb rejects a rerank that repeats the base precision
                # (it would be a no-op), and it would only surface on the
                # first query -- so catch the combination here.
                raise SystemExit(
                    f"--rerank {rerank} is a no-op on {precision}-ranked results; "
                    "pick a different precision."
                )
            print(f"Searching {n_documents} documents from {source} with semantic (vector) search.")
            print(f"Embeddings available at {', '.join(precisions)} precision.")
            if args.index is None:
                print("Try a query that shares little or no vocabulary with what it should find, e.g.:")
                for example in EXAMPLE_QUERIES:
                    print(f"  {example}")
        else:
            print(f"Searching {n_documents} documents from {source} with lexical (BM25) search.")

        print("Type a query and press Enter to search. :help for commands, blank line or Ctrl-D to quit.\n")

        repl(
            index,
            top_k=args.top_k,
            text_field=args.text_field,
            max_chars=args.max_chars,
            semantic=args.semantic,
            precision=precision,
            rerank=rerank,
            precisions=precisions,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
