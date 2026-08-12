"""Generate the committed public benchmark fixtures for `make bench` (issue #97).

Run this ONCE to (re)generate the fixtures under ``eval/public/``; the benchmark
itself (``bench_public.py``) never calls it. The fixtures are committed so the
public number is scored against a *fixed, reviewable* query set rather than
whatever a re-shuffle happens to pick.

Two files per size:

- ``enron_qa_<size>_corpus.txt``  — one Enron-QA ``path`` per line; the corpus.
- ``enron_qa_<size>_queries.jsonl`` — ``{query, answer_path, category}`` per line.

**Why a path manifest rather than "same seed, same shuffle":** the corpus must
be reproducible from the upstream dataset, and a seeded shuffle only reproduces
if upstream row *order* never changes — which nothing guarantees. Selecting rows
by an explicit committed path list is stable under any reordering, and it makes
"is the gold document actually in the corpus?" a property you can check by
reading the files instead of trusting a PRNG.

Selection is made order-independent here too: rows are sorted by path before the
seeded shuffle, so re-running this against a re-ordered upstream dataset
reproduces the same fixtures.

Usage:
    python -m scripts.eval.gen_public_benchset            # both sizes
    python -m scripts.eval.gen_public_benchset --size small
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random

# corpus size, query count.
#
# Both tiers are deliberately large enough to DISCRIMINATE. An earlier draft used
# a 500-document corpus for speed and it sat at the ceiling — dense 96.7 / hybrid
# 95.3 at R@1, i.e. dense "winning" purely on noise. Retrieval difficulty is
# driven by the size of the distractor pool, so a corpus small enough to be
# trivially fast is also small enough to measure nothing.
#
# Query count is nearly free (scoring 360 queries across both arms costs ~75 s
# against ~100 s to build 2 000 documents), so queries are spent generously to
# tighten the confidence interval rather than rationed for speed.
SIZES = {"standard": (2000, 360), "large": (10000, 360)}

DATASET = "MichaelR207/enron_qa_0922"
SPLIT = "test"
SEED = 13
OUT_DIR = pathlib.Path(__file__).resolve().parents[2] / "eval" / "public"


def usable_rows(rows):
    """Rows carrying both a question and enough body text to be retrievable.

    The length floor drops stubs (headers with an empty body) that would be
    unretrievable by any system and would depress every arm equally — noise in
    the metric, not signal about retrieval.
    """
    return [r for r in rows if r.get("questions") and len(r.get("email") or "") > 120]


def select(rows, n_corpus, n_queries):
    """Deterministically pick the corpus and the queries scored against it.

    Sorted by path *before* shuffling so the result depends only on the set of
    rows, never on the order the loader yields them in.
    """
    pool = sorted(usable_rows(rows), key=lambda r: r["path"])
    random.Random(SEED).shuffle(pool)
    corpus = pool[:n_corpus]
    queries = [
        {
            "query": r["questions"][0],
            # The gold document's id. `path` is Enron-QA's own per-user mailbox
            # path and is what we set as message_id at index time, so this is
            # what a hit is compared against.
            "answer_path": r["path"],
            "category": "enron-qa",
        }
        for r in corpus[:n_queries]
    ]
    return corpus, queries


def write(size: str, rows) -> None:
    n_corpus, n_queries = SIZES[size]
    corpus, queries = select(rows, n_corpus, n_queries)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cpath = OUT_DIR / f"enron_qa_{size}_corpus.txt"
    cpath.write_text("".join(f"{r['path']}\n" for r in corpus), encoding="utf-8")

    qpath = OUT_DIR / f"enron_qa_{size}_queries.jsonl"
    with qpath.open("w", encoding="utf-8") as fh:
        for q in queries:
            fh.write(json.dumps(q, ensure_ascii=False) + "\n")

    gold = {q["answer_path"] for q in queries}
    missing = gold - {r["path"] for r in corpus}
    assert not missing, f"{len(missing)} gold path(s) absent from the corpus"
    print(f"{size}: {len(corpus)} corpus paths -> {cpath.name}")
    print(f"{size}: {len(queries)} queries -> {qpath.name}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--size", choices=[*SIZES, "both"], default="both")
    args = ap.parse_args()

    from datasets import load_dataset  # noqa: PLC0415 — heavy, and only needed here

    print(f"loading {DATASET} [{SPLIT}] ...", flush=True)
    rows = load_dataset(DATASET, split=SPLIT).to_list()
    print(f"  {len(rows)} rows", flush=True)

    for size in SIZES if args.size == "both" else [args.size]:
        write(size, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
