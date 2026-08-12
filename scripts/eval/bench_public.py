"""The public, reproducible retrieval benchmark — `make bench` (issue #97).

Builds a Qdrant collection from a fixed slice of the **public** Enron-QA dataset
and reports recall@k for two locally-runnable arms:

- ``dense``        — bge-m3 dense vectors only.
- ``dense+sparse`` — bge-m3 dense + learned-sparse, fused with RRF (the default
  retrieval path).

Anyone can run this and get the same number. That is the whole point: it turns
the project's retrieval claim from "trust me" into "check me".

**What this benchmark deliberately does not measure.** Enron-QA rows carry no
conversation linkage (the schema is ``email / questions / path / user / ...``;
``path`` is a per-user mailbox path), so thread reconstruction — a real part of
the system — cannot be scored here and is not reported. Nor is the cross-encoder
rerank arm: it needs a paid NVIDIA endpoint, which would make the number
unreproducible for a reader without a key. Both are measured in the private
harness; this file only reports what a stranger can regenerate.

Usage:
    make bench                 # standard set (2000 docs / 360 queries, ~3 min)
    make bench SIZE=large      # large set   (10000 docs / 360 queries, harder)
    python -m scripts.eval.bench_public --size standard --skip-build
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib

FIXTURES = pathlib.Path(__file__).resolve().parents[2] / "eval" / "public"
DATASET = "MichaelR207/enron_qa_0922"
SPLIT = "test"
ARMS = ("dense", "dense+sparse")
KS = (1, 5, 10)
TOP_K = 10


def load_fixtures(size: str):
    """The committed corpus manifest and query set for *size*."""
    corpus_file = FIXTURES / f"enron_qa_{size}_corpus.txt"
    query_file = FIXTURES / f"enron_qa_{size}_queries.jsonl"
    if not corpus_file.exists() or not query_file.exists():
        raise SystemExit(
            f"missing fixtures for size={size!r} ({corpus_file.name}, {query_file.name}).\n"
            "Regenerate with: python -m scripts.eval.gen_public_benchset"
        )
    paths = [
        ln.strip() for ln in corpus_file.read_text(encoding="utf-8").splitlines() if ln.strip()
    ]
    queries = [
        json.loads(ln) for ln in query_file.read_text(encoding="utf-8").splitlines() if ln.strip()
    ]
    return paths, queries


def to_email(row):
    """One Enron-QA row -> a NormalizedEmail.

    The row's ``email`` field is a header block followed by the body; Enron-QA
    separates them with a ``=====`` rule when it has one. ``path`` becomes the
    message_id because it is what the committed query set names as gold.
    """
    from src.data.models import NormalizedEmail  # noqa: PLC0415

    txt = row["email"] or ""
    subject, sender = "", "someone@enron.com"
    for line in txt.split("\n")[:8]:
        if line.startswith("Subject:"):
            subject = line[8:].strip()
        elif line.startswith("Sender:"):
            sender = line[7:].strip() or sender
    body = txt
    if "=====" in txt:
        tail = txt.split("=====", 1)[1].strip()
        if len(tail) > 40:
            body = tail
    return NormalizedEmail(
        sender=sender,
        subject=subject,
        date=None,
        body=body,
        source="enron-qa",
        source_id=row["path"],
        recipients="recipients@enron.com",
        cc=None,
        message_id=row["path"],
    )


def fetch_corpus(paths):
    """Pull the manifest's rows from the public dataset, in manifest order.

    Selecting by explicit path (rather than re-deriving a seeded shuffle) is what
    makes the corpus stable if upstream ever reorders rows.
    """
    from datasets import load_dataset  # noqa: PLC0415

    wanted = set(paths)
    rows = {}
    for row in load_dataset(DATASET, split=SPLIT):
        if row["path"] in wanted:
            rows[row["path"]] = row
            if len(rows) == len(wanted):
                break
    missing = wanted - rows.keys()
    if missing:
        raise SystemExit(
            f"{len(missing)} manifest path(s) not found in {DATASET}[{SPLIT}] — "
            "the upstream dataset has changed; regenerate the fixtures."
        )
    return [rows[p] for p in paths]


def recall_at_k(ranks, k, n):
    """Fraction of queries whose gold document ranked inside the top *k*."""
    if not n:
        return 0.0
    return 100.0 * sum(1 for r in ranks if r is not None and r < k) / n


Z = 1.96  # 95%


def wilson_interval(p_pct, n):
    """The 95% Wilson score interval for a proportion, as (lo, hi) in percent.

    Wilson rather than the textbook Wald interval (``z*sqrt(p(1-p)/n)``), and
    bounds rather than a single ``±``, because this benchmark's recall values sit
    at 0.76-0.99 — exactly where Wald misbehaves. At p=0.975, n=360 the true
    interval is [95.3, 98.7]: 1.0 pp asymmetric, so a symmetric ± overstates the
    upside and understates the downside. At p=1.0 Wald degenerates to ±0.00 and
    would claim perfect certainty from 360 samples; Wilson gives [98.9, 100].
    """
    if not n:
        return (0.0, 0.0)
    p = max(0.0, min(1.0, p_pct / 100.0))
    denom = 1 + Z * Z / n
    centre = (p + Z * Z / (2 * n)) / denom
    half = (Z / denom) * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n))
    return (100.0 * max(centre - half, 0.0), 100.0 * min(centre + half, 1.0))


def mcnemar_exact(ranks_a, ranks_b, k):
    """Exact McNemar test that arm B beats arm A at @k. Returns (b, c, p_value).

    The arms are scored on the *same* queries, so the comparison is paired and
    the marginal confidence intervals — which overlap here — are the wrong test:
    they discard the pairing and are badly conservative. McNemar looks only at
    the queries the arms disagree on, where ``b`` = A hit and B missed, ``c`` =
    B hit and A missed. Under the null those split 50/50, so the p-value is an
    exact two-sided binomial rather than a chi-square approximation (the
    discordant counts here are small enough that the approximation is unsafe).
    """
    b = c = 0
    for ra, rb in zip(ranks_a, ranks_b):
        hit_a = ra is not None and ra < k
        hit_b = rb is not None and rb < k
        if hit_a and not hit_b:
            b += 1
        elif hit_b and not hit_a:
            c += 1
    n = b + c
    if not n:
        return (0, 0, 1.0)
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / (2**n)
    return (b, c, min(1.0, 2 * tail))


def gold_rank(nodes, gold_path):
    """Position of the gold document among the hits, de-duplicated by message.

    Chunk-level hits are collapsed to their message first: several chunks of the
    same email are one retrieved document, and counting them separately would
    inflate recall@k for every arm.
    """
    seen = []
    for node in nodes:
        mid = node.metadata.get("message_id")
        if mid not in seen:
            seen.append(mid)
        if mid == gold_path:
            return len(seen) - 1
    return None


def build(paths, collection, embedder):
    from src.indexing.contextual_index import build_contextual_index  # noqa: PLC0415

    rows = fetch_corpus(paths)
    emails = [to_email(r) for r in rows]
    print(f"  {len(emails)} emails -> collection {collection!r}", flush=True)
    res = build_contextual_index(
        emails,
        collection=collection,
        embedder=embedder,
        embed_summary=False,
        recreate=True,
        # The corpus IS the benchmark; dropping "noise" would silently change
        # what is being scored and make the number non-comparable.
        apply_noise_filter=False,
        qdrant_url=os.environ.get("QDRANT_URL", "http://localhost:6333"),
    )
    print(f"  indexed {res.chunks} chunks from {res.kept_emails} emails", flush=True)


def score(queries, collection, embedder):
    from src.query.hybrid import build_hybrid_searcher  # noqa: PLC0415

    results = {}
    for arm in ARMS:
        mode = "dense" if arm == "dense" else "hybrid"
        searcher = build_hybrid_searcher(
            collection,
            embedder=embedder,
            mode=mode,
            dense_top_k=TOP_K,
            sparse_top_k=TOP_K,
            qdrant_url=os.environ.get("QDRANT_URL", "http://localhost:6333"),
        )
        ranks = [gold_rank(searcher.search(q["query"]), q["answer_path"]) for q in queries]
        results[arm] = ranks
        print(f"  scored {arm}", flush=True)
    return results


def report(results, n, size):
    print(f"\n  Enron-QA ({size}) — public retrieval benchmark, n={n} queries")
    print(f"  {'arm':14s}" + "".join(f"{'R@' + str(k):>22s}" for k in KS))
    for arm in ARMS:
        row = f"  {arm:14s}"
        for k in KS:
            val = recall_at_k(results[arm], k, n)
            lo, hi = wilson_interval(val, n)
            row += f"{val:8.1f} [{lo:4.1f},{hi:5.1f}]"
        print(row)
    print("\n  Brackets are 95% Wilson score intervals. They are asymmetric near the")
    print("  ceiling, which is why bounds are printed rather than a single ±.")

    # The marginal intervals above overlap; the paired test is what settles it.
    for k in KS:
        b, c, p = mcnemar_exact(results[ARMS[0]], results[ARMS[1]], k)
        verdict = "significant" if p < 0.05 else "not significant"
        print(
            f"  paired @{k}: sparse fixes {c}, breaks {b} "
            f"-> McNemar exact p={p:.4f} ({verdict} at 0.05)"
        )
    print("\n  Arms are dense-only vs dense+learned-sparse (RRF). No rerank, no")
    print("  thread reconstruction — see the module docstring for why.\n")


def main() -> int:
    ap = argparse.ArgumentParser(description="Public Enron-QA retrieval benchmark")
    ap.add_argument("--size", choices=("standard", "large"), default="standard")
    ap.add_argument(
        "--skip-build", action="store_true", help="reuse the existing collection (re-score only)"
    )
    args = ap.parse_args()

    paths, queries = load_fixtures(args.size)
    collection = f"enron-qa-public-{args.size}"
    print(f"benchmark: {len(paths)} docs, {len(queries)} queries, collection {collection!r}")

    from src.ingest.embedder import BgeM3Embedder  # noqa: PLC0415

    embedder = BgeM3Embedder()
    if not args.skip_build:
        build(paths, collection, embedder)
    report(score(queries, collection, embedder), len(queries), args.size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
