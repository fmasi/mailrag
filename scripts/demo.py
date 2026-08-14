"""`make demo` — show what contextual embedding buys, on public data, with no API key.

Builds TWO indexes over the same 1,200 public Enron emails:

    plain    — each email embedded as-is
    context  — each email embedded together with a summary of what came before it
               in its conversation

then asks both the same questions. The point is not that mailrag returns more
text; returning a whole thread once you have a hit is parent-document retrieval
and plenty of systems do it. The point is **findability**: a terse reply is a bag
of common words that no query will surface, until its vector carries the context
its own text omits.

Everything needed is committed under ``eval/demo/`` — corpus, pre-computed
summaries, and validated questions — so this runs offline and spends no LLM
tokens. Regenerating the fixtures is documented in ``docs/BENCHMARK.md``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import re
import sys
import textwrap

REPO = pathlib.Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

FIXTURES = REPO / "eval" / "demo"
PLAIN, CONTEXT = "mailrag-demo-plain", "mailrag-demo-context"
KS = (1, 5, 10)


def _qdrant() -> str:
    return os.environ.get("QDRANT_URL", "http://localhost:6333")


def load_fixtures():
    corpus = [json.loads(x) for x in (FIXTURES / "corpus.jsonl").read_text().splitlines() if x]
    summaries = {
        r["message_id"]: r["summary"]
        for r in (
            json.loads(x) for x in (FIXTURES / "summaries.jsonl").read_text().splitlines() if x
        )
    }
    queries = [json.loads(x) for x in (FIXTURES / "questions.jsonl").read_text().splitlines() if x]
    return corpus, summaries, queries


def to_emails(corpus):
    import datetime as _dt

    from src.data.models import NormalizedEmail  # noqa: PLC0415

    out = []
    for r in corpus:
        d = None
        if r.get("date"):
            try:
                d = _dt.datetime.fromisoformat(r["date"])
            except ValueError:
                d = None
        out.append(
            NormalizedEmail(
                sender=r["sender"],
                subject=r["subject"],
                date=d,
                body=r["body"],
                source="enron-demo",
                source_id=r["message_id"],
                recipients=r.get("recipients") or "",
                cc=None,
                message_id=r["message_id"],
                # conversations were derived from subject + shared participants;
                # the root rides in `references` so compute_thread_id groups them.
                references=r["thread"],
            )
        )
    return out


def build(emails, summaries, embedder, *, collection, with_summaries):
    from src.indexing.contextual_index import build_contextual_index  # noqa: PLC0415

    res = build_contextual_index(
        emails,
        collection=collection,
        embedder=embedder,
        summaries=summaries if with_summaries else None,
        embed_summary=with_summaries,
        recreate=True,
        apply_noise_filter=False,
        qdrant_url=_qdrant(),
    )
    label = "with thread context" if with_summaries else "plain"
    print(f"    {collection:24} {res.chunks:5d} chunks  ({label})", flush=True)


def norm_thread(t):
    """Thread ids lose their angle brackets on the way into the payload
    (`normalize_message_id`), so both sides must be normalised before comparison.
    Comparing one normalised side against one raw side silently yields T@k = 0.0%
    — which happened once during development and is easy to miss because the
    number is merely wrong, not an error."""
    return (t or "").strip().lstrip("<").rstrip(">")


def thread_coverage(retrieved_msgs, gold_msgs, k):
    """Fraction of the answering conversation present in the top-*k* messages.

    An empty gold set means the fixture references a thread absent from the
    corpus — a broken fixture, not a score of zero. Returns None so the caller
    can drop it rather than divide by zero or silently record 0%.
    """
    if not gold_msgs:
        return None
    return len(set(retrieved_msgs[:k]) & set(gold_msgs)) / len(gold_msgs)


def gold_rank(nodes, gold):
    """Position of the gold message among hits, collapsed to distinct messages."""
    seen = []
    for n in nodes:
        mid = n.metadata.get("message_id")
        if mid not in seen:
            seen.append(mid)
        if mid == gold:
            return len(seen) - 1
    return None


def score(queries, collection, embedder):
    from src.query.hybrid import build_hybrid_searcher  # noqa: PLC0415

    s = build_hybrid_searcher(
        collection,
        embedder=embedder,
        mode="hybrid",
        dense_top_k=20,
        sparse_top_k=20,
        qdrant_url=_qdrant(),
    )
    return [gold_rank(s.search(q["query"]), q["gold"]) for q in queries]


def mcnemar(a, b, k):
    """Exact paired test: does arm B beat arm A at @k? Returns (fixes, breaks, p)."""
    fx = sum(1 for x, y in zip(a, b) if not _hit(x, k) and _hit(y, k))
    bk = sum(1 for x, y in zip(a, b) if _hit(x, k) and not _hit(y, k))
    n = fx + bk
    if not n:
        return fx, bk, 1.0
    tail = sum(math.comb(n, i) for i in range(min(fx, bk) + 1)) / 2**n
    return fx, bk, min(1.0, 2 * tail)


def _hit(r, k):
    return r is not None and r < k


def recall(ranks, k):
    return 100.0 * sum(1 for r in ranks if _hit(r, k)) / len(ranks)


def _clean(t, n=88):
    t = re.sub(r"(?m)^\s*>[> ]*", "", t or "")
    t = re.sub(r"-{3,}\s*(Original Message|Forwarded by).*", "", t, flags=re.S)
    return textwrap.shorten(" ".join(t.split()), n, placeholder="…")


def worked_example(queries, ranks_plain, ranks_ctx, corpus, embedder):
    """One question the plain index misses and the context index finds."""
    by_id = {r["message_id"]: r for r in corpus}
    pick = next(
        (i for i, q in enumerate(queries) if not _hit(ranks_plain[i], 5) and _hit(ranks_ctx[i], 1)),
        None,
    )
    if pick is None:
        return
    q = queries[pick]
    gold = by_id.get(q["gold"], {})
    print("\n  ── a question the plain index cannot answer " + "─" * 30)
    print(f'\n  Q: "{q["query"]}"\n')
    print(f'  The message that answers it:  "{_clean(gold.get("body"), 72)}"')
    print("  Embedded on its own, that is a handful of common words — nothing for")
    print("  a query to match on.\n")
    p_rank = ranks_plain[pick]
    print(f"    plain index    → {'not in top 20' if p_rank is None else f'rank {p_rank + 1}'}")
    print(f"    with context   → rank {ranks_ctx[pick] + 1}")
    print("\n  Same corpus, same embedder, same question. The only difference is that")
    print("  the second index embedded each message with what preceded it.")


def thread_section(embedder):
    """The second lever: once you find a message, is the CONVERSATION the answer?

    Scored on questions whose answer genuinely spans several messages — the case
    thread reconstruction exists for, and the one a per-message question set
    cannot show.
    """
    import collections  # noqa: PLC0415

    path = FIXTURES / "questions_spanning.jsonl"
    if not path.exists():
        return
    qs = [json.loads(x) for x in path.read_text().splitlines() if x]
    corpus = [json.loads(x) for x in (FIXTURES / "corpus.jsonl").read_text().splitlines() if x]

    members = collections.defaultdict(set)
    for r in corpus:
        members[norm_thread(r["thread"])].add(r["message_id"])

    from src.query.hybrid import build_hybrid_searcher  # noqa: PLC0415

    s = build_hybrid_searcher(
        CONTEXT,
        embedder=embedder,
        mode="hybrid",
        dense_top_k=20,
        sparse_top_k=20,
        qdrant_url=_qdrant(),
    )
    tranks, coverage, skipped = [], {5: [], 10: []}, 0
    for q in qs:
        gold = norm_thread(q["thread"])
        gold_msgs = members.get(gold, set())
        if not gold_msgs:
            skipped += 1
            continue
        msgs, threads = [], []
        for node in s.search(q["query"]):
            mid = node.metadata.get("message_id")
            tid = norm_thread(node.metadata.get("thread_id"))
            if mid not in msgs:
                msgs.append(mid)
            if tid not in threads:
                threads.append(tid)
        tranks.append(threads.index(gold) if gold in threads else None)
        for k in coverage:
            coverage[k].append(thread_coverage(msgs, gold_msgs, k))

    if skipped:
        print(f"\n  ! {skipped} spanning question(s) reference a thread absent from the corpus")
    if not tranks:
        return
    scored = len(tranks)
    print(f"\n  ── and when the answer spans several messages ({scored} questions) " + "─" * 12)
    print(
        f"\n    the right conversation is found:  T@1 {recall(tranks, 1):.1f}%"
        f"   T@5 {recall(tranks, 5):.1f}%"
    )
    print("\n    how much of that conversation you actually get:")
    for k in (5, 10):
        pct = 100 * sum(coverage[k]) / scored
        print(f"      top-{k:<2} messages   → {pct:5.1f}% of it")
    print("      thread expansion → 100.0% of it, whenever the thread is found")
    print("\n    A generic RAG hands you half the conversation. That is the half")
    print("    the answer is usually missing from.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip-build", action="store_true", help="reuse existing collections")
    args = ap.parse_args()

    corpus, summaries, queries = load_fixtures()
    print(
        f"\n  {len(corpus)} public Enron emails · {len(queries)} validated questions · no API key\n"
    )

    from src.ingest.embedder import BgeM3Embedder  # noqa: PLC0415

    embedder = BgeM3Embedder()
    if not args.skip_build:
        emails = to_emails(corpus)
        print("  building two indexes:", flush=True)
        build(emails, summaries, embedder, collection=PLAIN, with_summaries=False)
        build(emails, summaries, embedder, collection=CONTEXT, with_summaries=True)

    print("\n  scoring…", flush=True)
    rp = score(queries, PLAIN, embedder)
    rc = score(queries, CONTEXT, embedder)

    print("\n  Finding the message that answers the question:\n")
    print(f"    {'index':22}" + "".join(f"{'R@' + str(k):>9}" for k in KS))
    for name, r in (("plain", rp), ("with thread context", rc)):
        print(f"    {name:22}" + "".join(f"{recall(r, k):8.1f}%" for k in KS))

    print()
    for k in KS:
        fx, bk, p = mcnemar(rp, rc, k)
        mark = "significant" if p < 0.05 else "not significant"
        print(
            f"    @{k:<2} context fixes {fx:2d}, breaks {bk:2d}  →  McNemar exact p={p:.4f}  ({mark})"
        )

    worked_example(queries, rp, rc, corpus, embedder)
    thread_section(embedder)

    print("\n  Two levers, on a public corpus with conversations derived from subject +")
    print("  participants: contextual embedding (findability) and thread expansion")
    print("  (completeness). Reranking and noise cleanup are not measured here.")
    print("  `make bench` is the retrieval benchmark; docs/CLAIMS.md tracks every figure.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
