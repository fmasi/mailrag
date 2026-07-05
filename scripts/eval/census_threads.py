# scripts/eval/census_threads.py
"""Corpus-wide thread-size + assembled-token census for the bounding decision.

Scrolls every point in a collection, groups by thread_id, reconstructs emails per
thread (reusing src.query.thread_expand), estimates assembled-block tokens, and
prints percentiles. Aggregate numbers only — no corpus content is printed.

Run on the HOST (rag env, QDRANT_URL set):
  QDRANT_URL=http://localhost:6333 conda run -n rag --no-capture-output \
    python scripts/eval/census_threads.py --collection work-rag \
    --out eval/out/census.json | tee eval/out/census.log
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.query.hybrid import _qdrant_client
from src.query.thread_expand import estimate_tokens, group_into_emails, render_thread

_PAGE = 512


def _percentile(sorted_vals, q):
    if not sorted_vals:
        return 0
    idx = min(len(sorted_vals) - 1, int(q * (len(sorted_vals) - 1)))
    return sorted_vals[idx]


def run(collection, out_path):
    client = _qdrant_client()
    by_tid = {}
    offset = None
    total_points = 0
    while True:
        points, offset = client.scroll(
            collection_name=collection,
            limit=_PAGE,
            with_payload=True,
            with_vectors=False,
            offset=offset,
        )
        for p in points:
            total_points += 1
            tid = (p.payload or {}).get("thread_id")
            if tid:
                by_tid.setdefault(tid, []).append(p.payload)
        if offset is None:
            break

    sizes, tokens = [], []
    for tid, payloads in by_tid.items():
        emails = group_into_emails(payloads)
        sizes.append(len(emails))
        tokens.append(estimate_tokens(render_thread(tid, emails)))
    sizes.sort()
    tokens.sort()
    n = len(sizes)
    over = {b: sum(1 for t in tokens if t > b) for b in (4000, 8000, 16000)}
    report = {
        "collection": collection,
        "total_points": total_points,
        "total_threads": n,
        "singletons": sum(1 for s in sizes if s == 1),
        "singleton_pct": round(100 * sum(1 for s in sizes if s == 1) / n, 1) if n else 0,
        "emails_per_thread": {
            "mean": round(sum(sizes) / n, 2) if n else 0,
            "p50": _percentile(sizes, 0.50),
            "p90": _percentile(sizes, 0.90),
            "p99": _percentile(sizes, 0.99),
            "max": sizes[-1] if sizes else 0,
        },
        "assembled_tokens": {
            "p50": _percentile(tokens, 0.50),
            "p90": _percentile(tokens, 0.90),
            "p99": _percentile(tokens, 0.99),
            "max": tokens[-1] if tokens else 0,
        },
        "threads_over_budget": over,
        "pct_over_budget": {str(b): round(100 * c / n, 2) if n else 0 for b, c in over.items()},
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps(report, indent=2), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", default="work-rag")
    ap.add_argument("--out", default="eval/out/census.json")
    args = ap.parse_args()
    run(args.collection, args.out)


if __name__ == "__main__":
    main()
