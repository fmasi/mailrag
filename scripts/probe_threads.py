# scripts/probe_threads.py
"""Manual probe: print assembled threads for a few queries against a live collection.

Run (rag env, with QDRANT_URL set):
  QDRANT_URL=http://localhost:6333 conda run -n rag --no-capture-output \
    python scripts/probe_threads.py --collection work-rag --query "when did we agree to meet"
"""

import argparse

from src.query.hybrid import build_hybrid_searcher


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", default="work-rag")
    ap.add_argument("--query", required=True)
    ap.add_argument("--rerank", action="store_true")
    args = ap.parse_args()

    searcher = build_hybrid_searcher(args.collection, mode="hybrid", rerank=args.rerank)
    contexts = searcher.search_threads(args.query)
    print(f"=== {len(contexts)} thread(s) for: {args.query!r} ===")
    for i, ctx in enumerate(contexts, 1):
        print(f"\n--- thread {i} ({len(ctx.emails)} emails) ---")
        print(ctx.text)


if __name__ == "__main__":
    main()
