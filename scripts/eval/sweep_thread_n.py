# scripts/eval/sweep_thread_n.py
"""Sweep how many top seed-threads to expand (N) vs answer-coverage + context size.

For each query, expand the search hits into threads (in rank order) and ask, for
N in {1,2,3,5,10,all}: is the KNOWN answer email inside the first-N expanded
threads? Coverage uses the known answer_message_id only — NO LLM judging. Prints
aggregate coverage + avg email count per N (no corpus content).

Run on the HOST (rag env; QDRANT_URL set):
  QDRANT_URL=http://localhost:6333 conda run -n rag --no-capture-output \
    python scripts/eval/sweep_thread_n.py --queries eval/out/queries.jsonl \
    | tee eval/out/sweep_thread_n.log
"""
import argparse
import json
import os
import statistics
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.ingest.embedder import BgeM3Embedder
from src.query.hybrid import build_hybrid_searcher

C, CP = "work-rag", "work-rag-ctx"
NS = [1, 2, 3, 5, 10]  # plus "all"


def run(queries_path, top_k):
    print("loading bge-m3 (silent ~1 min)...", flush=True)
    embedder = BgeM3Embedder()

    def mk(collection):
        return build_hybrid_searcher(
            collection, embedder=embedder, mode="hybrid", rerank=False,
            dense_top_k=max(top_k, 20), sparse_top_k=max(top_k, 20), top_n=top_k)

    searchers = {"C+thread": mk(C), "Cprime+thread": mk(CP)}

    with open(queries_path) as fh:
        queries = [json.loads(l) for l in fh if l.strip()]

    # cov[base][N or 'all'][cat] -> list of 0/1 ; size[base][N] -> list of email counts
    cov = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    size = defaultdict(lambda: defaultdict(list))

    for q in queries:
        query, cat, ans = q["query"], q["category"], q["answer_message_id"]
        for base, s in searchers.items():
            ctxs = s.search_threads(query)  # ThreadContexts in seed-rank order
            for N in NS + ["all"]:
                sel = ctxs if N == "all" else ctxs[:N]
                mids = {e.message_id for c in sel for e in c.emails}
                hit = 1 if ans in mids else 0
                cov[base][N]["all"].append(hit)
                cov[base][N][cat].append(hit)
                size[base][N].append(len(mids))
        print(f"  done: {query[:50]!r}", flush=True)

    def mean(xs):
        return statistics.mean(xs) if xs else 0.0

    for base in searchers:
        print(f"\n=== {base} — answer-coverage by N (top-N seed threads expanded) ===")
        print(f'{"N":>4} {"cov(all)":>9} {"terse":>7} {"content":>8} {"spanning":>9} {"avg#emails":>11}')
        for N in NS + ["all"]:
            c = cov[base][N]
            print(f'{str(N):>4} {mean(c["all"]):>9.0%} {mean(c["terse"]):>7.0%} '
                  f'{mean(c["content"]):>8.0%} {mean(c["spanning"]):>9.0%} '
                  f'{mean(size[base][N]):>11.1f}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", default="eval/out/queries.jsonl")
    ap.add_argument("--top-k", type=int, default=10)
    args = ap.parse_args()
    run(args.queries, args.top_k)


if __name__ == "__main__":
    main()
