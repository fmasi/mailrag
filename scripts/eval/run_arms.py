# scripts/eval/run_arms.py
"""Run the 5 eval arms per query; flatten-to-emails; build judge pool; record
retrieved-thread token sizes. All outputs (real content) -> eval/out (gitignored).

Arms:
  C                      collection=work-rag,     mode=hybrid, rerank=False
  C+rerank               collection=work-rag,     mode=hybrid, rerank=True
  C+rerank+thread        collection=work-rag,     search_threads(), rerank=True
  Cprime                 collection=work-rag-ctx, mode=hybrid, rerank=False
  Cprime+rerank          collection=work-rag-ctx, mode=hybrid, rerank=True

Run on the HOST (rag env; QDRANT_URL set):
  QDRANT_URL=http://localhost:6333 conda run -n rag --no-capture-output \
    python scripts/eval/run_arms.py --queries eval/out/queries.jsonl \
    --top-k 10 --outdir eval/out | tee eval/out/run_arms.log
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.ingest.embedder import BgeM3Embedder
from src.query.hybrid import build_hybrid_searcher
from src.query.thread_expand import estimate_tokens
from src.eval.flatten import flatten_nodes, flatten_threads
from src.eval.pooling import build_pool

C, CP = "work-rag", "work-rag-ctx"


def _hit_dict(h):
    return {"message_id": h.message_id, "subject": h.subject,
            "body": h.body, "summary": h.summary}


def run(queries_path, top_k, outdir):
    print("loading bge-m3 (silent ~1 min)...", flush=True)
    embedder = BgeM3Embedder()

    def mk(collection, rerank):
        return build_hybrid_searcher(
            collection, embedder=embedder, mode="hybrid", rerank=rerank,
            dense_top_k=max(top_k, 20), sparse_top_k=max(top_k, 20), top_n=top_k)

    s_c = mk(C, False)
    s_c_rr = mk(C, True)
    s_cp = mk(CP, False)
    s_cp_rr = mk(CP, True)

    with open(queries_path) as fh:
        queries = [json.loads(l) for l in fh if l.strip()]

    arm_rankings = []   # per query: {query meta, arms: {arm: [message_id,...]}}
    pool_rows = []      # per query: {query, pool: [hit_dict,...]}
    bounding = []       # per query: {query, threads:[{n_emails, tokens},...]}

    for q in queries:
        query = q["query"]
        arms = {
            "C": flatten_nodes(s_c.search(query)[:top_k]),
            "C+rerank": flatten_nodes(s_c_rr.search(query)[:top_k]),
            "C+rerank+thread": flatten_threads(s_c_rr.search_threads(query)),
            "Cprime": flatten_nodes(s_cp.search(query)[:top_k]),
            "Cprime+rerank": flatten_nodes(s_cp_rr.search(query)[:top_k]),
        }
        pool = build_pool(arms)
        arm_rankings.append({
            "query": query, "category": q["category"],
            "answer_message_id": q["answer_message_id"],
            "arms": {a: [h.message_id for h in hits] for a, hits in arms.items()},
        })
        pool_rows.append({"query": query, "pool": [_hit_dict(h) for h in pool]})
        ctxs = s_c_rr.search_threads(query)
        bounding.append({"query": query,
                         "threads": [{"n_emails": len(c.emails),
                                      "tokens": estimate_tokens(c.text)} for c in ctxs]})
        print(f"  done: {query[:60]!r} (pool={len(pool)})", flush=True)

    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "arm_rankings.json"), "w") as fh:
        json.dump(arm_rankings, fh, indent=2)
    with open(os.path.join(outdir, "pool.jsonl"), "w") as fh:
        for r in pool_rows:
            fh.write(json.dumps(r) + "\n")
    with open(os.path.join(outdir, "bounding_retrieved.json"), "w") as fh:
        json.dump(bounding, fh, indent=2)

    sizes = [t["tokens"] for b in bounding for t in b["threads"]]
    over = {b: sum(1 for s in sizes if s > b) for b in (4000, 8000, 16000)}
    print(f"\nretrieved threads={len(sizes)} | over-budget {over} "
          f"(pct { {b: round(100*c/len(sizes),1) if sizes else 0 for b,c in over.items()} })",
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", default="eval/out/queries.jsonl")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--outdir", default="eval/out")
    args = ap.parse_args()
    run(args.queries, args.top_k, args.outdir)


if __name__ == "__main__":
    main()
