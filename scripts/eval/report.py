# scripts/eval/report.py
"""Compute the per-category metric table across arms + the bounding summary, and
print the two pre-registered decisions. Aggregate numbers only (safe to paste
into docs). Reads arm_rankings.json + grades.json + bounding_retrieved.json.

Run:
  conda run -n rag --no-capture-output \
    python scripts/eval/report.py --outdir eval/out | tee eval/out/report.txt
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.eval.decisions import decide
from src.eval.metrics import aggregate, mrr, ndcg_at_k, precision_at_k, recall_at_k, success_at_k
from src.eval.pooling import total_relevant

ARMS = ["C", "C+rerank", "C+rerank+thread", "Cprime", "Cprime+rerank"]
REL = 2


def _rows_for_arm(rankings, grades, arm):
    rows = []
    for r in rankings:
        q = r["query"]
        ids = r["arms"].get(arm, [])
        g = [grades.get(q, {}).get(mid, 0) for mid in ids]
        tr = total_relevant(grades.get(q, {}), REL)
        rows.append(
            {
                "category": r["category"],
                "ndcg@10": ndcg_at_k(g, 10),
                "ndcg@5": ndcg_at_k(g, 5),
                "p@10": precision_at_k(g, 10, REL),
                "p@5": precision_at_k(g, 5, REL),
                "recall@10": recall_at_k(g, 10, REL, tr),
                "mrr": mrr(g, REL),
                "success@10": success_at_k(ids, r["answer_message_id"], 10),
            }
        )
    return rows


def run(outdir):
    rankings = json.load(open(os.path.join(outdir, "arm_rankings.json")))
    grades = json.load(open(os.path.join(outdir, "grades.json")))
    bounding = json.load(open(os.path.join(outdir, "bounding_retrieved.json")))
    keys = ["ndcg@10", "ndcg@5", "p@10", "p@5", "recall@10", "mrr", "success@10"]

    print("\n=== per-arm metrics (overall + per-category) ===")
    for arm in ARMS:
        agg = aggregate(_rows_for_arm(rankings, grades, arm), keys)
        print(f"\n--- {arm} ---")
        for grp in ("overall", "terse", "content", "spanning"):
            if grp in agg:
                m = agg[grp]
                print(f"  {grp:8s} (n={m['n']:>2}): " + "  ".join(f"{k}={m[k]:.3f}" for k in keys))

    sizes = [t["tokens"] for b in bounding for t in b["threads"]]
    pct = {
        b: round(100 * sum(1 for s in sizes if s > b) / len(sizes), 1) if sizes else 0
        for b in (4000, 8000, 16000)
    }
    print(f"\n=== bounding (retrieved threads, n={len(sizes)}) ===\n  pct over budget: {pct}")

    decisions = decide(rankings, grades, pct_over_8k=pct[8000])
    print(f"\n=== DECISIONS ===\n  {decisions}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="eval/out")
    args = ap.parse_args()
    run(args.outdir)


if __name__ == "__main__":
    main()
