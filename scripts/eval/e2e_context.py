# scripts/eval/e2e_context.py
"""End-to-end eval, step 1: build the retrieval CONTEXT each setup hands the
answer-AI. Model-independent — built ONCE and reused across every answer model.

For each query, dumps a row {query, category, answer_message_id, gold_text,
contexts:{setup: text}} to eval/out/e2e/contexts.jsonl. Outputs contain real
corpus content -> eval/out (gitignored).

Setups:
  no_context        ""                       (lower-bound anchor)
  answer_only       the gold answer email     (upper-bound anchor)
  plain_C           top-10 emails (no thread) (work-rag, no rerank)
  C_thread_n1       top-1 expanded thread      (work-rag)
  C_thread_n3       top-3 expanded threads
  C_thread_all      all expanded threads
  Cprime_thread_n3  top-3 expanded threads     (work-rag-ctx; C' ranking edge at tight budget)

Run on the HOST (rag env; QDRANT_URL set):
  QDRANT_URL=http://localhost:6333 conda run -n rag --no-capture-output \
    python scripts/eval/e2e_context.py --queries eval/out/queries.jsonl \
    --out eval/out/e2e/contexts.jsonl | tee eval/out/e2e_context.log
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.ingest.embedder import BgeM3Embedder
from src.query.hybrid import build_hybrid_searcher, _qdrant_client
from src.query.thread_expand import group_into_emails, render_thread, fetch_thread_payloads  # noqa: F401
from src.query.fusion import make_rank_fusion

C, CP = "work-rag", "work-rag-ctx"


def _render_emails(hits):
    """Render a list of EmailHit as 'Subject: ..\\n\\nbody' blocks."""
    blocks = []
    for h in hits:
        blocks.append(f"Subject: {h.subject}\n\n{h.body}".strip())
    return "\n\n---\n\n".join(blocks)


def _join_threads(ctxs, n):
    """Join the first-n ThreadContexts' rendered text."""
    sel = ctxs if n is None else ctxs[:n]
    return "\n\n========\n\n".join(c.text for c in sel)


def _gold_text(client, message_id):
    """Fetch the answer email by message_id and render it (Subject + body)."""
    from qdrant_client import models
    flt = models.Filter(must=[models.FieldCondition(
        key="message_id", match=models.MatchValue(value=message_id))])
    points, _ = client.scroll(collection_name=C, scroll_filter=flt,
                              limit=64, with_payload=True, with_vectors=False)
    if not points:
        return ""
    emails = group_into_emails([p.payload for p in points])
    e = emails[0]
    return f"Subject: {e.subject}\n\n{e.body}".strip()


def run(queries_path, out_path, fusion_p):
    print("loading bge-m3 (silent ~1 min)...", flush=True)
    embedder = BgeM3Embedder()
    client = _qdrant_client()

    def mk(collection, fusion_fn=None):
        return build_hybrid_searcher(
            collection, client=client, embedder=embedder, mode="hybrid",
            rerank=False, dense_top_k=20, sparse_top_k=20, top_n=10, fusion_fn=fusion_fn)

    s_cp = mk(CP)
    s_cp_pm = mk(CP, make_rank_fusion(p=fusion_p))   # power-mean fusion on C'

    with open(queries_path) as fh:
        queries = [json.loads(l) for l in fh if l.strip()]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as out:
        for q in queries:
            query = q["query"]
            gold = _gold_text(client, q["answer_message_id"])
            ctxs_cp = s_cp.search_threads(query)
            ctxs_cp_pm = s_cp_pm.search_threads(query)
            contexts = {
                "no_context": "",
                "answer_only": gold,
                "Cprime_thread_n3": _join_threads(ctxs_cp, 3),     # baseline (p=1)
                "Cprime_pm_n3": _join_threads(ctxs_cp_pm, 3),      # power-mean, top-3
                "Cprime_pm_n5": _join_threads(ctxs_cp_pm, 5),      # power-mean, top-5
            }
            out.write(json.dumps({
                "query": query, "category": q["category"],
                "answer_message_id": q["answer_message_id"],
                "gold_text": gold, "contexts": contexts,
            }) + "\n")
            print(f"  done: {query[:50]!r} (gold={'ok' if gold else 'MISSING'})", flush=True)
    print(f"wrote contexts -> {out_path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", default="eval/out/queries.jsonl")
    ap.add_argument("--out", default="eval/out/e2e/contexts.jsonl")
    ap.add_argument("--fusion-p", type=float, default=float("inf"),
                    help="power-mean exponent for the C' fusion arms (inf=max)")
    args = ap.parse_args()
    run(args.queries, args.out, args.fusion_p)


if __name__ == "__main__":
    main()
