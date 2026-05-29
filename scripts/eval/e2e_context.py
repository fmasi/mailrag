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


def run(queries_path, out_path):
    print("loading bge-m3 (silent ~1 min)...", flush=True)
    embedder = BgeM3Embedder()
    client = _qdrant_client()

    def mk(collection):
        return build_hybrid_searcher(
            collection, client=client, embedder=embedder, mode="hybrid",
            rerank=False, dense_top_k=20, sparse_top_k=20, top_n=10)

    s_c, s_cp = mk(C), mk(CP)

    with open(queries_path) as fh:
        queries = [json.loads(l) for l in fh if l.strip()]

    from src.eval.flatten import flatten_nodes
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as out:
        for q in queries:
            query = q["query"]
            gold = _gold_text(client, q["answer_message_id"])
            plain_hits = flatten_nodes(s_c.search(query))[:10]
            ctxs_c = s_c.search_threads(query)
            ctxs_cp = s_cp.search_threads(query)
            contexts = {
                "no_context": "",
                "answer_only": gold,
                "plain_C": _render_emails(plain_hits),
                "C_thread_n1": _join_threads(ctxs_c, 1),
                "C_thread_n3": _join_threads(ctxs_c, 3),
                "C_thread_all": _join_threads(ctxs_c, None),
                "Cprime_thread_n1": _join_threads(ctxs_cp, 1),
                "Cprime_thread_n3": _join_threads(ctxs_cp, 3),
                "Cprime_thread_all": _join_threads(ctxs_cp, None),
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
    args = ap.parse_args()
    run(args.queries, args.out)


if __name__ == "__main__":
    main()
