# scripts/eval/e2e_context.py
"""End-to-end eval, step 1: build the retrieval CONTEXT each setup hands the
answer-AI. Model-independent — built ONCE and reused across every answer model.

For each query, dumps a row {query, category, answer_message_id, gold_text,
contexts:{setup: text}} to eval/out/e2e/contexts.jsonl. Outputs contain real
corpus content -> eval/out (gitignored).

Setups (5 HyDE arms):
  no_context              ""                          (lower-bound anchor)
  answer_only             the gold answer email        (upper-bound anchor)
  Cprime_n3_raw           top-3 threads, raw query     (C', baseline RRF)
  Cprime_n3_hyde_pure     top-3 threads, hypothetical only  (C', HyDE pure)
  Cprime_n3_hyde_augment  top-3 threads, query+hypothetical (C', HyDE augment)

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
from src.query.hybrid import _qdrant_client, build_hybrid_searcher
from src.query.hyde import combine_query
from src.query.thread_expand import (  # noqa: F401
    fetch_thread_payloads,
    group_into_emails,
    render_thread,
)

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

    flt = models.Filter(
        must=[models.FieldCondition(key="message_id", match=models.MatchValue(value=message_id))]
    )
    points, _ = client.scroll(
        collection_name=C, scroll_filter=flt, limit=64, with_payload=True, with_vectors=False
    )
    if not points:
        return ""
    emails = group_into_emails([p.payload for p in points])
    e = emails[0]
    return f"Subject: {e.subject}\n\n{e.body}".strip()


def _load_hyde(path):
    """{query: hypothetical} from a hyde_queries.jsonl, or {} if path is falsy."""
    if not path:
        return {}
    out = {}
    for line in open(path):
        if line.strip():
            r = json.loads(line)
            out[r["query"]] = r.get("hypothetical", "")
    return out


def run(queries_path, out_path, hyde_file):
    print("loading bge-m3 (silent ~1 min)...", flush=True)
    embedder = BgeM3Embedder()
    client = _qdrant_client()

    # C', hybrid, RRF default (the recommended stack); HyDE only changes the query string.
    s_cp = build_hybrid_searcher(
        CP,
        client=client,
        embedder=embedder,
        mode="hybrid",
        rerank=False,
        dense_top_k=20,
        sparse_top_k=20,
        top_n=10,
    )

    hyde_map = _load_hyde(hyde_file)

    with open(queries_path) as fh:
        queries = [json.loads(l) for l in fh if l.strip()]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as out:
        for q in queries:
            query = q["query"]
            gold = _gold_text(client, q["answer_message_id"])
            hyp = hyde_map.get(query, "")
            raw_ctx = _join_threads(s_cp.search_threads(query), 3)
            pure_ctx = _join_threads(s_cp.search_threads(combine_query(query, hyp, "pure")), 3)
            aug_ctx = _join_threads(s_cp.search_threads(combine_query(query, hyp, "augment")), 3)
            contexts = {
                "no_context": "",
                "answer_only": gold,
                "Cprime_n3_raw": raw_ctx,  # baseline (raw query)
                "Cprime_n3_hyde_pure": pure_ctx,  # search with hypothetical only
                "Cprime_n3_hyde_augment": aug_ctx,  # search with query + hypothetical
            }
            out.write(
                json.dumps(
                    {
                        "query": query,
                        "category": q["category"],
                        "answer_message_id": q["answer_message_id"],
                        "gold_text": gold,
                        "contexts": contexts,
                    }
                )
                + "\n"
            )
            print(f"  done: {query[:50]!r} (gold={'ok' if gold else 'MISSING'})", flush=True)
    print(f"wrote contexts -> {out_path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", default="eval/out/queries.jsonl")
    ap.add_argument("--out", default="eval/out/e2e/contexts.jsonl")
    ap.add_argument("--hyde-file", default="eval/out/hyde_queries.jsonl")
    args = ap.parse_args()
    run(args.queries, args.out, args.hyde_file)


if __name__ == "__main__":
    main()
