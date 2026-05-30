# scripts/eval/diagnose_coverage.py
"""Diagnose the retrieval-coverage misses (issue #12).

For each eval query, find where the gold thread/email ranks in dense-only,
sparse-only, and hybrid retrieval on C (work-rag) and C' (work-rag-ctx), bucket
the cause (covered/budget/fusion/hard), and for hard misses run an oracle
escalation (gold subject -> gold body) to split index-defect from vocab-gap.
Outputs (real corpus content) -> eval/out (gitignored).

Run on the HOST (rag env; QDRANT_URL set):
  QDRANT_URL=http://localhost:6333 conda run -n rag --no-capture-output \
    python scripts/eval/diagnose_coverage.py --queries eval/out/queries.jsonl \
    --out eval/out/coverage_diag.jsonl | tee eval/out/coverage_diag.log
"""
import argparse
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.ingest.embedder import BgeM3Embedder
from src.query.hybrid import build_hybrid_searcher, _qdrant_client
from src.query.fusion import make_rank_fusion
from src.query.thread_expand import _node_metadata, fetch_thread_payloads, group_into_emails
from src.eval.coverage_diag import (
    best_gold_rank, classify_miss, distinct_thread_rank, is_terse, lexical_overlap,
    oracle_root_cause, is_bad_query)

C, CP = "work-rag", "work-rag-ctx"
DEEP_K = 200          # search/rank-trace depth: how far down each ranked list we look
                      # (distinct from K below, which is the "good rank" cutoff for fusion)
TOP_HITS, N, K = 10, 3, 20   # expansion pool / thread budget / single-mode "good rank"
OVERLAP_BAD = 0.15    # query<->thread overlap below this + hard + oracle-fail => bad query


def _hits(searcher, query):
    """Return ranked [{thread_id, message_id}] from a searcher.search() call."""
    out = []
    for node in searcher.search(query):
        md = _node_metadata(node)
        out.append({"thread_id": md.get("thread_id"), "message_id": md.get("message_id")})
    return out


def _gold_email(client, message_id):
    """Fetch the gold email (subject/body) for oracle queries; '' fields if missing."""
    from qdrant_client import models
    # Read the gold email from C (work-rag): the eval queries were generated from C,
    # so the gold email/thread is guaranteed present here; bodies are identical in C'.
    flt = models.Filter(must=[models.FieldCondition(
        key="message_id", match=models.MatchValue(value=message_id))])
    pts, _ = client.scroll(collection_name=C, scroll_filter=flt, limit=64,
                           with_payload=True, with_vectors=False)
    if not pts:
        return {"subject": "", "body": ""}
    emails = group_into_emails([p.payload for p in pts])
    e = emails[0]
    return {"subject": e.subject or "", "body": e.body or ""}


def _thread_text(client, thread_id):
    """Concatenated bodies of every email in the gold thread (for query<->thread overlap)."""
    # C (work-rag) is where the eval queries were generated, so the gold thread is
    # guaranteed present here regardless of which collection is being diagnosed.
    payloads = fetch_thread_payloads(client, C, [thread_id])
    emails = group_into_emails(payloads)
    return "\n".join(e.body for e in emails if e.body).strip()


def _ranks_for(searchers, query, gtid, gmid):
    """Compute the rank coordinates classify_miss needs, plus raw per-mode ranks."""
    h_hits = _hits(searchers["hybrid"], query)
    d_hits = _hits(searchers["dense"], query)
    s_hits = _hits(searchers["sparse"], query)
    hb = best_gold_rank(h_hits, gtid, gmid)
    db = best_gold_rank(d_hits, gtid, gmid)
    sb = best_gold_rank(s_hits, gtid, gmid)
    return {
        "hyb_distinct_rank": distinct_thread_rank(h_hits, gtid),
        "hyb_thread_rank": hb["thread_rank"],
        "hyb_email_rank": hb["email_rank"],
        "dense_thread_rank": db["thread_rank"],
        "sparse_thread_rank": sb["thread_rank"],
    }


def run(queries_path, out_path, fusion_p):
    print("loading bge-m3 (silent ~1 min)...", flush=True)
    embedder = BgeM3Embedder()
    client = _qdrant_client()

    def modes(coll, p):
        fusion_fn = make_rank_fusion(p=p)
        out = {}
        for m in ("dense", "sparse", "hybrid"):
            out[m] = build_hybrid_searcher(
                coll, client=client, embedder=embedder, mode=m, rerank=False,
                dense_top_k=DEEP_K, sparse_top_k=DEEP_K, top_n=DEEP_K,
                fusion_fn=(fusion_fn if m == "hybrid" else None))
        return out

    s_c, s_cp = modes(C, fusion_p), modes(CP, fusion_p)

    with open(queries_path) as fh:
        queries = [json.loads(l) for l in fh if l.strip()]

    hist = collections.Counter()      # bucket histogram on the headline setup (C')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as out:
        for q in queries:
            query, gtid, gmid = q["query"], q["thread_id"], q["answer_message_id"]
            gold = _gold_email(client, gmid)
            thread_text = _thread_text(client, gtid)
            ranks_c = _ranks_for(s_c, query, gtid, gmid)
            ranks_cp = _ranks_for(s_cp, query, gtid, gmid)
            bucket = classify_miss(ranks_cp, TOP_HITS, N, K)   # headline = C'

            row = {
                "query": query, "category": q["category"],
                "thread_id": gtid, "answer_message_id": gmid,
                "bucket_cprime": bucket,
                "bucket_c": classify_miss(ranks_c, TOP_HITS, N, K),
                "ranks_c": ranks_c, "ranks_cprime": ranks_cp,
                "gold_terse": is_terse(gold["body"]),
                "overlap_query_gold": lexical_overlap(query, gold["body"]),
                "overlap_query_thread": lexical_overlap(query, thread_text),
            }

            # Oracle escalation on hard misses (on C', the headline setup).
            if bucket == "hard":
                body_q = (gold["body"] or "")[:512]
                subj_q = gold["subject"] or ""
                oracle = {}
                for name, qq in (("subject", subj_q), ("body", body_q)):
                    if qq.strip():
                        r = best_gold_rank(_hits(s_cp["hybrid"], qq), gtid, gmid)
                        oracle[name] = r["thread_rank"]
                    else:
                        oracle[name] = None
                row["oracle"] = oracle
                row["root_cause"] = oracle_root_cause(oracle.get("body"), TOP_HITS)
                row["bad_query"] = is_bad_query(
                    oracle.get("body"), row["overlap_query_thread"], TOP_HITS, OVERLAP_BAD)

            out.write(json.dumps(row) + "\n")
            hist[bucket] += 1
            print(f"  {bucket:8s} {query[:48]!r}", flush=True)

    total = sum(hist.values())
    print(f"\n=== cause histogram (C', N={N}, p={fusion_p}) ===", flush=True)
    for b in ("covered", "budget", "fusion", "hard"):
        c = hist.get(b, 0)
        pct = (100 * c / total) if total else 0.0
        print(f"  {b:8s} {c:3d}  ({pct:.0f}%)", flush=True)
    print(f"wrote per-query diagnostic -> {out_path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", default="eval/out/queries.jsonl")
    ap.add_argument("--out", default="eval/out/coverage_diag.jsonl")
    ap.add_argument("--fusion-p", type=float, default=1.0,
                    help="power-mean exponent for hybrid fusion (1=RRF sum, inf=max)")
    args = ap.parse_args()
    run(args.queries, args.out, args.fusion_p)


if __name__ == "__main__":
    main()
