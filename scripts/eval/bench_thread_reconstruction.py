"""Test A: does thread reconstruction help? Compare EMAIL-recall (gold email's own
chunk in top-k) vs THREAD-recall (gold's thread present among the top-k distinct
threads -> thread expansion would surface the gold), by category. Private corpus.
"""

import json
import time
import urllib.error
import urllib.request

from scripts.eval._paths import bootstrap, data_path, require_key  # noqa: E402

bootstrap()
KEY = require_key(what="the cross-encoder rerank arm")
H = {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}
RR = "https://ai.api.nvidia.com/v1/retrieval/nvidia/reranking"
QFILE = data_path(
    "MAILRAG_EVAL_QUERIES", "eval/out/queries_360.jsonl", what="the private 360-query labelled set"
)
TOPK = 20


def jreq(url, body, hdr, timeout=60):
    for a in range(8):
        try:
            r = urllib.request.Request(
                url, data=json.dumps(body).encode(), headers=hdr, method="POST"
            )
            with urllib.request.urlopen(r, timeout=timeout) as z:
                return json.load(z)
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(min(2**a, 30))
                continue
            raise
    raise RuntimeError("429")


def rerank(q, passages):
    rk = jreq(
        RR,
        {
            "model": "nvidia/rerank-qa-mistral-4b",
            "query": {"text": q},
            "passages": [{"text": (t or " ")[:4000]} for t in passages],
        },
        H,
    )["rankings"]
    return [x["index"] for x in sorted(rk, key=lambda z: -z["logit"])]


from src.ingest.embedder import BgeM3Embedder
from src.query.hybrid import build_hybrid_searcher

queries = [json.loads(l) for l in open(QFILE)]
bge = BgeM3Embedder()
S = build_hybrid_searcher(
    "work-rag-ctx-threadaware", embedder=bge, mode="hybrid", dense_top_k=TOPK, sparse_top_k=TOPK
)
print(f"queries {len(queries)}", flush=True)


def email_rank(hits, gold):  # hits: list of (mid,tid)
    return next((i for i, (m, _) in enumerate(hits) if m == gold), None)


def thread_rank(hits, gtid):  # distinct-thread position of gold thread
    seen = []
    for _, t in hits:
        if t not in seen:
            seen.append(t)
        if t == gtid:
            return len(seen) - 1
    return None


from collections import defaultdict

arms = ["hybrid", "hybrid+rerank"]
agg = {a: defaultdict(lambda: defaultdict(float)) for a in arms}  # agg[arm][cat][metric]
for q in queries:
    cat = q.get("category", "?")
    gmid = q["answer_message_id"]
    gtid = q["thread_id"]
    nodes = S.search(q["query"])
    base = [(n.metadata.get("message_id"), n.metadata.get("thread_id")) for n in nodes]
    texts = [n.get_content() for n in nodes]
    order = rerank(q["query"], texts)
    variants = {"hybrid": base, "hybrid+rerank": [base[o] for o in order]}
    for a in arms:
        hits = variants[a]
        er = email_rank(hits, gmid)
        tr = thread_rank(hits, gtid)
        for cat_key in (cat, "ALL"):
            d = agg[a][cat_key]
            d["n"] += 1
            for k in (1, 5, 10):
                if er is not None and er < k:
                    d[f"e{k}"] += 1
                if tr is not None and tr < k:
                    d[f"t{k}"] += 1

cats = ["ALL", "terse", "content", "spanning"]
for a in arms:
    print(f"\n==== arm: {a}  (EMAIL-recall@k  vs  THREAD-recall@k, %) ====")
    print(
        f"{'category':10s} {'n':>4}  {'E@1':>5}{'T@1':>6}   {'E@5':>5}{'T@5':>6}   {'E@10':>5}{'T@10':>6}"
    )
    for c in cats:
        d = agg[a][c]
        n = d["n"]
        if not n:
            continue

        def p(m):
            return 100 * d[m] / n

        print(
            f"{c:10s} {int(n):>4}  {p('e1'):5.1f}{p('t1'):6.1f}   {p('e5'):5.1f}{p('t5'):6.1f}   {p('e10'):5.1f}{p('t10'):6.1f}"
        )
print("\nDONE", flush=True)
