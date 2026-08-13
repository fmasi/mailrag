"""Test B: A-vs-C on TREC 2010 Legal (e-discovery, real human qrels). Same 5 arms,
but graded ranking metrics (P@10/nDCG@10/R@100/AP) over many-relevant topics.
Query = full topic text (natural-language production request). Tests whether
dense+rerank wins on e-discovery (reproducing NVIDIA) while hybrid won on email.
"""

import json
import math
import os
import re

from scripts.eval._paths import bootstrap, data_path, jreq, require_key  # noqa: E402

bootstrap()
KEY = require_key(what="the NVIDIA dense+rerank comparison arm")
H = {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}
QD = os.environ["QDRANT_URL"]
TREC = data_path(
    "MAILRAG_EVAL_TREC", "~/msgvault-eval-proof/trec", what="the TREC Legal benchmark directory"
)
EMB = "https://integrate.api.nvidia.com/v1/embeddings"
RR = "https://ai.api.nvidia.com/v1/retrieval/nvidia/reranking"
TOPN = 300
RR_K = 50


def e5q(q):
    return jreq(
        EMB,
        {
            "model": "nvidia/nv-embedqa-e5-v5",
            "input": [q[:8000]],
            "input_type": "query",
            "truncate": "END",
        },
        hdr=H,
    )["data"][0]["embedding"]


def rerank(q, passages):
    rk = jreq(
        RR,
        {
            "model": "nvidia/rerank-qa-mistral-4b",
            "query": {"text": q[:4000]},
            "passages": [{"text": (t or " ")[:4000]} for t in passages],
        },
        hdr=H,
    )["rankings"]
    return [x["index"] for x in sorted(rk, key=lambda z: -z["logit"])]


qrels = {}
for l in open(f"{TREC}/qrels_trec.tsv"):
    t, d, r = l.split("\t")
    qrels.setdefault(t, {})[d] = int(r)
topics = json.load(open(f"{TREC}/topics.json"))


def sanitize(q):
    return re.sub(r"\s+", " ", re.sub(r"[^0-9A-Za-z ]+", " ", q)).strip()


from src.ingest.embedder import BgeM3Embedder
from src.query.hybrid import build_hybrid_searcher

bge = BgeM3Embedder()
A_hyb = build_hybrid_searcher(
    "trec-bge", embedder=bge, mode="hybrid", dense_top_k=TOPN, sparse_top_k=TOPN
)
A_den = build_hybrid_searcher("trec-bge", embedder=bge, mode="dense", dense_top_k=TOPN)
print(f"topics={sorted(topics)}", flush=True)


def dedup(pairs):  # [(docid,text)] -> ranked unique docids + text map
    seen = {}
    order = []
    for d, t in pairs:
        if d not in seen:
            seen[d] = t
            order.append(d)
    return order, seen


def metrics(ranked, relset):
    hits = [1 if d in relset else 0 for d in ranked]
    R = len(relset)

    def p(k):
        return sum(hits[:k]) / k

    def r(k):
        return sum(hits[:k]) / R if R else 0

    def ndcg(k):
        dcg = sum(h / math.log2(i + 2) for i, h in enumerate(hits[:k]))
        idcg = sum(1 / math.log2(i + 2) for i in range(min(k, R)))
        return dcg / idcg if idcg else 0

    ap = 0
    c = 0
    for i, h in enumerate(hits):
        if h:
            c += 1
            ap += c / (i + 1)
    return {"P@10": p(10), "nDCG@10": ndcg(10), "R@100": r(100), "AP": ap / R if R else 0}


ARMS = ["A_dense", "A_hybrid", "A_hybrid+rerank", "C_dense", "C_dense+rerank"]
acc = {a: {m: 0.0 for m in ["P@10", "nDCG@10", "R@100", "AP"]} for a in ARMS}
for t in sorted(topics):
    relset = {d for d, r in qrels[t].items() if r == 1}
    q = sanitize(topics[t])
    nh = A_hyb.search(q)
    nd = A_den.search(q)
    a_hyb = [(n.metadata.get("message_id"), n.get_content()) for n in nh]
    a_den = [(n.metadata.get("message_id"), n.get_content()) for n in nd]
    qv = e5q(q)
    ch = jreq(
        f"{QD}/collections/trec-e5/points/search",
        {"vector": {"name": "dense", "vector": qv}, "limit": TOPN, "with_payload": True},
    )["result"]
    c_den = [(h["payload"]["message_id"], h["payload"].get("text", "")) for h in ch]

    def rerank_pairs(pairs):
        head = pairs[:RR_K]
        order = rerank(q, [t for _, t in head])
        return [head[o] for o in order] + pairs[RR_K:]

    variants = {
        "A_dense": a_den,
        "A_hybrid": a_hyb,
        "A_hybrid+rerank": rerank_pairs(a_hyb),
        "C_dense": c_den,
        "C_dense+rerank": rerank_pairs(c_den),
    }
    for a in ARMS:
        ranked, _ = dedup(variants[a])
        m = metrics(ranked, relset)
        for k in m:
            acc[a][k] += m[k] / len(topics)
        if a in ("A_hybrid", "C_dense+rerank"):
            print(f"  topic {t} {a}: P@10={m['P@10']:.2f} R={len(relset)}", flush=True)

print("\n==== TREC Legal (e-discovery, 3 topics, real human qrels) ====")
print(f"{'arm':20s} {'P@10':>7} {'nDCG@10':>8} {'R@100':>7} {'AP':>7}")
for a in ARMS:
    print(
        f"{a:20s} {acc[a]['P@10']:7.3f} {acc[a]['nDCG@10']:8.3f} {acc[a]['R@100']:7.3f} {acc[a]['AP']:7.3f}"
    )
print("\nDONE", flush=True)
