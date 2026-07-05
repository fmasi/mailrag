"""A-vs-C retrieval benchmark: bge-m3 learned-sparse hybrid (A) vs a NVIDIA-native
dense-embedding + reranking-NIM recipe (C), scored against hard gold labels.

A arms use the real production path (``build_hybrid_searcher`` + bge-m3). C arms
query a dense NIM-embedded collection (see build_nim_dense_collection.py) and
rerank with the NVIDIA reranking NIM. The SAME reranker is used on every rerank
arm, and the embedding dimension is held constant, so the comparison isolates
*recipe* (learned-sparse hybrid vs dense+rerank), not vector size or reranker.

Usage:
    NVIDIA_API_KEY=nvapi-... python scripts/eval/bench_avc.py QUERIES.jsonl \
        [HYBRID_COL=work-rag-ctx-threadaware] [E5_COL=work-rag-e5]

QUERIES.jsonl rows: {"query": str, "answer_message_id": str, "category": str?}

Ops notes: hosted NIM rate-limits above ~3 concurrent (HTTP 429) -> 3 workers +
backoff. bge-m3 needs HF offline mode if no HF token (model loads from cache).
"""

import json
import os
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

WT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, WT)
os.chdir(WT)
os.environ.setdefault("QDRANT_URL", "http://localhost:6333")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

QFILE = (
    sys.argv[1] if len(sys.argv) > 1 else os.environ.get("QUERIES", "eval/out/queries_360.jsonl")
)
HYBRID = os.environ.get("HYBRID_COL", "work-rag-ctx-threadaware")
E5 = os.environ.get("E5_COL", "work-rag-e5")
QD = os.environ["QDRANT_URL"]
TOPK = int(os.environ.get("TOPK", "20"))
KEY = os.environ.get("NVIDIA_API_KEY")
assert KEY, "set NVIDIA_API_KEY"
H = {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}
EMB = "https://integrate.api.nvidia.com/v1/embeddings"
RR = "https://ai.api.nvidia.com/v1/retrieval/nvidia/reranking"
E5_MODEL = os.environ.get("E5_MODEL", "nvidia/nv-embedqa-e5-v5")
RR_MODEL = os.environ.get("RR_MODEL", "nvidia/rerank-qa-mistral-4b")


def jreq(url, body=None, method="POST", hdr=None, timeout=120):
    for attempt in range(8):
        try:
            r = urllib.request.Request(
                url,
                data=(json.dumps(body).encode() if body is not None else None),
                headers=(hdr or {"Content-Type": "application/json"}),
                method=method,
            )
            with urllib.request.urlopen(r, timeout=timeout) as z:
                return json.load(z) if z.length != 0 else {}
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(min(2**attempt, 30))
                continue
            raise
    raise RuntimeError("429 exhausted")


def e5_query(q):
    return jreq(
        EMB, {"model": E5_MODEL, "input": [q], "input_type": "query", "truncate": "END"}, hdr=H
    )["data"][0]["embedding"]


def rerank(q, passages):
    rk = jreq(
        RR,
        {
            "model": RR_MODEL,
            "query": {"text": q},
            "passages": [{"text": (t or " ")[:4000]} for t in passages],
        },
        hdr=H,
    )["rankings"]
    return [x["index"] for x in sorted(rk, key=lambda z: -z["logit"])]


def main():
    queries = [json.loads(l) for l in open(QFILE)]
    print(f"queries: {len(queries)}  topk={TOPK}  hybrid={HYBRID}  e5={E5}", flush=True)

    from src.ingest.embedder import BgeM3Embedder
    from src.query.hybrid import build_hybrid_searcher

    bge = BgeM3Embedder()
    A_hyb = build_hybrid_searcher(
        HYBRID, embedder=bge, mode="hybrid", dense_top_k=TOPK, sparse_top_k=TOPK
    )
    A_den = build_hybrid_searcher(HYBRID, embedder=bge, mode="dense", dense_top_k=TOPK)

    cand = {}
    t0 = time.time()
    for i, q in enumerate(queries):
        nh = A_hyb.search(q["query"])
        nd = A_den.search(q["query"])
        cand[i] = {
            "A_hybrid": [(n.metadata.get("message_id"), n.get_content()) for n in nh],
            "A_dense": [(n.metadata.get("message_id"), n.get_content()) for n in nd],
        }
    print(f"A retrieval done {time.time() - t0:.0f}s", flush=True)

    def c_search(i):
        qv = e5_query(queries[i]["query"])
        hits = jreq(
            f"{QD}/collections/{E5}/points/search",
            {"vector": {"name": "dense", "vector": qv}, "limit": TOPK, "with_payload": True},
        )["result"]
        return i, [(h["payload"]["message_id"], h["payload"].get("text", "")) for h in hits]

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=3) as ex:
        for i, lst in ex.map(c_search, range(len(queries))):
            cand[i]["C_dense"] = lst
    print(f"C retrieval done {time.time() - t0:.0f}s", flush=True)

    def do_rerank(args):
        i, arm = args
        base = cand[i][arm]
        order = rerank(queries[i]["query"], [t for _, t in base])
        return i, arm, [base[o][0] for o in order]

    jobs = [(i, "A_hybrid") for i in range(len(queries))] + [
        (i, "C_dense") for i in range(len(queries))
    ]
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=3) as ex:
        for i, arm, mids in ex.map(do_rerank, jobs):
            cand[i][arm + "+rerank"] = [(m, None) for m in mids]
    print(f"rerank done {time.time() - t0:.0f}s ({len(jobs)} calls)", flush=True)

    ARMS = ["A_dense", "A_hybrid", "A_hybrid+rerank", "C_dense", "C_dense+rerank"]

    def rank_of(lst, gold):
        return next((j for j, (m, _) in enumerate(lst) if m == gold), None)

    agg = {a: defaultdict(float) for a in ARMS}
    bycat = {a: defaultdict(lambda: defaultdict(float)) for a in ARMS}
    for i, q in enumerate(queries):
        gold = q["answer_message_id"]
        cat = q.get("category", "?")
        for a in ARMS:
            r = rank_of(cand[i][a], gold)
            A = agg[a]
            C = bycat[a][cat]
            A["n"] += 1
            C["n"] += 1
            if r is not None:
                for k in (1, 5, 10):
                    if r < k:
                        A[f"r{k}"] += 1
                        C[f"r{k}"] += 1
                A["mrr"] += 1 / (r + 1)
                C["mrr"] += 1 / (r + 1)

    def pct(a, k):
        return 100 * agg[a][f"r{k}"] / agg[a]["n"]

    print("\n==== A-vs-C (recall@k %, MRR@10) ====")
    print(f"{'arm':20s} {'R@1':>6} {'R@5':>6} {'R@10':>6} {'MRR':>6}")
    for a in ARMS:
        print(
            f"{a:20s} {pct(a, 1):6.1f} {pct(a, 5):6.1f} {pct(a, 10):6.1f} {agg[a]['mrr'] / agg[a]['n']:6.3f}"
        )
    cats = sorted({q.get("category", "?") for q in queries})
    print("\n==== by category: R@5 ====")
    print(f"{'arm':20s} " + " ".join(f"{c:>9}" for c in cats))
    for a in ARMS:
        print(
            f"{a:20s} "
            + " ".join(f"{100 * bycat[a][c]['r5'] / max(bycat[a][c]['n'], 1):9.1f}" for c in cats)
        )
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
