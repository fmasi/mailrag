"""Build a dense-only Qdrant collection by re-embedding an existing collection's
chunks with a NVIDIA embedding NIM (the C / NVIDIA-native side of the A-vs-C
retrieval benchmark).

Reads the chunk text (``summary`` + ``text`` payload) from a SOURCE hybrid
collection, embeds each via the hosted NVIDIA embedding endpoint
(OpenAI-compatible ``/v1/embeddings``), and upserts dense vectors into a DST
collection. Dense-only: the OpenAI embeddings API cannot carry learned sparse.

Usage:
    NVIDIA_API_KEY=nvapi-... python scripts/eval/build_nim_dense_collection.py \
        [SRC=work-rag-ctx-threadaware] [DST=work-rag-e5] [MODEL=nvidia/nv-embedqa-e5-v5]

Notes:
  - Concurrency is capped at 3 workers with exponential backoff: the hosted NIM
    returns HTTP 429 above ~8 concurrent requests.
  - ``truncate=END`` keeps over-length chunks from erroring; verify the corpus
    fits the model's token cap (e5 = 512) before trusting the comparison.
"""

import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from scripts.eval._paths import jreq

QD = os.environ.get("QDRANT_URL", "http://localhost:6333")
SRC = os.environ.get("SRC", "work-rag-ctx-threadaware")
DST = os.environ.get("DST", "work-rag-e5")
MODEL = os.environ.get("MODEL", "nvidia/nv-embedqa-e5-v5")
EMB = "https://integrate.api.nvidia.com/v1/embeddings"
KEY = os.environ.get("NVIDIA_API_KEY")
assert KEY, "set NVIDIA_API_KEY"
H = {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}


def main():
    pts, off = [], None
    while True:
        b = {"limit": 4000, "with_payload": ["message_id", "text", "summary"], "with_vector": False}
        if off:
            b["offset"] = off
        res = jreq(f"{QD}/collections/{SRC}/points/scroll", b)["result"]
        pts += res["points"]
        off = res.get("next_page_offset")
        if not off:
            break

    def etext(p):
        s = (p["payload"].get("summary") or "").strip()
        t = p["payload"].get("text") or ""
        return (s + "\n\n" + t) if s else t

    texts = [etext(p) for p in pts]
    mids = [p["payload"].get("message_id") for p in pts]
    bodies = [(p["payload"].get("text") or "") for p in pts]
    print(f"pulled {len(pts)} chunks from {SRC}", flush=True)

    def embed(batch):
        return [
            d["embedding"]
            for d in jreq(
                EMB,
                {"model": MODEL, "input": batch, "input_type": "passage", "truncate": "END"},
                hdr=H,
            )["data"]
        ]

    batches = [texts[i : i + 64] for i in range(0, len(texts), 64)]
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=3) as ex:
        results = list(ex.map(embed, batches))
    vecs = [v for r in results for v in r]
    dt = time.time() - t0
    print(
        f"embedded {len(vecs)} in {dt:.0f}s ({len(vecs) / dt:.0f}/s, {len(batches)} reqs)",
        flush=True,
    )

    try:
        jreq(f"{QD}/collections/{DST}", method="DELETE", timeout=15)
    except urllib.error.HTTPError:
        pass
    dim = len(vecs[0])
    jreq(
        f"{QD}/collections/{DST}",
        {"vectors": {"dense": {"size": dim, "distance": "Cosine"}}},
        method="PUT",
        timeout=15,
    )
    points = [
        {
            "id": i,
            "vector": {"dense": vecs[i]},
            "payload": {"message_id": mids[i], "text": bodies[i][:6000]},
        }
        for i in range(len(vecs))
    ]
    for j in range(0, len(points), 512):
        jreq(
            f"{QD}/collections/{DST}/points?wait=true",
            {"points": points[j : j + 512]},
            method="PUT",
            timeout=60,
        )
    cnt = jreq(f"{QD}/collections/{DST}", method="GET")["result"]["points_count"]
    print(f"DONE {DST}: {cnt} points (dim={dim})", flush=True)


if __name__ == "__main__":
    main()
