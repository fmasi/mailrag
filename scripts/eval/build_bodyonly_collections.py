"""Test C: build BODY-ONLY private collections (no summary prepended) to isolate
the contextual-summary feature. bge-m3 hybrid (dense+sparse) + e5 dense, over the
same work-rag chunk bodies. Compare bench vs the summary+body headline."""

import json
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from scripts.eval._paths import bootstrap  # noqa: E402

bootstrap()
from src.ingest import hybrid_qdrant as hq
from src.ingest.embedder import BgeM3Embedder
from src.ingest.sparse import lexical_weights_to_sparse

QD = os.environ["QDRANT_URL"]
SRC = "work-rag-ctx-threadaware"
KEY = os.environ.get("NVIDIA_API_KEY")
assert KEY
H = {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}
EMB = "https://integrate.api.nvidia.com/v1/embeddings"


def jreq(url, body=None, method="POST", hdr=None, timeout=120):
    for a in range(8):
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
                time.sleep(min(2**a, 30))
                continue
            raise
    raise RuntimeError("429")


# pull bodies
pts = []
off = None
while True:
    b = {"limit": 4000, "with_payload": ["message_id", "text"], "with_vector": False}
    if off:
        b["offset"] = off
    res = jreq(f"{QD}/collections/{SRC}/points/scroll", b)["result"]
    pts += res["points"]
    off = res.get("next_page_offset")
    if not off:
        break
mids = [p["payload"].get("message_id") for p in pts]
bodies = [((p["payload"].get("text") or "").strip() or "(no content)") for p in pts]
print(f"pulled {len(pts)} bodies", flush=True)

# --- bge-m3 hybrid body-only ---
bge = BgeM3Embedder()
client = hq.get_client(QD)
hq.ensure_hybrid_collection(client, "work-rag-bge-body", dim=1024, recreate=True)
t0 = time.time()
n = 0
for i in range(0, len(bodies), 256):
    bb = bodies[i : i + 256]
    bm = mids[i : i + 256]
    dense, sparse = bge.encode(bb, batch_size=32, max_length=512)
    points = []
    for j, (mid, dv, lw) in enumerate(zip(bm, dense, sparse)):
        idx, val = lexical_weights_to_sparse(lw)
        points.append(hq.make_point(i + j, dv, idx, val, {"message_id": mid, "text": bb[j][:6000]}))
    hq.upsert(client, "work-rag-bge-body", points)
    n += len(points)
print(f"bge-m3 body-only: {n} pts in {time.time() - t0:.0f}s", flush=True)


# --- e5 dense body-only ---
def embed(batch):
    return [
        d["embedding"]
        for d in jreq(
            EMB,
            {
                "model": "nvidia/nv-embedqa-e5-v5",
                "input": batch,
                "input_type": "passage",
                "truncate": "END",
            },
            hdr=H,
        )["data"]
    ]


batches = [bodies[i : i + 64] for i in range(0, len(bodies), 64)]
t0 = time.time()
with ThreadPoolExecutor(max_workers=3) as ex:
    res = list(ex.map(embed, batches))
vecs = [v for r in res for v in r]
print(f"e5 body-only embedded {len(vecs)} in {time.time() - t0:.0f}s", flush=True)
try:
    jreq(f"{QD}/collections/work-rag-e5-body", method="DELETE", timeout=15)
except urllib.error.HTTPError:
    pass
jreq(
    f"{QD}/collections/work-rag-e5-body",
    {"vectors": {"dense": {"size": 1024, "distance": "Cosine"}}},
    method="PUT",
    timeout=15,
)
P = [
    {
        "id": i,
        "vector": {"dense": vecs[i]},
        "payload": {"message_id": mids[i], "text": bodies[i][:6000]},
    }
    for i in range(len(vecs))
]
for j in range(0, len(P), 512):
    jreq(
        f"{QD}/collections/work-rag-e5-body/points?wait=true",
        {"points": P[j : j + 512]},
        method="PUT",
        timeout=60,
    )
print("DONE body-only collections: work-rag-bge-body, work-rag-e5-body", flush=True)
