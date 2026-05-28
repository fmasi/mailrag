"""3-way retrieval probe: dense-only vs hybrid vs hybrid+rerank, across collections.

Run on the HOST in the `rag` conda env (MPS + FlagEmbedding + live Qdrant).
Queries come from --queries <file> (one per line; '#' comments ignored); if omitted,
a few GENERIC built-in queries are used. Keep real/sensitive queries in a local
gitignored file, never in this tracked script.

Example:
  QDRANT_URL=http://localhost:6333 conda run -n rag --no-capture-output \
    python scripts/compare_retrieval.py \
    --collections work-rag work-rag-ctx \
    --queries ~/rag_pass2/probe_queries.txt --top-k 10 | tee ~/compare_retrieval.log
"""
import argparse
import os
import sys

# Make the repo root importable so `from src...` works regardless of cwd
# (matches scripts/build_local_eml_rag.py and scripts/llm_pass2.py).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEFAULT_QUERIES = [
    "product release schedule and generally available versions",
    "meeting reschedule new time proposed",
    "partner cloud platform integration status",
]


def _log(msg):
    print(msg, flush=True)


def load_queries(path):
    if not path:
        return DEFAULT_QUERIES
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#"):
                out.append(line)
    return out or DEFAULT_QUERIES


def node_line(node):
    md = getattr(node, "metadata", None) or getattr(getattr(node, "node", None), "metadata", {}) or {}
    subject = (md.get("subject") or "(no subject)")[:80]
    summary = (md.get("summary") or "")[:90]
    score = getattr(node, "score", None)
    score_s = f"{score:.4f}" if isinstance(score, (int, float)) else "  -   "
    return f"      [{score_s}] {subject} | {summary}"


def run(collections, queries, top_k):
    _log("Loading bge-m3 (FlagEmbedding, MPS) — this is silent for ~1 min...")
    from src.ingest.embedder import BgeM3Embedder
    from src.query.hybrid import build_hybrid_searcher

    embedder = BgeM3Embedder()  # shared across all searchers
    _log("bge-m3 loaded.\n")

    modes = [("dense-only", dict(mode="dense", rerank=False)),
             ("hybrid", dict(mode="hybrid", rerank=False)),
             ("hybrid+rerank", dict(mode="hybrid", rerank=True))]

    for collection in collections:
        for query in queries:
            _log(f"\n=== collection={collection} | query={query!r} ===")
            for label, kw in modes:
                searcher = build_hybrid_searcher(
                    collection, embedder=embedder,
                    dense_top_k=max(top_k, 20), sparse_top_k=max(top_k, 20),
                    top_n=top_k, **kw,
                )
                nodes = searcher.search(query)[:top_k]
                _log(f"  --- {label} (top {len(nodes)}) ---")
                for n in nodes:
                    _log(node_line(n))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--collections", nargs="+", default=["work-rag", "work-rag-ctx"])
    ap.add_argument("--queries", default=None, help="file with one query per line")
    ap.add_argument("--top-k", type=int, default=10)
    args = ap.parse_args(argv)
    run(args.collections, load_queries(args.queries), args.top_k)
    return 0


if __name__ == "__main__":
    sys.exit(main())
