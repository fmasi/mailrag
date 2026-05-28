# scripts/eval_summary_rerank.py
"""8-config matrix over two query sets.

Terse set: JSON [{"query": "...", "target": "<message_id>"}, ...] -> recall@10 + target rank.
Content set: plain text, one query per line -> top-5 subjects per config (eyeball).
Run on HOST in the `rag` env (MPS + live Qdrant). Real queries live in LOCAL files only.

  QDRANT_URL=http://localhost:6333 conda run -n rag --no-capture-output \
    python scripts/eval_summary_rerank.py \
    --terse ~/rag_pass2/terse_queries.json --content ~/rag_pass2/probe_queries.txt \
    | tee ~/eval_summary_rerank.log
"""
import argparse, json, sys

TOP_N = 10
CONFIGS = [
    ("C hybrid",           dict(collection="work-rag",     mode="hybrid", rerank=False)),
    ("C rerank-body  f20", dict(collection="work-rag",     mode="hybrid", rerank=True,  dense_top_k=20, sparse_top_k=20)),
    ("C rerank-summ  f20", dict(collection="work-rag",     mode="hybrid", rerank_with_summary=True, dense_top_k=20, sparse_top_k=20)),
    ("C' hybrid",          dict(collection="work-rag-ctx", mode="hybrid", rerank=False)),
    ("C' rerank-body f20", dict(collection="work-rag-ctx", mode="hybrid", rerank=True,  dense_top_k=20, sparse_top_k=20)),
    ("C rerank-body  f50", dict(collection="work-rag",     mode="hybrid", rerank=True,  dense_top_k=50, sparse_top_k=50)),
    ("C rerank-summ  f50", dict(collection="work-rag",     mode="hybrid", rerank_with_summary=True, dense_top_k=50, sparse_top_k=50)),
    ("C' rerank-body f50", dict(collection="work-rag-ctx", mode="hybrid", rerank=True,  dense_top_k=50, sparse_top_k=50)),
]

def _log(m): print(m, flush=True)

def node_msgid(n):
    md = getattr(n, "metadata", {}) or {}
    return md.get("message_id")

def node_subject(n):
    md = getattr(n, "metadata", {}) or {}
    return (md.get("subject") or "(no subject)")[:70]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--terse", default=None, help="JSON [{query,target(message_id)}]")
    ap.add_argument("--content", default=None, help="text, one query per line")
    args = ap.parse_args()

    _log("Loading bge-m3 (silent ~1 min)...")
    from src.ingest.embedder import BgeM3Embedder
    from src.query.hybrid import build_hybrid_searcher
    embedder = BgeM3Embedder()
    _log("loaded.\n")

    def make(cfg):
        return build_hybrid_searcher(embedder=embedder, top_n=TOP_N, **cfg)

    # Terse set -> recall@10 + rank
    if args.terse:
        terse = json.load(open(args.terse, encoding="utf-8"))
        _log(f"==== TERSE recall@{TOP_N} (target rank by message_id; '-' = not found) ====")
        _log("query".ljust(40) + " | " + " | ".join(name for name, _ in CONFIGS))
        for item in terse:
            q, target = item["query"], item["target"]
            cells = []
            for _, cfg in CONFIGS:
                nodes = make(cfg).search(q)[:TOP_N]
                ids = [node_msgid(n) for n in nodes]
                rank = ids.index(target) + 1 if target in ids else None
                cells.append(str(rank) if rank else "-")
            _log(q[:40].ljust(40) + " | " + " | ".join(c.center(len(name)) for c, (name, _) in zip(cells, CONFIGS)))

    # Content set -> top-5 subjects per config
    if args.content:
        queries = [l.strip() for l in open(args.content, encoding="utf-8")
                   if l.strip() and not l.startswith("#")]
        _log("\n\n==== CONTENT-RICH top-5 subjects per config ====")
        for q in queries:
            _log(f"\n### {q}")
            for name, cfg in CONFIGS:
                nodes = make(cfg).search(q)[:5]
                _log(f"  -- {name} --")
                for n in nodes:
                    _log(f"     {node_subject(n)}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
