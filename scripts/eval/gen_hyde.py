# scripts/eval/gen_hyde.py
"""Pre-generate one HyDE hypothetical answer per eval query (issue #16).

Reads eval/out/queries.jsonl, writes eval/out/hyde_queries.jsonl
({query, hypothetical} per line) using a fast local model (e4b@8bit). Generated
once and reused by diagnose_coverage.py + e2e_context.py across all HyDE arms.
Real content -> eval/out (gitignored).

Run on the HOST (rag env; RAG_LLM_BASE_URL + .env key):
  RAG_LLM_BASE_URL=http://localhost:1234/v1 RAG_HYDE_MODEL=gemma-4-e4b-it \
    conda run -n rag --no-capture-output python scripts/eval/gen_hyde.py \
    --queries eval/out/queries.jsonl --out eval/out/hyde_queries.jsonl \
    | tee eval/out/gen_hyde.log
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from dotenv import load_dotenv; load_dotenv()
except ImportError:
    pass

from src.llm.client import make_client, default_model
from src.query.hyde import generate_hypothetical


def run(queries_path, out_path, model, anchored=False):
    client = make_client()
    print(f"hyde model: {model} (anchored={anchored})", flush=True)
    rows = [json.loads(l) for l in open(queries_path) if l.strip()]
    empty = 0
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as out:
        for i, q in enumerate(rows, 1):
            query = q["query"]
            try:
                hyp = generate_hypothetical(client, model, query, anchored=anchored)
            except Exception as e:  # noqa: BLE001 - keep going, record empty
                print(f"  hyde error: {e}", flush=True)
                hyp = ""
            if not hyp.strip():
                empty += 1
            out.write(json.dumps({"query": query, "hypothetical": hyp}) + "\n")
            if i % 25 == 0:
                print(f"  generated {i}/{len(rows)}", flush=True)
    print(f"wrote {len(rows)} hypotheticals ({empty} empty) -> {out_path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", default="eval/out/queries.jsonl")
    ap.add_argument("--out", default="eval/out/hyde_queries.jsonl")
    ap.add_argument("--anchored", action="store_true",
                    help="preserve query anchors / invent nothing (EXPERIMENTS §12)")
    args = ap.parse_args()
    model = os.getenv("RAG_HYDE_MODEL", "").strip() or default_model()
    run(args.queries, args.out, model, anchored=args.anchored)


if __name__ == "__main__":
    main()
