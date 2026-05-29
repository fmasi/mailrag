# scripts/eval/e2e_answer.py
"""End-to-end eval, step 2: generate an ANSWER per (query, setup) with a local
answer model, from the cached context. Run once per answer model (reuses the
model-independent contexts.jsonl).

Output: eval/out/e2e/answers_<modeltag>.json = {query: {setup: answer_text}}.
Real content -> eval/out (gitignored).

Run on the HOST (rag env; RAG_LLM_BASE_URL + .env key):
  RAG_LLM_BASE_URL=http://localhost:1234/v1 RAG_ANSWER_MODEL=gemma-4-31b-it-mlx \
    conda run -n rag --no-capture-output \
    python scripts/eval/e2e_answer.py --contexts eval/out/e2e/contexts.jsonl \
    --tag gemma31b | tee eval/out/e2e_answer_gemma31b.log
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

from src.llm.client import make_client, default_model, chat

PROMPT = (
    "You answer a question using ONLY the emails provided as context. "
    "If the context does not contain enough information to answer, reply exactly: "
    "\"I don't know based on the provided emails.\" Be concise and factual.\n\n"
    "QUESTION:\n{query}\n\nCONTEXT EMAILS:\n{context}\n\nANSWER:"
)


def run(contexts_path, out_path, model):
    client = make_client()
    print(f"answer model: {model}", flush=True)
    rows = [json.loads(l) for l in open(contexts_path) if l.strip()]
    answers = {}
    total = sum(len(r["contexts"]) for r in rows)
    done = 0
    for r in rows:
        q = r["query"]
        answers[q] = {}
        for setup, ctx in r["contexts"].items():
            prompt = PROMPT.format(query=q, context=ctx if ctx else "(no emails provided)")
            try:
                answers[q][setup] = chat(client, model, prompt)
            except Exception as e:  # noqa: BLE001
                print(f"  answer error (setup={setup}): {e}", flush=True)
                answers[q][setup] = ""
            done += 1
            if done % 25 == 0:
                print(f"  generated {done}/{total}", flush=True)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(answers, fh, indent=2)
    print(f"wrote answers for {len(answers)} queries -> {out_path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", default="eval/out/e2e/contexts.jsonl")
    ap.add_argument("--tag", required=True, help="short model tag for the output filename")
    args = ap.parse_args()
    model = os.getenv("RAG_ANSWER_MODEL", "").strip() or default_model()
    run(args.contexts, f"eval/out/e2e/answers_{args.tag}.json", model)


if __name__ == "__main__":
    main()
