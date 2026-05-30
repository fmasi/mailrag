# scripts/eval/e2e_judge.py
"""End-to-end eval, step 3 (local): grade each (query, setup) ANSWER against the
gold answer email with the local judge model, 0-3. Reuses the contexts.jsonl for
gold_text and an answers_<tag>.json for the answers.

Output: eval/out/e2e/grades_<tag>.json = {query: {setup: grade}} + a per-setup
mean-grade summary to stdout. Real content -> eval/out (gitignored).

Run on the HOST (rag env; RAG_LLM_BASE_URL + .env key; strong judge model):
  RAG_LLM_BASE_URL=http://localhost:1234/v1 RAG_JUDGE_MODEL=gemma-4-26b-a4b-it-mlx@8bit \
    conda run -n rag --no-capture-output python scripts/eval/e2e_judge.py \
    --contexts eval/out/e2e/contexts_v2.jsonl --answers eval/out/e2e/answers_pm.json \
    --tag pm | tee eval/out/e2e_judge_pm.log
"""
import argparse
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from dotenv import load_dotenv; load_dotenv()
except ImportError:
    pass

from src.llm.client import make_client, default_model, chat
from src.eval.judge_parse import build_answer_judge_prompt, parse_grade


def run(contexts_path, answers_path, out_path, model):
    client = make_client()
    print(f"judge model: {model}", flush=True)
    gold = {}
    for line in open(contexts_path):
        if line.strip():
            r = json.loads(line)
            gold[r["query"]] = r.get("gold_text", "")
    answers = json.load(open(answers_path))

    grades = {}
    per_setup = collections.defaultdict(list)
    for query, by_setup in answers.items():
        grades[query] = {}
        ref = gold.get(query, "")
        for setup, answer in by_setup.items():
            try:
                raw = chat(client, model, build_answer_judge_prompt(query, answer, ref))
                g = parse_grade(raw)
            except Exception as e:  # noqa: BLE001
                print(f"  judge error ({setup}): {e}", flush=True)
                g = 0
            grades[query][setup] = g
            per_setup[setup].append(g)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    json.dump(grades, open(out_path, "w"), indent=2)
    print("\n=== mean answer grade per setup ===", flush=True)
    for setup, gs in per_setup.items():
        print(f"  {setup:18s} {sum(gs)/len(gs):.2f}  (n={len(gs)})", flush=True)
    print(f"wrote grades -> {out_path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", default="eval/out/e2e/contexts.jsonl")
    ap.add_argument("--answers", required=True)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    model = os.getenv("RAG_JUDGE_MODEL", "").strip() or default_model()
    run(args.contexts, args.answers, f"eval/out/e2e/grades_{args.tag}.json", model)


if __name__ == "__main__":
    main()
