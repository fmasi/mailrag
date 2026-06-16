# scripts/eval/judge.py
"""Judge pool emails for relevance with the local Gemma judge model.

Default: grade all (query, email) pairs in pool.jsonl -> grades.json
  { "<query>": { "<message_id>": grade, ... }, ... }
--dump-calib N: instead, write the pairs for the first N queries to
  calib_pairs.jsonl (one {query, message_id, email_text} per line) for the
  reference (Opus) judge to grade with the SAME rubric.

Run on the HOST (rag env; RAG_LLM_API_BASE + .env key; pick the strong judge model):
  RAG_LLM_API_BASE=http://localhost:1234/v1 RAG_JUDGE_MODEL=gemma-4-31b-it-mlx \
    conda run -n rag --no-capture-output \
    python scripts/eval/judge.py --pool eval/out/pool.jsonl --out eval/out/grades.json \
    | tee eval/out/judge.log
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
from src.eval.judge_parse import build_judge_prompt, parse_grade


def _email_text(hit):
    return f"Subject: {hit.get('subject','')}\n\n{hit.get('body','')}".strip()


def _load_pool(path):
    with open(path) as fh:
        return [json.loads(l) for l in fh if l.strip()]


def dump_calib(pool_rows, n, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    written = 0
    with open(out_path, "w") as fh:
        for row in pool_rows[:n]:
            for hit in row["pool"]:
                fh.write(json.dumps({"query": row["query"],
                                     "message_id": hit["message_id"],
                                     "email_text": _email_text(hit)}) + "\n")
                written += 1
    print(f"wrote {written} calib pairs -> {out_path}", flush=True)


def run(pool_rows, out_path, model):
    client = make_client()
    print(f"judge model: {model}", flush=True)
    grades = {}
    total = sum(len(r["pool"]) for r in pool_rows)
    done = 0
    for row in pool_rows:
        q = row["query"]
        grades[q] = {}
        for hit in row["pool"]:
            prompt = build_judge_prompt(q, _email_text(hit))
            try:
                grades[q][hit["message_id"]] = parse_grade(chat(client, model, prompt))
            except Exception as e:  # noqa: BLE001
                print(f"  judge error (default 0): {e}", flush=True)
                grades[q][hit["message_id"]] = 0
            done += 1
            if done % 25 == 0:
                print(f"  judged {done}/{total}", flush=True)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(grades, fh, indent=2)
    print(f"wrote grades for {len(grades)} queries -> {out_path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="eval/out/pool.jsonl")
    ap.add_argument("--out", default="eval/out/grades.json")
    ap.add_argument("--dump-calib", type=int, default=0,
                    help="if >0, dump first-N-queries' pairs for the reference judge instead of grading")
    ap.add_argument("--calib-out", default="eval/out/calib_pairs.jsonl")
    args = ap.parse_args()
    pool_rows = _load_pool(args.pool)
    if args.dump_calib > 0:
        dump_calib(pool_rows, args.dump_calib, args.calib_out)
        return
    model = os.getenv("RAG_JUDGE_MODEL", "").strip() or default_model()
    run(pool_rows, args.out, model)


if __name__ == "__main__":
    main()
