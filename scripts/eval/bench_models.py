# scripts/eval/bench_models.py
"""Micro-benchmark a loaded LM Studio model: tokens/sec + time-to-first-token on a
SHORT and a LONG prompt. Uses LM Studio's native /api/v0 endpoint (returns a `stats`
block). Prints only timing numbers — no generated content.

Run (model must already be loaded):
  RAG_LLM_API_KEY=... python scripts/eval/bench_models.py --model <id> --label <tag>
"""
import argparse
import json
import os
import sys
import urllib.request

try:
    from dotenv import load_dotenv; load_dotenv()
except ImportError:
    pass

BASE = os.getenv("RAG_LLM_BASE_URL", "http://localhost:1234/v1").replace("/v1", "")
KEY = os.getenv("RAG_LLM_API_KEY", "lm-studio")


def _ctx_prompts():
    """Pull a small (plain_C) and a large (C_thread_n3) real context as test prompts."""
    rows = [json.loads(l) for l in open("eval/out/e2e/contexts.jsonl") if l.strip()]
    r = rows[0]
    short = f"Answer concisely.\n\nQUESTION:\n{r['query']}\n\nCONTEXT:\n{r['contexts']['plain_C']}\n\nANSWER:"
    long = f"Answer concisely.\n\nQUESTION:\n{r['query']}\n\nCONTEXT:\n{r['contexts']['C_thread_n3']}\n\nANSWER:"
    return short, long


def _call(model, prompt):
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}],
                       "temperature": 0.0, "max_tokens": 200}).encode()
    req = urllib.request.Request(f"{BASE}/api/v0/chat/completions", data=body,
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {KEY}"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.load(resp)


def _stats(model, prompt):
    d = _call(model, prompt)
    s = d.get("stats", {}) or {}
    u = d.get("usage", {}) or {}
    return {
        "prompt_tokens": u.get("prompt_tokens"),
        "gen_tokens": s.get("predicted_tokens_count") or u.get("completion_tokens"),
        "tok_per_sec": s.get("tokens_per_second"),
        "ttft_s": s.get("time_to_first_token"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", default="")
    args = ap.parse_args()
    short, long = _ctx_prompts()
    print(f"=== {args.label or args.model} ===", flush=True)
    for tag, p in (("short", short), ("long", long)):
        try:
            st = _stats(args.model, p)
            print(f"  {tag:5} prompt_tok={st['prompt_tokens']:>6} gen_tok={st['gen_tokens']:>4} "
                  f"tok/s={st['tok_per_sec']:.1f} ttft={st['ttft_s']:.2f}s", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"  {tag:5} ERROR: {e}", flush=True)


if __name__ == "__main__":
    main()
