# scripts/eval/gen_queries.py
"""Generate synthetic-from-corpus eval queries with the local generator model.

For each sampled thread, prompt the model to emit a natural user question whose
answer lives in that thread + the message_id of the answer email. Category mix is
hypothesis-weighted. Generated from thread BODY content only (leakage guard). Output
(real content) -> eval/out (gitignored).

Run on the HOST (rag env; QDRANT_URL + RAG_LLM_* set; .env loaded for keys):
  QDRANT_URL=http://localhost:6333 RAG_LLM_BASE_URL=http://localhost:1234/v1 \
    conda run -n rag --no-capture-output \
    python scripts/eval/gen_queries.py --collection work-rag \
    --n-terse 48 --n-content 48 --n-spanning 24 --out eval/out/queries.jsonl
"""
import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from dotenv import load_dotenv; load_dotenv()
except ImportError:
    pass

from src.query.hybrid import _qdrant_client
from src.query.thread_expand import group_into_emails, render_thread
from src.llm.client import make_client, default_model, chat
from src.eval.query_validator import build_validation_prompt, parse_validation

_PAGE = 512

# category -> instruction nudging the generator toward that query shape
_CATEGORY_PROMPT = {
    "terse": "The answer should hinge on a short/terse reply in the thread (an "
             "acknowledgement, confirmation, or one-line decision). Ask who said what or "
             "what was confirmed.",
    "content": "Ask a specific factual/technical question answered by the substantive "
               "content of the thread. Prefer exact terms, names, or acronyms used.",
    "spanning": "Ask a question whose answer must be assembled across several emails in "
                "the thread (e.g. what was decided and when, how a plan evolved).",
}

# Appended to every category instruction: force content questions, ban artifact/meta ones.
_ARTIFACT_RULE = (
    " The answer MUST be a specific content fact stated in the chosen answer email. "
    "NEVER ask about the email/thread/meeting as an artifact (its subject line, who is on "
    "it, how many messages it has, or when it was sent)."
)


def _sample_threads(client, collection, k, min_emails, max_emails, seed):
    """Collect threads with email-count in [min,max]; return up to k (seeded shuffle)."""
    by_tid = {}
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection, limit=_PAGE,
            with_payload=True, with_vectors=False, offset=offset)
        for p in points:
            tid = (p.payload or {}).get("thread_id")
            if tid:
                by_tid.setdefault(tid, []).append(p.payload)
        if offset is None:
            break
    rng = random.Random(seed)
    items = []
    for tid, payloads in by_tid.items():
        emails = group_into_emails(payloads)
        if min_emails <= len(emails) <= max_emails:
            items.append((tid, emails))
    rng.shuffle(items)
    return items[:k]


def _gen_one(client, model, category, tid, emails):
    body = render_thread(tid, emails)
    mids = [e.message_id for e in emails]
    prompt = (
        "You are building a search-eval question from an email thread.\n"
        f"{_CATEGORY_PROMPT[category]}{_ARTIFACT_RULE}\n\n"
        "Return STRICT JSON: {\"query\": <user question>, \"answer_message_id\": <one of "
        f"these ids: {mids}>}}. Use ONLY information present below.\n\nTHREAD:\n{body}"
    )
    raw = chat(client, model, prompt)
    raw = raw.strip().lstrip("`")
    start, end = raw.find("{"), raw.rfind("}")
    obj = json.loads(raw[start:end + 1])
    q = (obj.get("query") or "").strip()
    amid = (obj.get("answer_message_id") or "").strip()
    if not q or amid not in mids:
        return None
    return {"query": q, "category": category, "thread_id": tid, "answer_message_id": amid}


def _validate_one(client, model, row, emails):
    """LLM validator gate: True iff the candidate query is a good content question.

    Builds the validation prompt from the thread text + the answer email's body, calls
    the model, and returns parse_validation's verdict dict {"keep", "reason"}.
    """
    thread_text = render_thread(row["thread_id"], emails)
    answer_body = next(
        (e.body for e in emails if e.message_id == row["answer_message_id"]), "")
    raw = chat(client, model, build_validation_prompt(
        row["query"], thread_text, answer_body))
    return parse_validation(raw)


def run(collection, counts, out_path, seed):
    client = _qdrant_client()
    llm = make_client()
    model = os.getenv("RAG_GEN_MODEL", "").strip() or default_model()
    print(f"generator model: {model}", flush=True)
    vmodel = os.getenv("RAG_VALIDATE_MODEL", "").strip() or model
    print(f"validator model: {vmodel}", flush=True)
    rows = []
    for category, k in counts.items():
        threads = _sample_threads(client, collection, k * 5, 3, 25, seed + hash(category) % 1000)
        made = 0
        for tid, emails in threads:
            if made >= k:
                break
            try:
                row = _gen_one(llm, model, category, tid, emails)
            except Exception as e:  # noqa: BLE001 - skip a bad generation, keep going
                print(f"  skip ({category}): {e}", flush=True)
                row = None
            if not row:
                continue
            try:
                verdict = _validate_one(llm, vmodel, row, emails)
            except Exception as e:  # noqa: BLE001 - skip on validator error, keep going
                print(f"  validate-skip ({category}): {e}", flush=True)
                continue
            if not verdict["keep"]:
                print(f"  reject ({category}): {verdict['reason'][:60]}", flush=True)
                continue
            rows.append(row); made += 1
            print(f"  [{category}] {made}/{k}", flush=True)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} queries -> {out_path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", default="work-rag")
    ap.add_argument("--n-terse", type=int, default=48)
    ap.add_argument("--n-content", type=int, default=48)
    ap.add_argument("--n-spanning", type=int, default=24)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="eval/out/queries.jsonl")
    args = ap.parse_args()
    counts = {"terse": args.n_terse, "content": args.n_content, "spanning": args.n_spanning}
    run(args.collection, counts, args.out, args.seed)


if __name__ == "__main__":
    main()
