# src/eval/query_validator.py
"""Validator gate for synthetic eval queries (issue #18).

A second-pass check that a generated query is a GOOD eval question: answerable by a
specific content fact in the answer email, and NOT an artifact/meta question about the
email/thread/meeting itself. Pure logic (prompt build + verdict parse) lives here so it
is unit-testable; the LLM call is in scripts/eval/gen_queries.py. Mirrors judge_parse.
"""

from __future__ import annotations

import json


def build_validation_prompt(query: str, thread_text: str, answer_body: str) -> str:
    """Build the STRICT-JSON validator prompt embedding all three inputs."""
    return (
        "You are vetting a candidate question for a retrieval-eval set.\n"
        "ACCEPT only if BOTH hold:\n"
        "  1. The question is answerable by a SPECIFIC CONTENT FACT stated in the ANSWER "
        "EMAIL.\n"
        "  2. The question is about the CONTENT, NOT about the email/thread/meeting as an "
        "artifact (its subject line, who is on it, how many messages, when it was sent).\n"
        "Reject vague, meta, or unanswerable questions.\n\n"
        f"QUESTION:\n{query}\n\n"
        f"ANSWER EMAIL:\n{answer_body}\n\n"
        f"THREAD CONTEXT:\n{thread_text}\n\n"
        'Return STRICT JSON: {"keep": true|false, "reason": "<short>"}'
    )


def parse_validation(raw: str) -> dict:
    """Parse the validator's JSON verdict; fail-closed (keep=False) on any ambiguity."""
    if not raw:
        return {"keep": False, "reason": "empty output"}
    s = raw.strip().lstrip("`")
    start, end = s.find("{"), s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {"keep": False, "reason": "no json object"}
    try:
        obj = json.loads(s[start : end + 1])
    except ValueError:
        return {"keep": False, "reason": "unparseable json"}
    keep = obj.get("keep")
    if not isinstance(keep, bool):
        return {"keep": False, "reason": "missing/invalid keep"}
    return {"keep": keep, "reason": str(obj.get("reason") or "")}
