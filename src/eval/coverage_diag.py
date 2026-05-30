# src/eval/coverage_diag.py
"""Pure logic for the retrieval-coverage miss diagnostic (issue #12).

Rank-finding + miss-classification over ranked hit lists. Qdrant-free: operates on
plain dicts ({"thread_id", "message_id"}) so it unit-tests in the lightweight env.
The live driver is scripts/eval/diagnose_coverage.py. See
docs/superpowers/specs/2026-05-30-coverage-miss-diagnostic-design.md.
"""
from __future__ import annotations

import re
from typing import List, Optional


def best_gold_rank(hits: List[dict], gold_thread_id: str, gold_message_id: str) -> dict:
    """Node-level ranks (0-based) of the gold thread and gold email in a ranked list.

    `hits` is rank-ordered; each item has "thread_id" and "message_id". Returns the
    index of the first hit belonging to the gold thread ("thread_rank") and of the
    first hit that IS the gold email ("email_rank"); None when absent.
    """
    thread_rank: Optional[int] = None
    email_rank: Optional[int] = None
    for i, h in enumerate(hits):
        if thread_rank is None and h.get("thread_id") == gold_thread_id:
            thread_rank = i
        if email_rank is None and h.get("message_id") == gold_message_id:
            email_rank = i
        if thread_rank is not None and email_rank is not None:
            break
    return {"thread_rank": thread_rank, "email_rank": email_rank}


def distinct_thread_rank(hits: List[dict], gold_thread_id: str) -> Optional[int]:
    """Position of the gold thread among DISTINCT thread_ids in hit order (0-based).

    Mirrors thread expansion: the Nth distinct thread_id becomes the Nth expanded
    thread, so this is the budget-coverage coordinate. None if never seen.
    """
    seen: List[str] = []
    for h in hits:
        tid = h.get("thread_id")
        if tid and tid not in seen:
            seen.append(tid)
            if tid == gold_thread_id:
                return len(seen) - 1
    return None


_TERSE_MAX_CHARS = 80
_WORD = re.compile(r"[a-z0-9]+")


def is_terse(body: str, max_chars: int = _TERSE_MAX_CHARS) -> bool:
    """True when the email body is empty or shorter than `max_chars` after strip."""
    return len((body or "").strip()) < max_chars


def _tokens(text: str) -> set:
    return set(_WORD.findall((text or "").lower()))


def lexical_overlap(query: str, text: str) -> float:
    """Fraction of distinct query tokens present in `text` (0..1); 0 if query empty."""
    q = _tokens(query)
    if not q:
        return 0.0
    return len(q & _tokens(text)) / len(q)
