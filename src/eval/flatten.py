# src/eval/flatten.py
"""Flatten any retrieval arm's output to a ranked, deduped list of emails.

All arms (chunk/email retrievers and the thread-aware expander) are reduced to a
common unit — one EmailHit per message_id, in rank order — so a single judge
label per (query, message_id) scores every arm fairly. Dependency-free: operates
on duck-typed objects so it imports in the lightweight test env.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class EmailHit:
    message_id: str
    subject: str
    body: str
    summary: str = ""


def _meta(node) -> dict:
    inner = getattr(node, "node", node)
    return getattr(inner, "metadata", None) or getattr(node, "metadata", None) or {}


def _content(node) -> str:
    inner = getattr(node, "node", node)
    getter = getattr(inner, "get_content", None)
    if callable(getter):
        try:
            return getter() or ""
        except TypeError:
            return getter("") or ""
    return getattr(inner, "text", "") or ""


def flatten_nodes(nodes) -> List[EmailHit]:
    """Email-arm nodes → ranked deduped EmailHits (first occurrence wins)."""
    seen = set()
    out: List[EmailHit] = []
    for node in nodes:
        md = _meta(node)
        mid = md.get("message_id")
        if not mid or mid in seen:
            continue
        seen.add(mid)
        out.append(EmailHit(
            message_id=mid,
            subject=md.get("subject") or "",
            body=_content(node),
            summary=md.get("summary") or "",
        ))
    return out


def flatten_threads(contexts) -> List[EmailHit]:
    """Thread-aware contexts → ranked deduped EmailHits (thread order, threads in hit order)."""
    seen = set()
    out: List[EmailHit] = []
    for ctx in contexts:
        for e in ctx.emails:
            if not e.message_id or e.message_id in seen:
                continue
            seen.add(e.message_id)
            out.append(EmailHit(
                message_id=e.message_id,
                subject=e.subject or "",
                body=e.body or "",
                summary=getattr(e, "summary", "") or "",
            ))
    return out
