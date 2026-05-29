# src/query/thread_expand.py
"""Thread-aware retrieval: expand reranked hits into full attributed email threads.

Reads the existing hybrid collection (no re-embedding). See
docs/superpowers/specs/2026-05-29-thread-aware-retrieval-design.md.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional


@dataclass
class ThreadEmail:
    message_id: str
    sender: str
    to: str
    cc: str
    date: str
    subject: str
    body: str
    summary: str = ""


@dataclass
class ThreadContext:
    thread_id: str
    subject: str
    emails: List[ThreadEmail]
    text: str
    bounded: bool = False


def _node_metadata(node) -> dict:
    """Return metadata whether given a NodeWithScore or a bare TextNode."""
    inner = getattr(node, "node", node)
    return getattr(inner, "metadata", {}) or {}


def extract_thread_ids(nodes) -> List[str]:
    """Distinct thread_ids of the retrieved hits, in first-seen order."""
    seen: List[str] = []
    for node in nodes:
        tid = _node_metadata(node).get("thread_id")
        if tid and tid not in seen:
            seen.append(tid)
    return seen


_SCROLL_PAGE = 256


def fetch_thread_payloads(client, collection: str, thread_ids: List[str]) -> List[dict]:
    """Scroll every point whose thread_id is in `thread_ids`, return their payloads.

    Uses a MatchAny filter on the thread_id KEYWORD index. Paginates until the
    scroll cursor is exhausted.
    """
    if not thread_ids:
        return []
    from qdrant_client import models

    flt = models.Filter(
        must=[models.FieldCondition(
            key="thread_id", match=models.MatchAny(any=list(thread_ids))
        )]
    )
    payloads: List[dict] = []
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection,
            scroll_filter=flt,
            limit=_SCROLL_PAGE,
            with_payload=True,
            with_vectors=False,
            offset=offset,
        )
        payloads.extend(p.payload for p in points)
        if offset is None:
            break
    return payloads


def order_by_date(emails: List[ThreadEmail]) -> List[ThreadEmail]:
    """Chronological order; unparseable/unknown dates sort last (stable)."""
    def key(e: ThreadEmail):
        d = e.date or ""
        # ISO-8601 strings sort lexicographically; "unknown" / "" sort last.
        bad = not d or d == "unknown"
        return (bad, d)
    return sorted(emails, key=key)


def group_into_emails(payloads: List[dict]) -> List[ThreadEmail]:
    """Collapse chunk payloads into one ThreadEmail per message_id.

    93% of emails are single-chunk; the ~7% multi-chunk ones have their chunk
    `text` fields concatenated (best-effort order — no chunk_index field exists
    yet; see follow-up issue). Identity/metadata is taken from the first chunk.
    """
    by_mid: dict[str, List[dict]] = {}
    order: List[str] = []
    for p in payloads:
        mid = p.get("message_id") or ""
        if mid not in by_mid:
            by_mid[mid] = []
            order.append(mid)
        by_mid[mid].append(p)

    emails: List[ThreadEmail] = []
    for mid in order:
        chunks = by_mid[mid]
        head = chunks[0]
        body = "\n".join(c.get("text", "") for c in chunks).strip()
        emails.append(ThreadEmail(
            message_id=mid,
            sender=head.get("sender", ""),
            to=head.get("to", ""),
            cc=head.get("cc", ""),
            date=head.get("date", ""),
            subject=head.get("subject", ""),
            body=body,
            summary=head.get("summary", ""),
        ))
    return emails
