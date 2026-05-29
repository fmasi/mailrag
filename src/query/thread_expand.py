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


_EMDASH = "—"


def _short_date(date: str) -> str:
    """'2015-01-08T16:05:00+00:00' -> '2015-01-08 16:05'; pass through if odd."""
    if not date or date == "unknown":
        return "unknown"
    d = date.replace("T", " ")
    return d[:16]


def render_thread(thread_id: str, emails: List[ThreadEmail]) -> str:
    """Render emails as an attributed, chronological thread block for the LLM.

    Each email gets an explicit From/To/Cc/Date header because the vector store
    excludes to/cc from the default LLM metadata block (src/data/models.py).
    """
    subject = emails[0].subject if emails else thread_id
    lines = [f"[Thread: {subject}]", ""]
    for e in emails:
        cc = e.cc.strip() if e.cc else ""
        lines.append(
            f"[{_short_date(e.date)}] From: {e.sender}  To: {e.to}  "
            f"Cc: {cc or _EMDASH}"
        )
        lines.append(f"  {e.body.strip()}")
        lines.append("")
    return "\n".join(lines).strip()


def assemble_threads(nodes, client, collection: str) -> List[ThreadContext]:
    """Expand retrieved nodes into ordered, attributed ThreadContexts.

    Fetches all points per thread_id once, then builds one ThreadContext per
    thread_id (in hit order).
    """
    thread_ids = extract_thread_ids(nodes)
    if not thread_ids:
        return []
    payloads = fetch_thread_payloads(client, collection, thread_ids)

    by_tid: dict[str, List[dict]] = {tid: [] for tid in thread_ids}
    for p in payloads:
        tid = p.get("thread_id")
        if tid in by_tid:
            by_tid[tid].append(p)

    contexts: List[ThreadContext] = []
    for tid in thread_ids:
        emails = order_by_date(group_into_emails(by_tid[tid]))
        if not emails:
            continue
        text = render_thread(tid, emails)
        contexts.append(ThreadContext(
            thread_id=tid, subject=emails[0].subject, emails=emails, text=text,
        ))
    return contexts


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
