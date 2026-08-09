# src/query/thread_expand.py
"""Thread-aware retrieval: expand reranked hits into full attributed email threads.

Reads the existing hybrid collection (no re-embedding).
"""

from __future__ import annotations

from dataclasses import dataclass
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
        must=[models.FieldCondition(key="thread_id", match=models.MatchAny(any=list(thread_ids)))]
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
        lines.append(f"[{_short_date(e.date)}] From: {e.sender}  To: {e.to}  Cc: {cc or _EMDASH}")
        lines.append(f"  {e.body.strip()}")
        lines.append("")
    return "\n".join(lines).strip()


def build_thread_contexts(client, collection: str, thread_ids: List[str]) -> List[ThreadContext]:
    """Build ordered, attributed ThreadContexts for exactly these thread_ids.

    A pure **key lookup**: the ids are fetched by an exact payload filter (see
    :func:`fetch_thread_payloads`), never by similarity. Returned in the order
    the ids were given; ids with no points are skipped rather than erroring, so
    a partially-stale id list degrades to fewer threads instead of failing.

    Split out of :func:`assemble_threads` so callers that already KNOW the
    thread_id can reach it without a retrieval round-trip — see issue #109,
    where resolving a known id by re-running search found it only ~25% of the
    time.
    """
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
        contexts.append(
            ThreadContext(
                thread_id=tid,
                subject=emails[0].subject,
                emails=emails,
                text=text,
            )
        )
    return contexts


def assemble_threads(nodes, client, collection: str) -> List[ThreadContext]:
    """Expand retrieved nodes into ordered, attributed ThreadContexts.

    Fetches all points per thread_id once, then builds one ThreadContext per
    thread_id (in hit order). Search is used only to DISCOVER which threads are
    relevant; turning those ids into threads is the exact lookup in
    :func:`build_thread_contexts`.
    """
    return build_thread_contexts(client, collection, extract_thread_ids(nodes))


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars/token) — dependency-free heuristic."""
    return len(text) // 4


def bound_thread(
    ctx: ThreadContext,
    max_tokens: int,
    summarizer: Optional[Callable[[str], str]] = None,
) -> ThreadContext:
    """Keep an assembled thread within `max_tokens`. No-op when under budget.

    Over budget: keep the root (first) and the most-recent email verbatim. The
    middle is either summarized (if a `summarizer` callback is supplied) or
    elided with a marker. Off by default at the call site (max_tokens unset).
    """
    if estimate_tokens(ctx.text) <= max_tokens or len(ctx.emails) <= 2:
        return ctx

    root, latest = ctx.emails[0], ctx.emails[-1]
    middle = ctx.emails[1:-1]
    middle_text = render_thread(ctx.thread_id, middle)

    if summarizer is not None:
        middle_block = summarizer(middle_text)
    else:
        middle_block = f"[{len(middle)} earlier emails omitted]"

    head = render_thread(ctx.thread_id, [root])
    tail = render_thread(ctx.thread_id, [latest])
    # tail's leading "[Thread: ...]" header is redundant after head; drop it.
    tail_body = tail.split("\n", 2)[-1] if "\n" in tail else tail
    text = "\n\n".join([head, middle_block, tail_body]).strip()
    return ThreadContext(
        thread_id=ctx.thread_id,
        subject=ctx.subject,
        emails=ctx.emails,
        text=text,
        bounded=True,
    )


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
        emails.append(
            ThreadEmail(
                message_id=mid,
                sender=head.get("sender", ""),
                to=head.get("to", ""),
                cc=head.get("cc", ""),
                date=head.get("date", ""),
                subject=head.get("subject", ""),
                body=body,
                summary=head.get("summary", ""),
            )
        )
    return emails
