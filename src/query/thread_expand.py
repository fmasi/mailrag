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
