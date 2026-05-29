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
