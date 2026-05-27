# src/ingest/embed_text.py
"""Assemble the text that gets embedded for a chunk.

Contextual retrieval: optionally prepend the email's one-line Pass-2 summary to
the chunk text before embedding, so terse/reply chunks ("Approved, go ahead.")
gain a topical, entity-rich vector (and the summary's terms also strengthen the
sparse/lexical side). Opt-in via the build's ``--embed-summary`` flag; the
default build leaves the embedded text unchanged.
"""
from __future__ import annotations

from typing import Optional


def prepend_summary(text: str, summary: Optional[str]) -> str:
    """Return ``summary + '\\n\\n' + text`` when a non-empty summary is given,
    else *text* unchanged."""
    if summary and summary.strip():
        return f"{summary.strip()}\n\n{text}"
    return text
