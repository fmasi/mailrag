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

SUMMARY_EMBED_HEADROOM = 256  # extra token budget reserved for a prepended summary
# so it augments the body chunk rather than displacing its tail at encode time (EXPERIMENTS §4.7).


def embed_max_length(
    chunk_size: int,
    embed_summary: bool,
    override: Optional[int] = None,
    headroom: int = SUMMARY_EMBED_HEADROOM,
) -> int:
    """Token ceiling for bge-m3 encoding.

    Default is ``chunk_size`` (body-only builds, unchanged). When a summary is
    prepended (``embed_summary``), reserve ``headroom`` extra tokens so the
    summary does NOT truncate the body chunk (fixes the §4.7 displacement
    confound). An explicit ``override`` (e.g. a --embed-max-length CLI value)
    always wins. Result is clamped to bge-m3's 8192 ceiling.
    """
    if override is not None:
        return min(override, 8192)
    base = chunk_size + headroom if embed_summary else chunk_size
    return min(base, 8192)


def prepend_summary(text: str, summary: Optional[str]) -> str:
    """Return ``summary + '\\n\\n' + text`` when a non-empty summary is given,
    else *text* unchanged."""
    if summary and summary.strip():
        return f"{summary.strip()}\n\n{text}"
    return text
