# src/data/identity.py
"""Content-based email identity.

The Pass-2 cache is keyed primarily by the sha256 of the raw ``.eml`` bytes,
which is brittle against re-export: re-exporting the same mailbox can change
the mbox preamble, header order, line endings, or MIME encoding, yielding a
different file hash for an unchanged email and forcing the expensive LLM sweep
to rerun.

``content_sha256`` derives a stable identifier from the *parsed* email fields
(normalized whitespace and line endings, canonical date), so it survives that
reformatting.  ``Message-ID`` is deliberately NOT folded in: it is stored as a
separate fallback key, keeping the two signals independent.
"""
from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Optional, Union


def _norm_inline(s: Optional[str]) -> str:
    """Collapse all whitespace runs to single spaces and strip the ends."""
    return " ".join((s or "").split())


def _norm_body(body: Optional[str]) -> str:
    """Normalize line endings, strip per-line trailing whitespace, strip ends."""
    text = (body or "").replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in text.split("\n")]
    return "\n".join(lines).strip()


def content_sha256(*, sender: str = "", subject: str = "",
                   date: Union[datetime, str, None] = None,
                   body: str = "") -> str:
    """Stable hex sha256 over the normalized content of an email.

    ``date`` may be a ``datetime`` (canonicalized via ``isoformat``) or an
    already-formatted string; both yield the same digest for the same instant.
    """
    date_s = date.isoformat() if isinstance(date, datetime) else (date or "")
    parts = [_norm_inline(sender), _norm_inline(subject), date_s, _norm_body(body)]
    return hashlib.sha256("\x00".join(parts).encode("utf-8")).hexdigest()


def email_identity(*, sender: str = "", subject: str = "",
                   date: Union[datetime, str, None] = None, body: str = "",
                   message_id: str = "") -> "tuple[Optional[str], str]":
    """Return ``(normalized Message-ID or None, content_sha256)`` for an email.

    The single place both the run (write) and build (read) paths derive the
    cache's stable fallback identifiers, so they always agree.
    """
    from src.data.threading import normalize_message_id
    mid = normalize_message_id(message_id or "") or None
    return mid, content_sha256(sender=sender, subject=subject, date=date, body=body)
