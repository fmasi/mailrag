"""Email thread reconstruction from RFC 5322 headers.

A reply carries ``In-Reply-To`` (its parent's Message-ID) and ``References``
(the Message-IDs of the whole ancestor chain, root first). We derive a stable
``thread_id`` so every message in a conversation groups together:

    1. the root of ``References`` (first id) when present,
    2. else ``In-Reply-To`` (the immediate parent),
    3. else the message's own ``Message-ID`` (it is itself a thread root).

For header-less datasets (e.g. the public HF Enron corpus which carries no
Message-ID/In-Reply-To/References) a subject-slug fallback groups emails by
normalised subject when all three headers are absent.  Existing 3-argument
call-sites are unaffected because ``subject`` defaults to ``""`` and the
fallback only triggers when the header-based key would be empty.

This is a pragmatic, dependency-free grouping key. It is not the full JWZ
graph-resolution algorithm, but in practice every message in a thread shares
the same References root, so it groups conversations reliably.
"""

import re


def normalize_message_id(value) -> str:
    """Return a Message-ID with surrounding ``< >`` and whitespace stripped."""
    if not value:
        return ""
    v = value.strip()
    if v.startswith("<") and v.endswith(">"):
        v = v[1:-1]
    return v.strip()


def subject_slug(subject: str) -> str:
    """Normalize a subject into a thread key.

    Strips Re:/Fwd:/Fw: prefixes (any nesting depth), lowercases, and
    collapses internal whitespace.  Returns ``""`` for empty / None input.
    """
    s = (subject or "").strip()
    while True:
        m = re.match(r'^\s*(re|fwd|fw)\s*:\s*', s, flags=re.IGNORECASE)
        if not m:
            break
        s = s[m.end():]
    return re.sub(r'\s+', ' ', s).strip().lower()


def compute_thread_id(
    message_id: str = "",
    in_reply_to: str = "",
    references: str = "",
    subject: str = "",
) -> str:
    """Derive a stable thread id from the threading headers (see module docs).

    When all three RFC 5322 headers are absent (``message_id``, ``in_reply_to``,
    and ``references`` are all empty) **and** a ``subject`` is supplied, the
    function falls back to a ``subj:<slug>`` key so that header-less datasets
    such as the public HF Enron corpus still form meaningful thread groups.

    Existing 3-argument callers are unaffected: ``subject`` defaults to ``""``
    so the fallback never triggers for them.
    """
    refs = (references or "").split()
    if refs:
        root = refs[0]
    elif in_reply_to:
        parts = in_reply_to.split()
        root = parts[0] if parts else ""
    else:
        root = message_id or ""
    key = normalize_message_id(root)
    if not key and subject:
        slug = subject_slug(subject)
        if slug:
            return f"subj:{slug}"
    return key


def assign_subject_fallback_thread_ids(emails):
    """Set a subject-slug ``thread_id`` on every email that lacks one.

    Header-less corpora (e.g. the public HF Enron export) have no RFC threading
    headers, so without this every email lands in one empty-key bucket. Setting
    the id up front makes summary grouping, chunk persistence, and retrieval-time
    assembly share one key. Returns the count assigned. Tolerates frozen/mocked
    objects (skips them silently)."""
    n = 0
    for e in emails:
        if getattr(e, "thread_id", None):
            continue
        tid = compute_thread_id(
            getattr(e, "message_id", "") or "",
            getattr(e, "in_reply_to", "") or "",
            getattr(e, "references", "") or "",
            subject=getattr(e, "subject", "") or "",
        )
        if tid:
            try:
                e.thread_id = tid
                n += 1
            except (AttributeError, TypeError):
                pass
    return n
