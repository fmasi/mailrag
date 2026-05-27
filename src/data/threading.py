"""Email thread reconstruction from RFC 5322 headers.

A reply carries ``In-Reply-To`` (its parent's Message-ID) and ``References``
(the Message-IDs of the whole ancestor chain, root first). We derive a stable
``thread_id`` so every message in a conversation groups together:

    1. the root of ``References`` (first id) when present,
    2. else ``In-Reply-To`` (the immediate parent),
    3. else the message's own ``Message-ID`` (it is itself a thread root).

This is a pragmatic, dependency-free grouping key. It is not the full JWZ
graph-resolution algorithm, but in practice every message in a thread shares
the same References root, so it groups conversations reliably.
"""


def normalize_message_id(value) -> str:
    """Return a Message-ID with surrounding ``< >`` and whitespace stripped."""
    if not value:
        return ""
    v = value.strip()
    if v.startswith("<") and v.endswith(">"):
        v = v[1:-1]
    return v.strip()


def compute_thread_id(message_id: str = "", in_reply_to: str = "", references: str = "") -> str:
    """Derive a stable thread id from the threading headers (see module docs)."""
    refs = (references or "").split()
    if refs:
        root = refs[0]
    elif in_reply_to:
        parts = in_reply_to.split()
        root = parts[0] if parts else ""
    else:
        root = message_id or ""
    return normalize_message_id(root)
