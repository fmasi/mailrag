"""Where a collection's attachment store lives on disk.

One resolver, used by everything that reads or writes a store: the MCP server,
the ``attachments`` verbs and continuous sync. They have to agree, or sync
fills one directory while the server reads another.
"""

from __future__ import annotations

import os
import re
from typing import Optional

DEFAULT_ATTACH_STORE = "~/.mailrag/attachments"  # expanded by AttachmentStore


def safe_dirname(collection: str) -> str:
    """Reduce a collection name to one safe path segment.

    Collection names come from config, so a name like ``../other`` must not be
    able to walk out of the store root.
    """
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", collection).strip("-.") or "default"
    return cleaned[:120]


def resolve_attach_store(store: Optional[str] = None, collection: Optional[str] = None) -> str:
    """Resolve the attachment store for one collection.

    Precedence: explicit ``store`` > ``$RAG_ATTACH_STORE`` / ``~/.mailrag/attachments``,
    **plus the collection name as a subdirectory**.

    Stores are physically separate per collection rather than one store filtered
    by a predicate. A shared store leaks in practice, not just in theory: with a
    single index, four thread ids existed in both a work and a personal corpus,
    so listing either returned the other's attachments — and ``get_attachment``
    accepted any sha256 from any corpus with nothing to scope it. Filtering
    would fix both, right up until the first query that forgets the predicate.
    Separate directories cannot be un-separated by a missing ``WHERE`` clause,
    which is the property worth having when the two corpora are someone's
    employer and their private life.
    """
    if store:
        return store
    root = os.environ.get("RAG_ATTACH_STORE") or DEFAULT_ATTACH_STORE
    if not collection:
        return root
    return os.path.join(root, safe_dirname(collection))
