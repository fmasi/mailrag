"""Deterministic Qdrant point IDs, so re-indexing is idempotent (issue #101).

Until now every chunk was upserted under its LlamaIndex ``node_id`` — a **random
UUID minted per run**. Two consequences followed:

* re-running a build without ``recreate=True`` duplicated every chunk, so
* every build path had to pass ``recreate=True`` and drop the whole collection.

That makes "add the 40 emails that arrived since Tuesday" impossible, which is the
blocker for continuous sync. Here the id is derived from *what the chunk is*
instead of *when it was written*::

    point_id = uuid5(MAILRAG_NAMESPACE, f"{doc_key}:{chunk_index}")

``doc_key`` is the stable per-document key set by the producers
(``NormalizedEmail.to_document`` for bodies, ``build_attachment_documents`` for
attachments) and carried as the LlamaIndex ``doc_id``; ``chunk_index`` is the
chunk's ordinal **within its own document**, so re-chunking the same body yields
the same ids regardless of what else is in the corpus.

Ids are assigned *before* the corpus-wide content dedup pass so that whether some
other email happened to contain an identical chunk cannot shift the ids of the
chunks that survive.
"""

from __future__ import annotations

import hashlib
import uuid
from typing import Iterable, List

# A fixed, arbitrary v4 UUID used as the uuid5 namespace. Changing it changes
# every point id in every collection — treat it as a wire format constant.
MAILRAG_NAMESPACE = uuid.UUID("6f1d2a3e-9c47-4b8a-9f1e-2d5c7a840b13")


def point_id(doc_key: str, chunk_index: int) -> str:
    """Return the deterministic point id for chunk *chunk_index* of *doc_key*.

    Same inputs always produce the same id; different inputs practically never
    collide (uuid5 is a namespaced sha1).
    """
    return str(uuid.uuid5(MAILRAG_NAMESPACE, f"{doc_key}:{chunk_index}"))


def content_hash(text: str) -> str:
    """Stable sha256 of a chunk's exact text.

    Stored as a payload field so a future incremental run can cheaply ask Qdrant
    "do I already hold this exact chunk?" without re-embedding it, and so the
    corpus-order-dependent in-run dedup (``dedup_by_content``) has a durable
    counterpart in the index.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _doc_key_of(node) -> str:
    """Best-effort stable document key for *node*.

    Normally the ``ref_doc_id`` set by the splitter from the parent Document's
    ``doc_id``. A node with no source relationship (hand-built in tests, or a
    Document indexed without splitting) falls back to its own ``node_id`` — still
    deterministic when the caller supplied a stable one.
    """
    return getattr(node, "ref_doc_id", None) or getattr(node, "node_id", "") or ""


def assign_deterministic_ids(nodes: Iterable) -> List:
    """Rewrite each node's ``id_`` to its deterministic point id, in place.

    Call this immediately after splitting and **before** any dedup/filtering, so
    the ordinal of a chunk within its document is a property of that document
    alone. Returns the same list for chaining.
    """
    out: List = list(nodes)
    seen: dict[str, int] = {}
    for n in out:
        key = _doc_key_of(n)
        idx = seen.get(key, 0)
        seen[key] = idx + 1
        n.id_ = point_id(key, idx)
    return out
