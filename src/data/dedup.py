"""Exact-content deduplication for chunks/nodes.

Boilerplate — email signatures, confidentiality/legal disclaimers, newsletter
and digest footers — repeats verbatim across many emails. Dropping
exact-duplicate chunk text before embedding avoids spending vectors (and query
noise) on identical content.

Stdlib-only so it is host-testable; the indexing path calls it on the parsed
nodes just before embedding.
"""

import hashlib


def dedup_by_content(items, key):
    """Return *items* with later exact-content duplicates removed (first kept).

    *key* maps an item to the string that identifies a duplicate; surrounding
    whitespace is ignored so trivial trailing newlines don't defeat the match.
    Order of first occurrence is preserved.
    """
    seen = set()
    kept = []
    for item in items:
        digest = hashlib.sha256(key(item).strip().encode("utf-8")).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        kept.append(item)
    return kept
