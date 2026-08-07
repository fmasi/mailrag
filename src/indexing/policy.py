"""Index policy fingerprint: refuse to mix incompatible vectors in one collection.

Before incremental indexing existed, every build recreated its collection, so
changing the body cleanup or the chunk size simply rebuilt everything and the
question never arose. Appending changed that: an incremental run after a policy
change quietly writes vectors produced under *new* rules alongside vectors
produced under the old ones, in the same collection, with nothing to indicate
it. Retrieval still "works" — it just silently ranks two differently-preprocessed
populations against each other.

So each point carries a short fingerprint of the rules that produced it, and an
incremental run checks the collection's existing fingerprint before writing. A
mismatch stops the run and asks for one ``--recreate``.

The versions below are **manual**: bump them when a change would alter the
vectors produced from identical input. That is a judgement call the code cannot
make for itself, which is exactly why it is a constant with a comment rather
than a hash of the source.

----

Attribution
    The generation-fingerprint idea — versioning the preprocessing/chunk policy
    so a change stales the index and forces a rebuild instead of mixing layouts
    — is taken from **msgvault** (https://github.com/kenn-io/msgvault),
    ``internal/vector/config.go`` (``preprocessVersion`` / ``embedPolicyVersion``
    folded into ``GenerationFingerprint``), MIT licensed, Copyright (c)
    2025-2026 Wes McKinney. See the repository ``NOTICE`` file.
"""

from __future__ import annotations

import hashlib
from typing import Optional

# Bump when body cleanup changes what text reaches the embedder:
#   1 — initial: HTML→text, reply-chain stripping, calendar collapse
#   2 — added base64/data-URI stripping, URL tracking-param stripping,
#       signature-block stripping, whitespace normalization
#       (src/data/body_cleanup.py)
#   3 — signature stripping made conservative and idempotent: it now fires only
#       when there is exactly one "-- " delimiter and the removed block is
#       signature-shaped. Version 2 bodies were cleaned under the old rule, so
#       they must not share a fingerprint with these.
#   4 — signature delimiter widened to any trailing horizontal whitespace and
#       the whitespace trim moved before it (the "--  " case made cleaning
#       non-idempotent); "ref" removed from the tracking-parameter list; subject
#       and sender caps halved for token headroom.
PREPROCESS_VERSION = 4

# Bump when the chunk layout changes shape for identical preprocessed text —
# a different splitter, a change to how attachments are split, or a change to
# how chunk ordinals are assigned (which would move every point id).
#   1 — SentenceSplitter over bodies + structure-aware attachment chunking,
#       deterministic uuid5 point ids
CHUNK_POLICY_VERSION = 1


def _as_int(value) -> int:
    """Coerce to int, or 0. A duck-typed embedder may expose a ``dim`` that is
    not a number, and a fingerprint helper must never be the thing that fails a
    build — an unknown value that hashes consistently is fine."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def policy_fingerprint(
    *,
    chunk_size: int,
    chunk_overlap: int,
    embed_summary: bool,
    embedder_name: str = "",
    dim: int = 0,
) -> str:
    """A short, stable hash of everything that determines a point's vector.

    Deliberately includes the *tunable* parameters as well as the version
    constants: re-indexing at ``chunk_size=1024`` into a collection built at
    512 produces exactly the same silent mixing, and is a far easier mistake to
    make than editing the preprocessing code.
    """
    parts = [
        f"preprocess={PREPROCESS_VERSION}",
        f"chunk_policy={CHUNK_POLICY_VERSION}",
        f"chunk_size={_as_int(chunk_size)}",
        f"chunk_overlap={_as_int(chunk_overlap)}",
        f"embed_summary={bool(embed_summary)}",
        f"embedder={embedder_name or 'unknown'}",
        f"dim={_as_int(dim)}",
    ]
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]


def describe_mismatch(collection: str, existing: Optional[str], incoming: str) -> str:
    """The error text shown when an append would mix policies.

    Names both fingerprints and the fix, because the failure is otherwise
    baffling — nothing is broken, the run simply must not proceed.
    """
    return (
        f"collection '{collection}' was built under index policy {existing!r}, but this "
        f"run produces {incoming!r}. Appending would mix vectors made under different "
        "preprocessing/chunking rules in one collection, which silently degrades "
        "ranking. Rebuild once with --recreate, or point this run at the settings the "
        "collection was built with (chunk_size / chunk_overlap / --embed-summary / "
        "embedder)."
    )
