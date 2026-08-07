"""Qdrant hybrid (dense + sparse) collection management and upsert.

The hybrid collection stores two named vectors per chunk: ``dense``
(bge-m3 1024-d, cosine) and ``sparse`` (bge-m3 lexical weights). Integration
component — requires qdrant-client + a running Qdrant.
"""

from typing import Iterable, List

from qdrant_client import QdrantClient, models

DENSE = "dense"
SPARSE = "sparse"

# Max message_keys per delete filter, so a large delta cannot build a filter the
# server rejects for size.
_DELETE_BATCH = 256


def get_client(url: str = "http://localhost:6333", api_key: str = "") -> QdrantClient:
    """Build a Qdrant client via the shared seam (``src/config/qdrant.py``)."""
    from src.config.qdrant import get_qdrant_client

    return get_qdrant_client(url=url, api_key=api_key)


# Payload indexes so filtering is fast/allowed: recipients & sender (full-text,
# for "emails involving person X"), subject (full-text), and exact-match keys
# (thread = whole conversation, source, date). Shared by both collection shapes.
def _payload_indexes():
    return [
        ("sender", models.PayloadSchemaType.TEXT),
        ("to_full", models.PayloadSchemaType.TEXT),
        ("cc_full", models.PayloadSchemaType.TEXT),
        ("subject", models.PayloadSchemaType.TEXT),
        ("thread_id", models.PayloadSchemaType.KEYWORD),
        ("source", models.PayloadSchemaType.KEYWORD),
        ("date", models.PayloadSchemaType.KEYWORD),
        # Attachment lineage (issue #80): filter "chunks from attachments",
        # "chunks of this email's files", and full-text on the filename.
        ("content_kind", models.PayloadSchemaType.KEYWORD),
        ("parent_message_id", models.PayloadSchemaType.KEYWORD),
        ("attachment_name", models.PayloadSchemaType.TEXT),
        # Incremental indexing (issue #101): message_key is the stable per-email
        # identity shared by an email's body and attachment chunks — the filter
        # delete_by_message_keys() deletes on. content_hash lets a delta run ask
        # "do I already hold this exact chunk?" without re-embedding it.
        ("message_key", models.PayloadSchemaType.KEYWORD),
        ("content_hash", models.PayloadSchemaType.KEYWORD),
        # The preprocessing/chunking rules a point was produced under, so an
        # incremental run can refuse to mix policies (src/indexing/policy.py).
        ("policy_fingerprint", models.PayloadSchemaType.KEYWORD),
    ]


def _create_payload_indexes(client: QdrantClient, name: str) -> None:
    for field_name, schema in _payload_indexes():
        client.create_payload_index(
            collection_name=name, field_name=field_name, field_schema=schema
        )


def ensure_payload_indexes(client: QdrantClient, name: str) -> None:
    """Create any missing payload indexes on an **existing** collection.

    ``ensure_*_collection`` only indexes at creation time, so a collection built
    before a field was introduced (e.g. ``message_key``) would silently reject the
    filter that incremental indexing depends on. Creating an index that already
    exists is a no-op server-side; failures on individual fields are swallowed so
    one unsupported field cannot abort a build.
    """
    for field_name, schema in _payload_indexes():
        try:
            client.create_payload_index(
                collection_name=name, field_name=field_name, field_schema=schema
            )
        except Exception:  # noqa: BLE001 — index already present / not supported
            pass


def ensure_hybrid_collection(
    client: QdrantClient, name: str, dim: int = 1024, recreate: bool = False
) -> None:
    """Create the dense+sparse collection if missing (or recreate it)."""
    if recreate and client.collection_exists(name):
        client.delete_collection(name)
    if not client.collection_exists(name):
        client.create_collection(
            collection_name=name,
            vectors_config={DENSE: models.VectorParams(size=dim, distance=models.Distance.COSINE)},
            sparse_vectors_config={SPARSE: models.SparseVectorParams()},
        )
        _create_payload_indexes(client, name)


def ensure_dense_collection(
    client: QdrantClient, name: str, dim: int = 1024, recreate: bool = False
) -> None:
    """Create a dense-only collection (no sparse leg) if missing (or recreate it).

    For dense-only embedders (e.g. a NVIDIA NIM, ``produces_sparse=False``) whose
    endpoint cannot emit learned sparse weights.
    """
    if recreate and client.collection_exists(name):
        client.delete_collection(name)
    if not client.collection_exists(name):
        client.create_collection(
            collection_name=name,
            vectors_config={DENSE: models.VectorParams(size=dim, distance=models.Distance.COSINE)},
        )
        _create_payload_indexes(client, name)


def make_point(point_id, dense_vec, sparse_indices, sparse_values, payload) -> models.PointStruct:
    return models.PointStruct(
        id=point_id,
        vector={
            DENSE: [float(x) for x in dense_vec],
            SPARSE: models.SparseVector(indices=sparse_indices, values=sparse_values),
        },
        payload=payload,
    )


def make_dense_point(point_id, dense_vec, payload) -> models.PointStruct:
    """A point with only the dense named vector (for dense-only collections)."""
    return models.PointStruct(
        id=point_id,
        vector={DENSE: [float(x) for x in dense_vec]},
        payload=payload,
    )


def upsert(client: QdrantClient, name: str, points: List[models.PointStruct]) -> None:
    client.upsert(collection_name=name, points=points)


def has_legacy_points(client: QdrantClient, name: str, sample: int = 64) -> bool:
    """True when *name* holds points written before deterministic ids (issue #101).

    Points indexed by the old code carry random per-run UUIDs and no
    ``message_key``, so an incremental run cannot address them: the delete filter
    matches nothing and the upsert writes a second, differently-keyed copy of
    every chunk. Silently doubling a 20,000-email collection is the worst possible
    failure here, so the append path calls this and refuses.

    Sampling a page is enough — legacy collections are wholly legacy, since the
    field arrived with the id change. A missing/empty collection is not legacy.
    """
    try:
        points, _ = client.scroll(
            collection_name=name, limit=sample, with_payload=True, with_vectors=False
        )
    except Exception:  # noqa: BLE001 — collection absent or unreachable
        return False
    return any("message_key" not in (getattr(p, "payload", None) or {}) for p in points)


def collection_policy(client: QdrantClient, name: str) -> str:
    """Return the index-policy fingerprint *name* was built under, or "".

    Sampling one point is enough: a collection is only ever written by runs that
    agreed on the policy, because this is the check that enforces it. An empty
    string means "unknown" — an empty collection, an unreachable server, or one
    built before fingerprints existed — and callers must treat that as "no
    objection" rather than as a mismatch.
    """
    try:
        points, _ = client.scroll(
            collection_name=name, limit=1, with_payload=True, with_vectors=False
        )
    except Exception:  # noqa: BLE001 — collection absent or unreachable
        return ""
    for p in points or []:
        return str((getattr(p, "payload", None) or {}).get("policy_fingerprint") or "")
    return ""


def delete_by_message_keys(client: QdrantClient, name: str, keys: Iterable[str]) -> int:
    """Delete every point belonging to the given emails. Returns the number of keys.

    The first half of the incremental **delete-then-upsert** protocol (issue #101).
    Deterministic point ids alone make re-upserting a chunk idempotent, but they
    cannot remove a chunk that no longer exists — re-processing an email into
    *fewer* chunks would otherwise leave a stale tail behind. Deleting the email's
    points first makes the upsert a true replacement.

    Filtering on ``message_key`` catches an email's body **and** attachment chunks
    in one call, since both carry the same key.

    No-ops on an empty key set (an unfiltered delete here would wipe the
    collection). Batched so a large delta cannot build an oversized filter.
    """
    keys = [k for k in dict.fromkeys(keys) if k]
    if not keys:
        return 0
    for i in range(0, len(keys), _DELETE_BATCH):
        batch = keys[i : i + _DELETE_BATCH]
        client.delete(
            collection_name=name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(key="message_key", match=models.MatchAny(any=batch))
                    ]
                )
            ),
        )
    return len(keys)
