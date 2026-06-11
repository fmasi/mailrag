"""Qdrant hybrid (dense + sparse) collection management and upsert.

The hybrid collection stores two named vectors per chunk: ``dense``
(bge-m3 1024-d, cosine) and ``sparse`` (bge-m3 lexical weights). Integration
component — requires qdrant-client + a running Qdrant.
"""
from typing import List

from qdrant_client import QdrantClient, models

DENSE = "dense"
SPARSE = "sparse"


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
    ]


def _create_payload_indexes(client: QdrantClient, name: str) -> None:
    for field_name, schema in _payload_indexes():
        client.create_payload_index(collection_name=name, field_name=field_name, field_schema=schema)


def ensure_hybrid_collection(client: QdrantClient, name: str, dim: int = 1024, recreate: bool = False) -> None:
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


def ensure_dense_collection(client: QdrantClient, name: str, dim: int = 1024, recreate: bool = False) -> None:
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
