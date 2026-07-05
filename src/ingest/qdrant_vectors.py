"""Read dense vectors from a Qdrant collection and mean-pool per thread.

Used by the `explore` verb to reuse already-embedded vectors when the profile's
collection exists. Integration-adjacent (needs a live Qdrant in real use) but the
pure pooling logic is unit-tested with a fake client.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from src.ingest.hybrid_qdrant import DENSE


def _dense(vector):
    if isinstance(vector, dict):
        return vector.get(DENSE) or vector.get("dense")
    return vector


def read_thread_vectors(client, collection: str, *, batch: int = 256) -> Dict[str, np.ndarray]:
    """Scroll all points (dense vector + thread_id) and return
    ``{thread_id: mean_dense_vector}``. Points lacking a thread_id are skipped."""
    sums: Dict[str, np.ndarray] = {}
    counts: Dict[str, int] = {}
    offset = None
    while True:
        records, offset = client.scroll(
            collection_name=collection,
            with_vectors=[DENSE],
            with_payload=["thread_id"],
            limit=batch,
            offset=offset,
        )
        for rec in records:
            tid = (rec.payload or {}).get("thread_id")
            dv = _dense(rec.vector)
            if not tid or dv is None:
                continue
            v = np.asarray(dv, dtype=float)
            if tid in sums:
                sums[tid] += v
                counts[tid] += 1
            else:
                sums[tid] = v.copy()
                counts[tid] = 1
        if offset is None:
            break
    return {tid: sums[tid] / counts[tid] for tid in sums}
