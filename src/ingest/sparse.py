"""Convert bge-m3 lexical weights into Qdrant sparse-vector format.

FlagEmbedding's ``lexical_weights`` is a dict mapping a token id (as a string)
to a positive weight. Qdrant sparse vectors are expressed as parallel
``indices`` (ints) and ``values`` (floats). Non-positive weights are dropped so
the vector stays genuinely sparse.
"""

from typing import Dict, List, Tuple


def lexical_weights_to_sparse(lexical_weights: Dict[str, float]) -> Tuple[List[int], List[float]]:
    """Return (indices, values) for a bge-m3 lexical-weight dict."""
    indices: List[int] = []
    values: List[float] = []
    for token_id, weight in lexical_weights.items():
        w = float(weight)
        if w > 0:
            indices.append(int(token_id))
            values.append(w)
    return indices, values
