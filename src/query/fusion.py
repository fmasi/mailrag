"""Reciprocal Rank Fusion (RRF) as a LlamaIndex ``hybrid_fusion_fn`` callback.

LlamaIndex's QdrantVectorStore ships only ``relative_score_fusion``; RRF is the
portable, score-normalization-free default (it fuses *ranks*, not scores), so we
supply it via the framework's documented ``hybrid_fusion_fn`` hook. This is a
single callback, not a homegrown retrieval abstraction.

RRF score for a document = sum over each result list of 1 / (k + rank), where
rank is 0-based position in that list. ``alpha`` is accepted for signature
compatibility but unused (RRF is rank-based, not score-weighted).
"""

import math
from typing import Dict, List

from llama_index.core.schema import BaseNode
from llama_index.core.vector_stores.types import VectorStoreQueryResult


def _rank_fusion(
    dense_result: VectorStoreQueryResult,
    sparse_result: VectorStoreQueryResult,
    top_k: int,
    k: int,
    p: float,
    sparse_weight: float = 1.0,
) -> VectorStoreQueryResult:
    """Fuse two ranked lists by the power-mean of per-list RRF terms 1/(k+rank).

    p=1 -> arithmetic sum (classic RRF); p->inf -> CombMAX (per-doc best term),
    with the sum of terms as a deterministic tiebreak. ``sparse_weight`` scales the
    sparse list's terms (dense weight fixed at 1.0): >1 favours keyword/sparse hits,
    countering the densification of summary-embedded docs (EXPERIMENTS §13). Inputs
    are rank scores, so no score normalization is needed. ids/similarities/nodes stay
    strictly parallel (ids whose node is absent are dropped), then the top_k returned.
    """
    terms: Dict[str, List[float]] = {}
    node_by_id: Dict[str, BaseNode] = {}
    for result, weight in ((dense_result, 1.0), (sparse_result, sparse_weight)):
        ids = result.ids or []
        nodes = result.nodes or []
        for rank, _id in enumerate(ids):
            terms.setdefault(_id, []).append(weight * (1.0 / (k + rank + 1)))
            if rank < len(nodes):
                node_by_id.setdefault(_id, nodes[rank])

    def score(_id: str) -> float:
        xs = terms[_id]
        if math.isinf(p):
            return max(xs)
        return math.fsum(x**p for x in xs) ** (1.0 / p)

    # Sort by (combined score, sum-of-terms tiebreak), descending. terms.keys()
    # preserves first-seen order, so equal keys keep a stable order (matches the
    # original RRF at p=1).
    ranked = sorted(terms.keys(), key=lambda i: (score(i), math.fsum(terms[i])), reverse=True)
    fused_ids = [i for i in ranked if i in node_by_id][:top_k]
    fused_sims = [score(i) for i in fused_ids]
    fused_nodes = [node_by_id[i] for i in fused_ids]
    return VectorStoreQueryResult(nodes=fused_nodes, similarities=fused_sims, ids=fused_ids)


def reciprocal_rank_fusion(
    dense_result: VectorStoreQueryResult,
    sparse_result: VectorStoreQueryResult,
    alpha: float = 0.5,
    top_k: int = 2,
    k: int = 60,
) -> VectorStoreQueryResult:
    """Classic RRF (sum of 1/(k+rank)) as a LlamaIndex hybrid_fusion_fn callback.

    The p=1 special case of _rank_fusion. `alpha` is accepted for signature
    compatibility but unused (RRF is rank-based, not score-weighted).
    """
    return _rank_fusion(dense_result, sparse_result, top_k=top_k, k=k, p=1.0)


def make_rank_fusion(p: float = 1.0, k: int = 60, sparse_weight: float = 1.0):
    """Build a hybrid_fusion_fn that fuses by power-mean with exponent `p` (>=1).

    p=1 == reciprocal_rank_fusion; p=inf == CombMAX. p<1 is rejected (it rewards
    agreement — the wrong direction for un-burying single-modality top hits).
    sparse_weight (>=0) scales the sparse list's RRF terms; >1 favours keyword hits.
    """
    if p < 1:
        raise ValueError("p must be >= 1 (p<1 rewards agreement, the wrong direction)")
    if sparse_weight < 0:
        raise ValueError("sparse_weight must be >= 0")

    def fusion_fn(dense_result, sparse_result, alpha: float = 0.5, top_k: int = 2, k: int = k):
        return _rank_fusion(
            dense_result, sparse_result, top_k=top_k, k=k, p=p, sparse_weight=sparse_weight
        )

    return fusion_fn
