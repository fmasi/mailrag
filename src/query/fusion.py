"""Reciprocal Rank Fusion (RRF) as a LlamaIndex ``hybrid_fusion_fn`` callback.

LlamaIndex's QdrantVectorStore ships only ``relative_score_fusion``; RRF is the
portable, score-normalization-free default (it fuses *ranks*, not scores), so we
supply it via the framework's documented ``hybrid_fusion_fn`` hook. This is a
single callback, not a homegrown retrieval abstraction.

RRF score for a document = sum over each result list of 1 / (k + rank), where
rank is 0-based position in that list. ``alpha`` is accepted for signature
compatibility but unused (RRF is rank-based, not score-weighted).
"""
from typing import Dict

from llama_index.core.schema import BaseNode
from llama_index.core.vector_stores.types import VectorStoreQueryResult


def reciprocal_rank_fusion(
    dense_result: VectorStoreQueryResult,
    sparse_result: VectorStoreQueryResult,
    alpha: float = 0.5,
    top_k: int = 2,
    k: int = 60,
) -> VectorStoreQueryResult:
    scores: Dict[str, float] = {}
    node_by_id: Dict[str, BaseNode] = {}
    for result in (dense_result, sparse_result):
        ids = result.ids or []
        nodes = result.nodes or []
        for rank, _id in enumerate(ids):
            scores[_id] = scores.get(_id, 0.0) + 1.0 / (k + rank + 1)
            if rank < len(nodes):
                node_by_id.setdefault(_id, nodes[rank])
    # Keep ids/similarities/nodes strictly parallel: drop any id whose node is
    # absent (e.g. a sparse-only leg that returned ids without nodes), then take
    # the top_k, so consumers can safely zip the three lists.
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    fused = [(i, s) for i, s in ranked if i in node_by_id][:top_k]
    fused_ids = [i for i, _ in fused]
    fused_sims = [s for _, s in fused]
    fused_nodes = [node_by_id[i] for i in fused_ids]
    return VectorStoreQueryResult(nodes=fused_nodes, similarities=fused_sims, ids=fused_ids)
