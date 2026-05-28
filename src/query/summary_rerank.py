"""Summary-aware cross-encoder reranking.

Scores the reranker on ``summary + body`` instead of body alone, so the summary's
context informs RANKING (precision) without being embedded into the vector (which
causes drift). ``SummaryAwareReranker`` exposes ``postprocess_nodes(nodes, query_str)``
matching LlamaIndex node postprocessors, so HybridSearcher uses it interchangeably with
FlagEmbeddingReranker. FlagReranker import is lazy.
"""
from typing import List

DEFAULT_RERANK_MODEL = "BAAI/bge-reranker-v2-m3"


def build_rerank_text(node) -> str:
    """Text the cross-encoder reads: 'summary\\n\\nbody' when a summary is present, else body."""
    summary = (node.metadata.get("summary") or "").strip()
    body = node.get_content()
    return f"{summary}\n\n{body}" if summary else body


def rerank_by_scores(nodes: List, scores: List[float], top_n: int) -> List:
    """Attach scores, sort descending, return top_n. Pure (no model)."""
    paired = sorted(zip(nodes, scores), key=lambda ns: ns[1], reverse=True)
    out = []
    for node, score in paired[:top_n]:
        node.score = float(score)
        out.append(node)
    return out
