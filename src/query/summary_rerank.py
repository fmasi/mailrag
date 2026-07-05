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


class SummaryAwareReranker:
    """Cross-encoder reranker scoring on summary+body. Matches the node-postprocessor
    interface: ``postprocess_nodes(nodes, query_str=...)``. FlagReranker import is lazy;
    pass ``_reranker`` to inject a stub in tests."""

    def __init__(
        self,
        model: str = DEFAULT_RERANK_MODEL,
        top_n: int = 5,
        use_fp16: bool = True,
        _reranker=None,
    ):
        self._top_n = top_n
        if _reranker is not None:
            self._reranker = _reranker
        else:
            from FlagEmbedding import FlagReranker

            self._reranker = FlagReranker(model, use_fp16=use_fp16)

    def postprocess_nodes(self, nodes, query_str: str = None, query_bundle=None):
        if not nodes:
            return []
        q = (
            query_str
            if query_str is not None
            else (query_bundle.query_str if query_bundle is not None else "")
        )
        pairs = [[q, build_rerank_text(n)] for n in nodes]
        scores = self._reranker.compute_score(pairs)
        if not isinstance(scores, list):
            scores = [scores]
        return rerank_by_scores(nodes, scores, self._top_n)


def make_summary_reranker(model: str = DEFAULT_RERANK_MODEL, top_n: int = 5, use_fp16: bool = True):
    """Lazy factory (patchable in tests)."""
    return SummaryAwareReranker(model=model, top_n=top_n, use_fp16=use_fp16)
