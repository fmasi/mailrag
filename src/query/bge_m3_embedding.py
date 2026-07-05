"""Adapt our FlagEmbedding bge-m3 encoder to LlamaIndex query-time interfaces.

Two adapters:
- ``BgeM3LlamaIndexEmbedding`` — a ``BaseEmbedding`` that produces bge-m3 DENSE
  vectors for the hybrid query's dense leg, guaranteeing parity with the dense
  vectors the build pipeline stored (do NOT use HuggingFaceEmbedding("BAAI/bge-m3"):
  its pooling/normalization can drift from FlagEmbedding's output).
- ``make_bge_m3_sparse_query_fn`` — a ``sparse_query_fn`` that encodes the query's
  bge-m3 learned-sparse vector, so it lives in the same token-id vocabulary as the
  stored sparse vectors (bypasses LlamaIndex's default fastembed/SPLADE encoder).

The underlying ``BgeM3Embedder`` lazily imports FlagEmbedding only on first encode,
so importing this module is safe in the (FlagEmbedding-free) unit-test env.
"""

from typing import TYPE_CHECKING, Callable, List, Tuple

from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.embeddings import BaseEmbedding

from src.ingest.sparse import lexical_weights_to_sparse

if TYPE_CHECKING:
    from src.ingest.embedder import BgeM3Embedder


class BgeM3LlamaIndexEmbedding(BaseEmbedding):
    _embedder = PrivateAttr()

    def __init__(self, embedder: "BgeM3Embedder", model_name: str = "BAAI/bge-m3", **kwargs):
        super().__init__(model_name=model_name, **kwargs)
        self._embedder = embedder

    def _dense(self, texts: List[str]) -> List[List[float]]:
        """Return dense bge-m3 vectors as Python floats, shape [N, 1024]."""
        dense_vecs, _ = self._embedder.encode(texts)
        return [[float(x) for x in row] for row in dense_vecs]

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._dense([query])[0]

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._dense([text])[0]

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return self._dense(texts)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return self._dense(texts)


SparseEncoderCallable = Callable[[List[str]], Tuple[List[List[int]], List[List[float]]]]


def make_bge_m3_sparse_query_fn(embedder: "BgeM3Embedder") -> SparseEncoderCallable:
    """Return a LlamaIndex sparse_query_fn backed by bge-m3 lexical weights."""

    def _fn(texts: List[str]) -> Tuple[List[List[int]], List[List[float]]]:
        _, lexical_weights = embedder.encode(texts)
        indices: List[List[int]] = []
        values: List[List[float]] = []
        for weights in lexical_weights:
            idx, val = lexical_weights_to_sparse(weights)
            indices.append(idx)
            values.append(val)
        return indices, values

    return _fn
