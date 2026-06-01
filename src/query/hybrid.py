# src/query/hybrid.py
"""Build a LlamaIndex hybrid (dense+sparse RRF) searcher over an existing bge-m3
Qdrant collection, with an opt-in cross-encoder reranker.

Framework-native: QdrantVectorStore (hybrid) + node-postprocessor reranker. The
only custom pieces are the bge-m3 adapters (src/query/bge_m3_embedding.py) and the
RRF callback (src/query/fusion.py). FlagEmbedding/reranker imports are lazy so this
module imports cleanly in the unit-test env.
"""
import os
from typing import List

from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore

from src.query.bge_m3_embedding import (
    BgeM3LlamaIndexEmbedding,
    make_bge_m3_sparse_query_fn,
)
from src.query.fusion import reciprocal_rank_fusion
from src.query.summary_rerank import make_summary_reranker
from src.query.thread_expand import assemble_threads

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"
DEFAULT_RERANK_MODEL = "BAAI/bge-reranker-v2-m3"


def _qdrant_client():
    """Construct a Qdrant client from environment variables (collection passed separately).

    Reads ``os.environ`` directly rather than importing ``src.config.settings``: that
    module eagerly imports heavy cloud llama-index integrations (OpenAI/Perplexity) which
    the lightweight local ``rag`` query env does not (and should not, per the offline goal)
    have. ``QDRANT_URL`` is required; api-key/grpc are optional with local-friendly defaults.
    """
    from qdrant_client import QdrantClient

    url = (os.environ.get("QDRANT_URL") or "").strip()
    if not url:
        raise ValueError("QDRANT_URL environment variable is not set")
    api_key = (os.environ.get("QDRANT_API_KEY") or "").strip() or None
    grpc_raw = os.environ.get("QDRANT_PREFER_GRPC") or ""
    prefer_grpc = grpc_raw.strip().lower() in {"1", "true", "yes", "on"}
    return QdrantClient(url=url, api_key=api_key, prefer_grpc=prefer_grpc)


def _make_reranker(model: str = DEFAULT_RERANK_MODEL, top_n: int = 5, use_fp16: bool = True):
    """Lazy-construct the cross-encoder reranker (heavy deps; patchable in tests)."""
    from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker

    return FlagEmbeddingReranker(model=model, top_n=top_n, use_fp16=use_fp16)


class HybridSearcher:
    """Runs the retriever, then (optionally) the reranker postprocessor."""

    def __init__(self, retriever, reranker=None, *, client=None, collection=None):
        self._retriever = retriever
        self._reranker = reranker
        self._client = client
        self._collection = collection

    def search(self, query: str) -> List:
        nodes = self._retriever.retrieve(query)
        if self._reranker is not None:
            nodes = self._reranker.postprocess_nodes(nodes, query_str=query)
        return nodes

    def search_threads(self, query: str):
        """Search, then expand the hits into attributed ThreadContexts."""
        if self._client is None or self._collection is None:
            raise ValueError(
                "search_threads requires a Qdrant client and collection"
            )
        nodes = self.search(query)
        return assemble_threads(nodes, self._client, self._collection)


def build_hybrid_searcher(
    collection: str,
    *,
    client=None,
    embedder=None,
    mode: str = "hybrid",
    rerank: bool = False,
    rerank_with_summary: bool = False,
    fusion_fn=None,
    dense_top_k: int = 20,
    sparse_top_k: int = 20,
    top_n: int = 5,
) -> HybridSearcher:
    """Wire a HybridSearcher over an existing bge-m3 collection.

    mode="dense" -> dense-only baseline; mode="sparse" -> sparse-only; mode="hybrid" -> dense+sparse RRF.
    rerank=True attaches the cross-encoder reranker (top_n results).
    rerank_with_summary=True scores the cross-encoder on summary+body (takes precedence over rerank).
    fusion_fn overrides the hybrid fusion callback (default RRF; see make_rank_fusion).
    `client`/`embedder` are injectable for testing; built lazily otherwise.
    """
    if client is None:
        client = _qdrant_client()
    if embedder is None:
        from src.ingest.embedder import BgeM3Embedder

        embedder = BgeM3Embedder()

    # One bge-m3 sparse encoder for both query AND doc roles. We never index docs via
    # this query path, but QdrantVectorStore's constructor eagerly builds a *default*
    # sparse_doc_fn (fastembed/SPLADE) when one isn't supplied — which both pulls an
    # unwanted dep and would be the wrong vocabulary. Passing our bge-m3 fn avoids that.
    sparse_fn = make_bge_m3_sparse_query_fn(embedder)
    if fusion_fn is None:
        fusion_fn = reciprocal_rank_fusion
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection,
        enable_hybrid=True,
        dense_vector_name=DENSE_VECTOR_NAME,
        sparse_vector_name=SPARSE_VECTOR_NAME,
        sparse_query_fn=sparse_fn,
        sparse_doc_fn=sparse_fn,
        hybrid_fusion_fn=fusion_fn,
    )
    index = VectorStoreIndex.from_vector_store(
        vector_store, embed_model=BgeM3LlamaIndexEmbedding(embedder=embedder)
    )
    query_mode = {"hybrid": "hybrid", "sparse": "sparse"}.get(mode, "default")
    # sparse_top_k is forwarded in both modes; LlamaIndex ignores it in "default" (dense) mode.
    retriever = index.as_retriever(
        vector_store_query_mode=query_mode,
        similarity_top_k=dense_top_k,
        sparse_top_k=sparse_top_k,
    )
    if rerank_with_summary:
        reranker = make_summary_reranker(top_n=top_n)
    elif rerank:
        reranker = _make_reranker(top_n=top_n)
    else:
        reranker = None
    return HybridSearcher(retriever, reranker, client=client, collection=collection)
