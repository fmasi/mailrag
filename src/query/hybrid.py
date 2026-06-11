# src/query/hybrid.py
"""Build a LlamaIndex hybrid (dense+sparse RRF) searcher over an existing bge-m3
Qdrant collection, with an opt-in cross-encoder reranker.

Framework-native: QdrantVectorStore (hybrid) + node-postprocessor reranker. The
only custom pieces are the bge-m3 adapters (src/query/bge_m3_embedding.py) and the
RRF callback (src/query/fusion.py). FlagEmbedding/reranker imports are lazy so this
module imports cleanly in the unit-test env.
"""
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
NIM_RERANK_MODEL = "nvidia/rerank-qa-mistral-4b"      # the only hosted NVIDIA reranker
NIM_RERANK_BASE_URL = "https://ai.api.nvidia.com/v1"  # reranking is on a separate host


def _qdrant_client(qdrant_url=None):
    """Construct a Qdrant client via the shared seam (``src/config/qdrant.py``).

    ``qdrant_url`` is an explicit override (e.g. from a corpus profile) that takes
    precedence over the environment, so the host CLI can target ``localhost`` without
    inheriting a container-oriented ``QDRANT_URL`` from ``.env`` (issue #29). When
    unset, the seam falls back to ``$QDRANT_URL`` (required) plus the optional
    ``$QDRANT_API_KEY`` / ``$QDRANT_PREFER_GRPC``.

    The seam is lean (stdlib + a lazy ``qdrant_client`` import), so importing it keeps
    this offline query path free of the heavy llama-index integrations.
    """
    from src.config.qdrant import get_qdrant_client

    return get_qdrant_client(url=qdrant_url)


def _make_reranker(model: str = DEFAULT_RERANK_MODEL, top_n: int = 5, use_fp16: bool = True):
    """Lazy-construct the cross-encoder reranker (heavy deps; patchable in tests)."""
    from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker

    return FlagEmbeddingReranker(model=model, top_n=top_n, use_fp16=use_fp16)


def make_nim_reranker(model: str = NIM_RERANK_MODEL, top_n: int = 5,
                      api_key=None, base_url: str = NIM_RERANK_BASE_URL):
    """Lazy-construct the NVIDIA reranking-NIM node-postprocessor.

    The only hosted NVIDIA reranker is ``rerank-qa-mistral-4b`` on the separate
    ``ai.api.nvidia.com`` host (the integrate endpoint exposes no reranker). Reads
    the key from ``NVIDIA_API_KEY`` when not passed. Patchable in tests.
    """
    import os
    from llama_index.postprocessor.nvidia_rerank import NVIDIARerank

    key = api_key or os.environ.get("NVIDIA_API_KEY")
    return NVIDIARerank(model=model, base_url=base_url, api_key=key, top_n=top_n)


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
    qdrant_url=None,
    embedder=None,
    mode: str = "hybrid",
    rerank: bool = False,
    rerank_with_summary: bool = False,
    reranker=None,
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
    `qdrant_url` overrides where the lazily-built client connects (profile-driven,
    defaults to the environment); ignored when `client` is injected.
    """
    if client is None:
        client = _qdrant_client(qdrant_url)
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
    # An explicitly injected reranker (e.g. the NIM reranker) wins over the flags,
    # so a benchmark can hold ONE reranker constant across arms.
    if reranker is None:
        if rerank_with_summary:
            reranker = make_summary_reranker(top_n=top_n)
        elif rerank:
            reranker = _make_reranker(top_n=top_n)
    return HybridSearcher(retriever, reranker, client=client, collection=collection)


def build_dense_searcher(
    collection: str,
    *,
    embed_model,
    client=None,
    qdrant_url=None,
    reranker=None,
    dense_top_k: int = 20,
) -> HybridSearcher:
    """Searcher over a DENSE-ONLY collection (e.g. a NIM-embedded collection).

    No sparse leg, no fusion — the C (NVIDIA-native) counterpart to
    ``build_hybrid_searcher``. ``embed_model`` is a LlamaIndex embedding used for the
    query vector (e.g. ``NVIDIAEmbedding`` for the e5 NIM). ``reranker`` is an optional
    node-postprocessor (e.g. ``make_nim_reranker()``); when None, retrieval is
    dense-only with no rerank. ``client``/``qdrant_url`` mirror the hybrid builder.
    """
    if client is None:
        client = _qdrant_client(qdrant_url)
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection,
        dense_vector_name=DENSE_VECTOR_NAME,
    )
    index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)
    retriever = index.as_retriever(similarity_top_k=dense_top_k)
    return HybridSearcher(retriever, reranker, client=client, collection=collection)
