"""Embedders for mailrag: the ``Embedder`` protocol + the bge-m3 default impl.

bge-m3 (via FlagEmbedding) emits dense (1024-d) + learned sparse from one forward
pass, which is what drives hybrid retrieval. Runs on Apple-Silicon MPS by default
(fastest measured path on this machine; see project notes). The bge-m3 impl is an
integration component (requires FlagEmbedding + torch), exercised by the build
smoke-test, not the unit suite — so its identity/shape live in class-level metadata
that callers (collection sizing, hybrid-vs-dense selection) can read without loading
the model.
"""

from typing import Dict, List, Protocol, Tuple, runtime_checkable

import numpy as np


@runtime_checkable
class Embedder(Protocol):
    """Structural contract every mailrag embedder satisfies.

    Attributes:
        name: model identifier, recorded for collection provenance.
        dim: dense vector dimensionality (the Qdrant collection's vector size).
        produces_sparse: whether ``encode`` also returns learned sparse weights.
            True => hybrid-capable (dense + sparse); False => dense-only (e.g. a
            remote OpenAI-compatible / NIM embedder, whose endpoint cannot carry
            sparse), in which case the sparse list is empty.
    """

    name: str
    dim: int
    produces_sparse: bool

    def encode(
        self, texts: List[str], batch_size: int = 32, max_length: int = 512
    ) -> Tuple[np.ndarray, List[Dict[str, float]]]:
        """Return ``(dense_vecs [N, dim], sparse_weights [N dicts])``."""
        ...


class BgeM3Embedder:
    # Class-level metadata: fixed for bge-m3 and readable without loading the model.
    name: str = "BAAI/bge-m3"
    dim: int = 1024
    produces_sparse: bool = True

    def __init__(
        self, model_name: str = "BAAI/bge-m3", device: str | None = None, use_fp16: bool = True
    ):
        from FlagEmbedding import BGEM3FlagModel

        from src.ingest.device import pick_device

        if device is None:
            device = pick_device()

        self.name = model_name
        self.model = BGEM3FlagModel(model_name, use_fp16=use_fp16, devices=device)

    def encode(
        self, texts: List[str], batch_size: int = 32, max_length: int = 512
    ) -> Tuple[np.ndarray, List[Dict[str, float]]]:
        """Return (dense_vecs [N,1024], lexical_weights [N dicts])."""
        out = self.model.encode(
            texts,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
            batch_size=batch_size,
            max_length=max_length,
        )
        return out["dense_vecs"], out["lexical_weights"]


class NimEmbedder:
    """Dense-only embedder backed by an NVIDIA embedding NIM via the official
    LlamaIndex ``NVIDIAEmbedding`` connector.

    ``produces_sparse=False``: an OpenAI-style ``/embeddings`` endpoint returns a
    single dense vector and cannot carry bge-m3's learned sparse weights, so a
    NIM-embedded collection is dense-only (hybrid's sparse leg is unavailable on
    this path). ``dim``/``name`` identify the model so the index path sizes the
    collection correctly.
    """

    produces_sparse: bool = False

    # Dense dims of the hosted NVIDIA embedding NIMs (verified live 2026-06-10).
    _KNOWN_DIMS = {
        "nvidia/nv-embedqa-e5-v5": 1024,
        "nvidia/llama-nemotron-embed-1b-v2": 2048,
        "nvidia/nv-embed-v1": 4096,
    }

    def __init__(
        self,
        model: str = "nvidia/nv-embedqa-e5-v5",
        api_key: str | None = None,
        dim: int | None = None,
        truncate: str = "NONE",
        **kwargs,
    ):
        import os

        from llama_index.embeddings.nvidia import NVIDIAEmbedding

        self.name = model
        self.dim = dim if dim is not None else self._KNOWN_DIMS.get(model)
        if self.dim is None:
            raise ValueError(f"Unknown dim for embedder {model!r}; pass dim=... explicitly.")
        key = api_key or os.environ.get("NVIDIA_API_KEY")
        self._embed = NVIDIAEmbedding(model=model, api_key=key, **kwargs)
        # truncate="NONE" => the NIM errors on over-length input rather than
        # silently dropping the chunk tail (we must never lose content invisibly).
        try:
            self._embed.truncate = truncate
        except Exception:  # pragma: no cover - connector without the attribute
            pass

    def encode(
        self, texts: List[str], batch_size: int = 32, max_length: int = 512
    ) -> Tuple[np.ndarray, List[Dict[str, float]]]:
        """Return ``(dense_vecs [N, dim], [{} ...])`` — dense-only, empty sparse.

        Batching is handled by the connector (``embed_batch_size``); the
        ``batch_size``/``max_length`` args exist for ``Embedder`` compatibility.
        """
        vecs = self._embed.get_text_embedding_batch(list(texts))
        dense = np.asarray(vecs, dtype="float32")
        sparse: List[Dict[str, float]] = [{} for _ in texts]
        return dense, sparse


def make_embedder(kind: str = "bge-m3", **kwargs) -> Embedder:
    """Construct the configured embedder.

    - ``"bge-m3"`` (default): local dense + learned-sparse hybrid embedder.
    - ``"nvidia-e5"`` / ``"nvidia-nemotron"``: dense-only NVIDIA embedding NIMs.

    ``kwargs`` pass through to the impl (e.g. ``api_key``, ``dim``).
    """
    if kind in ("bge-m3", "bge_m3", "bgem3"):
        return BgeM3Embedder(**kwargs)
    if kind in ("nvidia-e5", "nv-embedqa-e5-v5", "e5"):
        return NimEmbedder(model="nvidia/nv-embedqa-e5-v5", **kwargs)
    if kind in ("nvidia-nemotron", "llama-nemotron-embed-1b-v2"):
        return NimEmbedder(model="nvidia/llama-nemotron-embed-1b-v2", **kwargs)
    raise ValueError(f"Unknown embedder kind: {kind!r}")
