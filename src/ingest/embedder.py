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

    def __init__(self, model_name: str = "BAAI/bge-m3", device: str | None = None, use_fp16: bool = True):
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


def make_embedder(kind: str = "bge-m3", **kwargs) -> Embedder:
    """Construct the configured embedder.

    ``kind="bge-m3"`` (default) is the local dense+sparse hybrid embedder. The
    NVIDIA-native dense-only embedder is registered here in a later step; until
    then any other kind raises ``ValueError``. ``kwargs`` pass through to the impl.
    """
    if kind in ("bge-m3", "bge_m3", "bgem3"):
        return BgeM3Embedder(**kwargs)
    raise ValueError(f"Unknown embedder kind: {kind!r}")
