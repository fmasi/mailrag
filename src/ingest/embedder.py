"""bge-m3 embedder via FlagEmbedding — dense (1024-d) + learned sparse.

Runs on Apple-Silicon MPS by default (fastest measured path on this machine;
see project notes). Integration component: requires FlagEmbedding + torch, so it
is exercised by the build smoke-test, not the unit suite.
"""
from typing import Dict, List, Tuple

import numpy as np


class BgeM3Embedder:
    def __init__(self, model_name: str = "BAAI/bge-m3", device: str | None = None, use_fp16: bool = True):
        from FlagEmbedding import BGEM3FlagModel
        from src.ingest.device import pick_device
        if device is None:
            device = pick_device()

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
