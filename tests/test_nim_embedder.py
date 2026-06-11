"""Unit tests for NimEmbedder — the dense-only NVIDIA-NIM embedder (hermetic).

The official LlamaIndex NVIDIAEmbedding connector is mocked, so these tests make
no network calls and spend no credits. NimEmbedder is dense-only
(produces_sparse=False): the OpenAI-style /embeddings endpoint returns a single
dense vector and cannot carry learned sparse weights.
"""
import unittest
from unittest import mock


def _patched_connector(batch_return):
    """Patch the lazily-imported NVIDIAEmbedding; return (patch_ctx, instance_mock)."""
    inst = mock.MagicMock()
    inst.get_text_embedding_batch.return_value = batch_return
    ctx = mock.patch("llama_index.embeddings.nvidia.NVIDIAEmbedding", return_value=inst)
    return ctx, inst


class TestNimEmbedder(unittest.TestCase):
    def test_produces_sparse_is_false(self):
        from src.ingest.embedder import NimEmbedder
        self.assertFalse(NimEmbedder.produces_sparse)

    def test_known_model_sets_dim_and_name(self):
        ctx, _ = _patched_connector([[0.0] * 1024])
        with ctx:
            from src.ingest.embedder import NimEmbedder
            e = NimEmbedder(model="nvidia/nv-embedqa-e5-v5", api_key="k")
            self.assertEqual(e.dim, 1024)
            self.assertEqual(e.name, "nvidia/nv-embedqa-e5-v5")

    def test_unknown_model_requires_explicit_dim(self):
        ctx, _ = _patched_connector([[0.0] * 8])
        with ctx:
            from src.ingest.embedder import NimEmbedder
            with self.assertRaisesRegex(ValueError, "dim"):
                NimEmbedder(model="nvidia/some-future-embedder", api_key="k")

    def test_encode_returns_dense_array_and_empty_sparse(self):
        ctx, inst = _patched_connector([[0.1, 0.2], [0.3, 0.4]])
        with ctx:
            from src.ingest.embedder import NimEmbedder
            e = NimEmbedder(model="nvidia/nv-embedqa-e5-v5", api_key="k", dim=2)
            dense, sparse = e.encode(["a", "b"])
            self.assertEqual(dense.shape, (2, 2))
            self.assertEqual(sparse, [{}, {}])  # dense-only -> no learned sparse
            inst.get_text_embedding_batch.assert_called_once()

    def test_truncate_none_set_on_connector(self):
        ctx, inst = _patched_connector([[0.0] * 1024])
        with ctx:
            from src.ingest.embedder import NimEmbedder
            NimEmbedder(model="nvidia/nv-embedqa-e5-v5", api_key="k")
            # fail-loud: never silently truncate over-length chunks
            self.assertEqual(inst.truncate, "NONE")

    def test_satisfies_embedder_protocol(self):
        ctx, _ = _patched_connector([[0.0] * 1024])
        with ctx:
            from src.ingest.embedder import NimEmbedder, Embedder
            self.assertIsInstance(NimEmbedder(api_key="k"), Embedder)

    def test_make_embedder_nvidia_e5(self):
        ctx, _ = _patched_connector([[0.0] * 1024])
        with ctx:
            from src.ingest.embedder import make_embedder, NimEmbedder
            e = make_embedder("nvidia-e5", api_key="k")
            self.assertIsInstance(e, NimEmbedder)
            self.assertEqual(e.dim, 1024)


if __name__ == "__main__":
    unittest.main()
