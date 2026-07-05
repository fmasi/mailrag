"""Unit tests for the Embedder protocol + factory (src.ingest.embedder).

The protocol is the seam that the NVIDIA-native (dense-only) embedder will plug
into later. bge-m3 metadata is class-level so it can be asserted WITHOUT loading
the heavy FlagEmbedding model (the embedder itself stays an integration component).
"""

import unittest
from unittest.mock import patch

from src.ingest.embedder import BgeM3Embedder, Embedder, make_embedder


class TestBgeM3Metadata(unittest.TestCase):
    def test_dim_is_1024(self):
        self.assertEqual(BgeM3Embedder.dim, 1024)

    def test_name_defaults_to_bge_m3(self):
        self.assertEqual(BgeM3Embedder.name, "BAAI/bge-m3")

    def test_produces_sparse_is_true(self):
        self.assertTrue(BgeM3Embedder.produces_sparse)


class TestEmbedderProtocol(unittest.TestCase):
    def test_compliant_object_is_instance(self):
        class Fake:
            name = "fake"
            dim = 8
            produces_sparse = False

            def encode(self, texts, batch_size=32, max_length=512):
                return ([], [])

        self.assertIsInstance(Fake(), Embedder)

    def test_object_missing_members_is_not_instance(self):
        class NotEmbedder:
            pass

        self.assertNotIsInstance(NotEmbedder(), Embedder)


class TestMakeEmbedder(unittest.TestCase):
    def test_returns_bge_m3_by_default(self):
        with patch("src.ingest.embedder.BgeM3Embedder") as B:
            out = make_embedder()
            B.assert_called_once()
            self.assertIs(out, B.return_value)

    def test_bge_m3_kind_accepted(self):
        with patch("src.ingest.embedder.BgeM3Embedder") as B:
            make_embedder("bge-m3", use_fp16=False)
            self.assertEqual(B.call_args.kwargs, {"use_fp16": False})

    def test_unknown_kind_raises(self):
        with self.assertRaisesRegex(ValueError, "Unknown embedder"):
            make_embedder("does-not-exist")


if __name__ == "__main__":
    unittest.main()
