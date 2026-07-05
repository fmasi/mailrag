"""Unit tests for the dense-only Qdrant collection path (hermetic, mocked client).

Dense-only embedders (e.g. a NVIDIA NIM, produces_sparse=False) need a collection
with just the ``dense`` named vector — no ``sparse`` leg — and points without a
sparse vector. Mirrors the hybrid path's payload indexes.
"""

import unittest
from unittest import mock

from src.ingest import hybrid_qdrant as hq


class TestDenseCollection(unittest.TestCase):
    def test_ensure_dense_collection_has_no_sparse_config(self):
        client = mock.MagicMock()
        client.collection_exists.return_value = False
        hq.ensure_dense_collection(client, "c", dim=1024, recreate=False)
        kwargs = client.create_collection.call_args.kwargs
        self.assertIn("dense", kwargs["vectors_config"])
        self.assertNotIn("sparse_vectors_config", kwargs)

    def test_ensure_dense_collection_creates_payload_indexes(self):
        client = mock.MagicMock()
        client.collection_exists.return_value = False
        hq.ensure_dense_collection(client, "c", dim=1024)
        self.assertGreaterEqual(client.create_payload_index.call_count, 5)

    def test_ensure_dense_collection_recreate_deletes_first(self):
        client = mock.MagicMock()
        client.collection_exists.return_value = True
        hq.ensure_dense_collection(client, "c", dim=8, recreate=True)
        client.delete_collection.assert_called_once_with("c")

    def test_make_dense_point_has_only_dense_vector(self):
        p = hq.make_dense_point("id1", [0.1, 0.2], {"text": "x"})
        self.assertIn("dense", p.vector)
        self.assertNotIn("sparse", p.vector)
        self.assertEqual(p.payload, {"text": "x"})


if __name__ == "__main__":
    unittest.main()
