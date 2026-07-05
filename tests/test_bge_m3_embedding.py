"""Tests for the bge-m3 LlamaIndex dense adapter + sparse query fn."""

import unittest
from unittest.mock import MagicMock

import numpy as np

from src.query.bge_m3_embedding import (
    BgeM3LlamaIndexEmbedding,
    make_bge_m3_sparse_query_fn,
)


class TestBgeM3DenseEmbedding(unittest.TestCase):
    def _embedder(self):
        emb = MagicMock()
        emb.encode.return_value = (np.array([[0.1, 0.2, 0.3]]), [{"5": 0.9}])
        return emb

    def test_query_embedding_returns_dense_floats(self):
        m = BgeM3LlamaIndexEmbedding(embedder=self._embedder())
        out = m._get_query_embedding("hello")
        self.assertEqual(len(out), 3)
        for got, exp in zip(out, [0.1, 0.2, 0.3]):
            self.assertAlmostEqual(got, exp, places=5)
        self.assertTrue(all(isinstance(x, float) for x in out))

    def test_text_embedding_matches_query_path(self):
        m = BgeM3LlamaIndexEmbedding(embedder=self._embedder())
        self.assertEqual(m._get_text_embedding("hi"), m._get_query_embedding("hi"))


class TestBgeM3SparseQueryFn(unittest.TestCase):
    def test_batched_indices_values_from_lexical_weights(self):
        emb = MagicMock()
        emb.encode.return_value = (np.zeros((1, 3)), [{"4764": 0.12, "53": 0.09}])
        fn = make_bge_m3_sparse_query_fn(emb)
        indices, values = fn(["query"])
        self.assertEqual(indices, [[4764, 53]])
        self.assertEqual([round(v, 2) for v in values[0]], [0.12, 0.09])
        self.assertTrue(all(isinstance(i, int) for i in indices[0]))

    def test_drops_non_positive_and_handles_empty(self):
        emb = MagicMock()
        emb.encode.return_value = (np.zeros((2, 3)), [{"1": 0.5, "2": 0.0}, {}])
        fn = make_bge_m3_sparse_query_fn(emb)
        indices, values = fn(["a", "b"])
        self.assertEqual(indices, [[1], []])
        self.assertEqual(values, [[0.5], []])


if __name__ == "__main__":
    unittest.main()
