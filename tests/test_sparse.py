"""Tests for converting bge-m3 lexical weights to Qdrant sparse-vector format."""
import unittest

from src.ingest import sparse


class TestLexicalWeightsToSparse(unittest.TestCase):
    def test_converts_token_id_weight_dict(self):
        idx, val = sparse.lexical_weights_to_sparse({"4764": 0.12, "53": 0.09})
        self.assertEqual(idx, [4764, 53])
        self.assertEqual([round(v, 2) for v in val], [0.12, 0.09])
        self.assertTrue(all(isinstance(i, int) for i in idx))
        self.assertTrue(all(isinstance(v, float) for v in val))

    def test_empty_dict(self):
        self.assertEqual(sparse.lexical_weights_to_sparse({}), ([], []))

    def test_drops_non_positive_weights(self):
        idx, val = sparse.lexical_weights_to_sparse({"1": 0.5, "2": 0.0, "3": -0.1})
        self.assertEqual(idx, [1])
        self.assertEqual(val, [0.5])


if __name__ == "__main__":
    unittest.main()
