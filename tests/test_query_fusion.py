"""Tests for the RRF hybrid_fusion_fn callback."""
import unittest

from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import VectorStoreQueryResult

from src.query.fusion import reciprocal_rank_fusion


def _result(ids):
    nodes = [TextNode(text=f"n{i}", id_=i) for i in ids]
    return VectorStoreQueryResult(nodes=nodes, similarities=[1.0] * len(ids), ids=list(ids))


class TestReciprocalRankFusion(unittest.TestCase):
    def test_item_ranked_high_by_both_wins(self):
        dense = _result(["a", "b", "c"])
        sparse = _result(["b", "a", "d"])
        out = reciprocal_rank_fusion(dense, sparse, alpha=0.5, top_k=4, k=60)
        # 'a' (ranks 0 and 1) and 'b' (ranks 1 and 0) lead; both beat c/d (one list only)
        self.assertEqual(set(out.ids[:2]), {"a", "b"})
        self.assertEqual(len(out.ids), 4)

    def test_top_k_truncates(self):
        dense = _result(["a", "b", "c"])
        sparse = _result(["d", "e", "f"])
        out = reciprocal_rank_fusion(dense, sparse, top_k=2)
        self.assertEqual(len(out.ids), 2)
        self.assertEqual(len(out.nodes), 2)

    def test_nodes_follow_ids(self):
        dense = _result(["a", "b"])
        sparse = _result(["a", "b"])
        out = reciprocal_rank_fusion(dense, sparse, top_k=2)
        self.assertEqual([n.node_id for n in out.nodes], out.ids)

    def test_empty_legs(self):
        empty = VectorStoreQueryResult(nodes=[], similarities=[], ids=[])
        out = reciprocal_rank_fusion(empty, empty, top_k=5)
        self.assertEqual(out.ids, [])
        self.assertEqual(out.nodes, [])

    def test_ids_without_nodes_stay_parallel(self):
        # a sparse-only style leg: ids present but no nodes; the fused output
        # must keep ids/similarities/nodes strictly parallel.
        dense = VectorStoreQueryResult(nodes=[], similarities=[], ids=["a", "b"])
        sparse = _result(["b", "c"])
        out = reciprocal_rank_fusion(dense, sparse, top_k=5)
        self.assertEqual(len(out.ids), len(out.nodes))
        self.assertEqual([n.node_id for n in out.nodes], out.ids)
        # only ids that have a node survive (b and c, from the sparse leg)
        self.assertEqual(set(out.ids), {"b", "c"})


if __name__ == "__main__":
    unittest.main()
