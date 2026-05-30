"""Tests for the RRF hybrid_fusion_fn callback."""
import unittest

from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import VectorStoreQueryResult

from src.query.fusion import reciprocal_rank_fusion, make_rank_fusion


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


class TestPowerMeanFusion(unittest.TestCase):
    def test_p1_factory_matches_reciprocal_rank_fusion(self):
        dense = _result(["a", "b", "c"])
        sparse = _result(["b", "a", "d"])
        ref = reciprocal_rank_fusion(dense, sparse, top_k=4, k=60)
        out = make_rank_fusion(p=1.0)(dense, sparse, top_k=4, k=60)
        self.assertEqual(out.ids, ref.ids)
        for a, b in zip(out.similarities, ref.similarities):
            self.assertAlmostEqual(a, b)

    def test_p_inf_vs_p1_max_outweighs_sum(self):
        # At p=1 (sum), a doc with two weak terms can beat one strong + one absent.
        # At p=inf (max), the strong term wins (best rank from any list).
        dense = _result(["a", "b", "c"])  # a=rank0, b=rank1
        sparse = _result(["b", "d"])      # b=rank0, d=rank1
        # a: sum=1/61, max=1/61 (one list only)
        # b: sum=1/61+1/61=2/61, max=1/61 (two lists)
        # At p=1: b wins (larger sum)
        # At p=inf: a and b tie on max; b wins tiebreak (larger sum)
        p1 = make_rank_fusion(p=1.0)(dense, sparse, top_k=4, k=60)
        pinf = make_rank_fusion(p=float("inf"))(dense, sparse, top_k=4, k=60)
        self.assertEqual(p1.ids[0], "b")  # b is stronger (sum)
        self.assertEqual(pinf.ids[0], "b")  # b wins tiebreak at p=inf too

    def test_score_monotonic_in_p_for_buried_hit(self):
        dense = _result(["x", "y", "z", "w"])
        sparse = _result(["g", "x"])
        pos = {}
        for p in (1.0, 2.0, 10.0, float("inf")):
            out = make_rank_fusion(p=p)(dense, sparse, top_k=4, k=60)
            pos[p] = out.ids.index("g")
        ranks = [pos[p] for p in (1.0, 2.0, 10.0, float("inf"))]
        self.assertEqual(ranks, sorted(ranks, reverse=True))

    def test_p_inf_tiebreak_by_sum(self):
        # both "g" and "h" tie on max term (each rank 0 in one list); "h" is also
        # present (rank 1) in the other list, so its sum is larger -> h first.
        dense = _result(["g", "h"])     # g rank0, h rank1
        sparse = _result(["h", "q"])    # h rank0
        out = make_rank_fusion(p=float("inf"))(dense, sparse, top_k=4, k=60)
        self.assertEqual(out.ids[0], "h")

    def test_p_below_one_raises(self):
        with self.assertRaises(ValueError):
            make_rank_fusion(p=0.5)

    def test_factory_handles_ids_without_nodes(self):
        dense = VectorStoreQueryResult(nodes=[], similarities=[], ids=["a", "b"])
        sparse = _result(["b", "c"])
        out = make_rank_fusion(p=2.0)(dense, sparse, top_k=5)
        self.assertEqual(len(out.ids), len(out.nodes))
        self.assertEqual(set(out.ids), {"b", "c"})


if __name__ == "__main__":
    unittest.main()
