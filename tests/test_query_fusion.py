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

    def test_buried_single_list_hit_rescued_at_high_p(self):
        # m is mediocre in BOTH lists (rank1 each); g is the TOP hit of one list only.
        # p=1 (sum): m's two terms beat g's one -> m ABOVE g.
        # p=inf (max): g's stronger single term beats m -> g ABOVE m (rescued). FLIP.
        dense = _result(["f", "m"])     # f rank0, m rank1
        sparse = _result(["g", "m"])    # g rank0, m rank1
        p1 = make_rank_fusion(p=1.0)(dense, sparse, top_k=5, k=60)
        pinf = make_rank_fusion(p=float("inf"))(dense, sparse, top_k=5, k=60)
        self.assertLess(p1.ids.index("m"), p1.ids.index("g"))      # m above g at p=1
        self.assertLess(pinf.ids.index("g"), pinf.ids.index("m"))  # g above m at p=inf

    def test_buried_hit_rank_improves_monotonically_in_p(self):
        dense = _result(["f", "m"])
        sparse = _result(["g", "m"])
        idx = []
        for p in (1.0, 2.0, 10.0, float("inf")):
            out = make_rank_fusion(p=p)(dense, sparse, top_k=5, k=60)
            idx.append(out.ids.index("g"))
        self.assertEqual(idx, sorted(idx, reverse=True))   # g's index never worsens as p grows
        self.assertLess(idx[-1], idx[0])                   # and is strictly better at p=inf than p=1

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


class TestSparseWeight(unittest.TestCase):
    def test_weight_1_matches_default_ordering(self):
        # sparse_weight=1.0 must reproduce classic RRF exactly (regression guard).
        dense = _result(["a", "b", "c"])
        sparse = _result(["b", "a", "d"])
        ref = reciprocal_rank_fusion(dense, sparse, top_k=4, k=60)
        out = make_rank_fusion(p=1.0, sparse_weight=1.0)(dense, sparse, top_k=4, k=60)
        self.assertEqual(out.ids, ref.ids)

    def test_negative_sparse_weight_rejected(self):
        with self.assertRaises(ValueError):
            make_rank_fusion(sparse_weight=-1.0)

    def test_up_weighting_sparse_rescues_a_sparse_only_hit(self):
        # 'd' is dense-rank-0 only; 'g' is sparse-rank-0 only. At weight 1 they tie
        # (dense leg seen first -> 'd' first); up-weighting sparse flips 'g' on top.
        dense = _result(["d"])
        sparse = _result(["g"])
        eq = make_rank_fusion(p=1.0, sparse_weight=1.0)(dense, sparse, top_k=2, k=60)
        self.assertEqual(eq.ids, ["d", "g"])
        up = make_rank_fusion(p=1.0, sparse_weight=2.0)(dense, sparse, top_k=2, k=60)
        self.assertEqual(up.ids[0], "g")


if __name__ == "__main__":
    unittest.main()
