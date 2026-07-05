# tests/eval/test_metrics.py
import math
import unittest

from src.eval.metrics import (
    aggregate,
    dcg,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    success_at_k,
)

REL = 2  # binary-relevance threshold


class PointMetricsTest(unittest.TestCase):
    def test_dcg_uses_log2_discount(self):
        # grades [3,2,0] -> 3/1 + 2/log2(3) + 0
        self.assertAlmostEqual(dcg([3, 2, 0]), 3 + 2 / math.log2(3))

    def test_ndcg_perfect_is_one(self):
        self.assertAlmostEqual(ndcg_at_k([3, 2, 1], 3), 1.0)

    def test_ndcg_zero_when_no_relevant(self):
        self.assertEqual(ndcg_at_k([0, 0, 0], 3), 0.0)

    def test_precision_at_k_threshold(self):
        # grades [3,1,2] @k=3, rel>=2 -> 2 relevant / 3
        self.assertAlmostEqual(precision_at_k([3, 1, 2], 3, REL), 2 / 3)

    def test_recall_at_k_uses_total_relevant_in_pool(self):
        # retrieved top-2 grades [3,1]; pool has 3 relevant total -> 1/3
        self.assertAlmostEqual(recall_at_k([3, 1], k=2, rel=REL, total_relevant=3), 1 / 3)

    def test_recall_zero_total_relevant_is_zero(self):
        self.assertEqual(recall_at_k([0], k=1, rel=REL, total_relevant=0), 0.0)

    def test_mrr_first_relevant_rank(self):
        self.assertAlmostEqual(mrr([0, 1, 2, 3], REL), 1 / 3)

    def test_mrr_none_relevant(self):
        self.assertEqual(mrr([0, 1, 1], REL), 0.0)

    def test_success_at_k(self):
        self.assertEqual(success_at_k(["a", "b", "c"], "b", 2), 1)
        self.assertEqual(success_at_k(["a", "b", "c"], "z", 2), 0)
        self.assertEqual(success_at_k(["a", "b", "c"], "c", 2), 0)


class AggregateTest(unittest.TestCase):
    def test_aggregate_groups_overall_and_by_category(self):
        rows = [
            {"category": "terse", "ndcg@10": 1.0, "p@10": 1.0},
            {"category": "terse", "ndcg@10": 0.0, "p@10": 0.0},
            {"category": "content", "ndcg@10": 0.5, "p@10": 0.4},
        ]
        agg = aggregate(rows, ["ndcg@10", "p@10"])
        self.assertAlmostEqual(agg["overall"]["ndcg@10"], 0.5)
        self.assertAlmostEqual(agg["terse"]["ndcg@10"], 0.5)
        self.assertAlmostEqual(agg["content"]["p@10"], 0.4)
        self.assertEqual(agg["overall"]["n"], 3)
        self.assertEqual(agg["terse"]["n"], 2)


if __name__ == "__main__":
    unittest.main()
