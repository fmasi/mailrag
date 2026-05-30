"""Tests for the coverage-miss diagnostic logic (issue #12)."""
import unittest

from src.eval.coverage_diag import (
    best_gold_rank, classify_miss, distinct_thread_rank, is_terse, lexical_overlap)


def _h(tid, mid):
    return {"thread_id": tid, "message_id": mid}


class TestBestGoldRank(unittest.TestCase):
    def test_gold_email_at_rank_zero(self):
        hits = [_h("T1", "m1"), _h("T2", "m2")]
        self.assertEqual(
            best_gold_rank(hits, "T1", "m1"), {"thread_rank": 0, "email_rank": 0})

    def test_gold_thread_sibling_before_gold_email(self):
        # a different email of the gold thread ranks above the gold email itself
        hits = [_h("T2", "m2"), _h("T1", "sibling"), _h("T1", "m1")]
        self.assertEqual(
            best_gold_rank(hits, "T1", "m1"), {"thread_rank": 1, "email_rank": 2})

    def test_gold_absent(self):
        hits = [_h("T2", "m2"), _h("T3", "m3")]
        self.assertEqual(
            best_gold_rank(hits, "T1", "m1"), {"thread_rank": None, "email_rank": None})

    def test_empty_hits(self):
        self.assertEqual(
            best_gold_rank([], "T1", "m1"), {"thread_rank": None, "email_rank": None})


class TestDistinctThreadRank(unittest.TestCase):
    def test_first_distinct_thread(self):
        hits = [_h("T1", "m1"), _h("T1", "m1b"), _h("T2", "m2")]
        self.assertEqual(distinct_thread_rank(hits, "T1"), 0)

    def test_third_distinct_thread(self):
        # gold thread is the 3rd DISTINCT thread despite repeats above it
        hits = [_h("T1", "a"), _h("T1", "b"), _h("T2", "c"), _h("T3", "d")]
        self.assertEqual(distinct_thread_rank(hits, "T3"), 2)

    def test_absent(self):
        hits = [_h("T1", "a"), _h("T2", "b")]
        self.assertIsNone(distinct_thread_rank(hits, "T9"))


class TestIsTerse(unittest.TestCase):
    def test_empty(self):
        self.assertTrue(is_terse(""))

    def test_whitespace_only(self):
        self.assertTrue(is_terse("   \n  "))

    def test_short_body(self):
        self.assertTrue(is_terse("ok thanks"))

    def test_long_body_not_terse(self):
        self.assertFalse(is_terse("x" * 200))


class TestLexicalOverlap(unittest.TestCase):
    def test_full_overlap(self):
        self.assertEqual(lexical_overlap("release timeline", "the release timeline is set"), 1.0)

    def test_zero_overlap(self):
        self.assertEqual(lexical_overlap("alpha beta", "gamma delta"), 0.0)

    def test_case_and_punctuation_normalized(self):
        self.assertEqual(lexical_overlap("Release, Timeline!", "release timeline"), 1.0)

    def test_partial_overlap(self):
        # 1 of 2 distinct query tokens present
        self.assertEqual(lexical_overlap("alpha beta", "alpha gamma"), 0.5)

    def test_empty_query_is_zero(self):
        self.assertEqual(lexical_overlap("", "anything"), 0.0)


class TestClassifyMiss(unittest.TestCase):
    # defaults used in the live run
    TOP_HITS, N, K = 10, 3, 20

    def _c(self, **ranks):
        base = {"hyb_thread_rank": None, "hyb_distinct_rank": None,
                "dense_thread_rank": None, "sparse_thread_rank": None}
        base.update(ranks)
        return classify_miss(base, self.TOP_HITS, self.N, self.K)

    def test_covered_when_distinct_rank_below_n(self):
        self.assertEqual(self._c(hyb_distinct_rank=2, hyb_thread_rank=5), "covered")

    def test_budget_when_in_pool_but_past_n_threads(self):
        # 4th distinct thread (rank 3 >= N=3) but its hit is within top_hits nodes
        self.assertEqual(self._c(hyb_distinct_rank=3, hyb_thread_rank=7), "budget")

    def test_fusion_when_single_mode_finds_it(self):
        # not in hybrid pool, but dense ranks it high
        self.assertEqual(self._c(dense_thread_rank=4), "fusion")

    def test_fusion_via_sparse(self):
        self.assertEqual(self._c(sparse_thread_rank=10), "fusion")

    def test_hard_when_deep_in_both(self):
        self.assertEqual(
            self._c(dense_thread_rank=150, sparse_thread_rank=180), "hard")

    def test_hard_when_absent_everywhere(self):
        self.assertEqual(self._c(), "hard")

    def test_budget_boundary_exactly_at_top_hits_is_not_budget(self):
        # hyb_thread_rank == top_hits (10) is OUTSIDE the pool (0-based) -> not budget
        self.assertEqual(self._c(hyb_thread_rank=10, dense_thread_rank=4), "fusion")

    def test_fusion_boundary_exactly_at_k_is_not_fusion(self):
        # dense rank == K (20) is outside "within first K" (0-based) -> hard
        self.assertEqual(self._c(dense_thread_rank=20), "hard")


if __name__ == "__main__":
    unittest.main()
