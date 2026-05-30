"""Tests for the coverage-miss diagnostic logic (issue #12)."""
import unittest

from src.eval.coverage_diag import (
    best_gold_rank, distinct_thread_rank, is_terse, lexical_overlap)


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


if __name__ == "__main__":
    unittest.main()
