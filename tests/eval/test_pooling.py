# tests/eval/test_pooling.py
import unittest

from src.eval.flatten import EmailHit
from src.eval.pooling import build_pool, graded_rankings, total_relevant


def H(mid, body=""):
    return EmailHit(message_id=mid, subject="S", body=body or mid)


class PoolingTest(unittest.TestCase):
    def test_build_pool_unions_unique_emails(self):
        arms = {"C": [H("a"), H("b")], "C'": [H("b"), H("c")]}
        pool = build_pool(arms)
        self.assertEqual(sorted(p.message_id for p in pool), ["a", "b", "c"])

    def test_build_pool_keeps_one_hit_per_message_id(self):
        arms = {"C": [H("a", "first")], "thread": [H("a", "second")]}
        pool = build_pool(arms)
        self.assertEqual(len(pool), 1)
        self.assertEqual(pool[0].body, "first")  # first arm seen wins

    def test_graded_rankings_maps_arm_order_to_grades(self):
        arms = {"C": [H("a"), H("c")], "C'": [H("b")]}
        grades = {"a": 3, "b": 1, "c": 0}
        gr = graded_rankings(arms, grades)
        self.assertEqual(gr["C"], [3, 0])
        self.assertEqual(gr["C'"], [1])

    def test_total_relevant_counts_pool_at_threshold(self):
        grades = {"a": 3, "b": 2, "c": 1, "d": 0}
        self.assertEqual(total_relevant(grades, rel=2), 2)

    # --- edge cases ---

    def test_build_pool_empty_arms(self):
        self.assertEqual(build_pool({}), [])

    def test_graded_rankings_arm_with_no_hits(self):
        gr = graded_rankings({"C": []}, {})
        self.assertEqual(gr, {"C": []})

    def test_total_relevant_empty_grades(self):
        self.assertEqual(total_relevant({}, rel=1), 0)

    def test_total_relevant_all_below_threshold(self):
        grades = {"a": 0, "b": 1}
        self.assertEqual(total_relevant(grades, rel=2), 0)


if __name__ == "__main__":
    unittest.main()
