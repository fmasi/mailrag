# tests/eval/test_agreement.py
import unittest

from src.eval.agreement import cohen_kappa, decision_flips, spearman


class KappaTest(unittest.TestCase):
    def test_perfect_agreement_is_one(self):
        self.assertAlmostEqual(cohen_kappa([0, 1, 2, 3], [0, 1, 2, 3]), 1.0)

    def test_chance_agreement_near_zero(self):
        a = [0, 0, 1, 1]
        b = [0, 1, 0, 1]
        self.assertLess(cohen_kappa(a, b), 0.5)


class SpearmanTest(unittest.TestCase):
    def test_monotonic_is_one(self):
        self.assertAlmostEqual(spearman([1, 2, 3, 4], [10, 20, 30, 40]), 1.0)

    def test_reversed_is_minus_one(self):
        self.assertAlmostEqual(spearman([1, 2, 3, 4], [40, 30, 20, 10]), -1.0)


class DecisionFlipTest(unittest.TestCase):
    def test_no_flip_when_same_ranking_and_decisions(self):
        local = {"retire_cprime": True, "auto_bound": False}
        ref = {"retire_cprime": True, "auto_bound": False}
        self.assertEqual(decision_flips(local, ref), [])

    def test_flags_each_flipped_decision(self):
        local = {"retire_cprime": True, "auto_bound": False}
        ref = {"retire_cprime": False, "auto_bound": True}
        self.assertEqual(sorted(decision_flips(local, ref)), ["auto_bound", "retire_cprime"])


if __name__ == "__main__":
    unittest.main()
