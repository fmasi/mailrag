import unittest

from src.eval.decisions import decide

# Minimal rankings: 1 terse query where thread-aware surfaces the relevant sibling,
# 1 content query where C-family beats C'.
RANKINGS = [
    {
        "query": "terse-q",
        "category": "terse",
        "answer_message_id": "sib",
        "arms": {
            "C": ["terse"],
            "C+rerank": ["terse"],
            "C+rerank+thread": ["sib", "terse"],
            "Cprime": ["sib"],
            "Cprime+rerank": ["sib"],
        },
    },
    {
        "query": "content-q",
        "category": "content",
        "answer_message_id": "doc",
        "arms": {
            "C": ["doc"],
            "C+rerank": ["doc"],
            "C+rerank+thread": ["doc"],
            "Cprime": ["off"],
            "Cprime+rerank": ["off"],
        },
    },
]
GRADES = {
    "terse-q": {"terse": 0, "sib": 3},
    "content-q": {"doc": 3, "off": 0},
}


class DecideTest(unittest.TestCase):
    def test_retire_cprime_true_when_thread_aware_dominates(self):
        d = decide(RANKINGS, GRADES)
        self.assertTrue(d["retire_cprime"])

    def test_returns_both_decision_keys(self):
        d = decide(RANKINGS, GRADES)
        self.assertIn("retire_cprime", d)
        self.assertIn("auto_bound", d)

    def test_auto_bound_requires_bounding_input(self):
        d = decide(RANKINGS, GRADES, pct_over_8k=12.0)
        self.assertTrue(d["auto_bound"])
        d2 = decide(RANKINGS, GRADES, pct_over_8k=3.0)
        self.assertFalse(d2["auto_bound"])


if __name__ == "__main__":
    unittest.main()
