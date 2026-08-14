"""`make demo`'s scoring logic and fixtures (#125).

The demo prints numbers a stranger will quote, so its arithmetic gets the same
treatment as the benchmark's: a demo that miscounts is worse than no demo.
Index-building and search need Qdrant and bge-m3 and are not unit-tested here.
"""

from __future__ import annotations

import unittest

from scripts.demo import FIXTURES, _hit, gold_rank, load_fixtures, mcnemar, recall


class _Node:
    def __init__(self, message_id):
        self.metadata = {"message_id": message_id}


class TestGoldRank(unittest.TestCase):
    def test_finds_the_gold_position(self):
        self.assertEqual(gold_rank([_Node("a"), _Node("gold")], "gold"), 1)

    def test_missing_gold_is_none(self):
        self.assertIsNone(gold_rank([_Node("a")], "gold"))

    def test_chunks_of_one_message_count_once(self):
        """Several chunks of the same email are one retrieved document; counting
        them separately would inflate both arms and flatter the comparison."""
        hits = [_Node("a"), _Node("a"), _Node("a"), _Node("gold")]
        self.assertEqual(gold_rank(hits, "gold"), 1)


class TestRecall(unittest.TestCase):
    def test_counts_ranks_inside_k(self):
        self.assertEqual(recall([0, 1, None], 5), 200 / 3)

    def test_a_rank_equal_to_k_is_outside(self):
        self.assertEqual(recall([5], 5), 0.0)

    def test_misses_stay_in_the_denominator(self):
        self.assertEqual(recall([0, None], 5), 50.0)


class TestMcNemar(unittest.TestCase):
    def test_counts_only_disagreements(self):
        a = [0, 0, None, None]  # hit, hit, miss, miss
        b = [0, None, 0, None]  # hit, miss, hit, miss
        fx, bk, _ = mcnemar(a, b, 5)
        self.assertEqual((fx, bk), (1, 1))

    def test_no_disagreement_is_p_one(self):
        self.assertEqual(mcnemar([0, 1], [0, 1], 5), (0, 0, 1.0))

    def test_a_one_sided_sweep_is_significant(self):
        fx, bk, p = mcnemar([None] * 8, [0] * 8, 5)
        self.assertEqual((fx, bk), (8, 0))
        self.assertLess(p, 0.05)

    def test_an_even_split_is_not_significant(self):
        _, _, p = mcnemar([0, 0, None, None], [None, None, 0, 0], 5)
        self.assertEqual(p, 1.0)

    def test_p_is_bounded(self):
        for a, b in (([0], [0]), ([None], [0]), ([0, None], [None, 0])):
            self.assertLessEqual(mcnemar(a, b, 5)[2], 1.0)


class TestHit(unittest.TestCase):
    def test_rank_zero_is_a_hit_at_one(self):
        """Guards the off-by-one between 0-indexed ranks and 1-indexed @k."""
        self.assertTrue(_hit(0, 1))
        self.assertFalse(_hit(1, 1))


class TestFixtures(unittest.TestCase):
    """The fixtures are committed, so drift between them is possible and would
    silently change what the demo prints."""

    def test_all_three_fixtures_exist(self):
        for name in ("corpus.jsonl", "summaries.jsonl", "questions.jsonl"):
            with self.subTest(name=name):
                self.assertTrue((FIXTURES / name).exists(), f"missing {name}")

    def test_every_question_has_its_answer_in_the_corpus(self):
        """A question whose gold message is absent is unanswerable and drags both
        arms down equally — silently making the demo look worse than it is."""
        corpus, _, questions = load_fixtures()
        ids = {r["message_id"] for r in corpus}
        missing = [q["gold"] for q in questions if q["gold"] not in ids]
        self.assertEqual(missing, [], f"{len(missing)} gold messages absent from the corpus")

    def test_summaries_cover_most_of_the_corpus(self):
        corpus, summaries, _ = load_fixtures()
        covered = sum(1 for r in corpus if r["message_id"] in summaries)
        self.assertGreater(covered / len(corpus), 0.9)

    def test_corpus_carries_derived_thread_roots(self):
        """Without a thread root the context arm has no preceding messages to
        summarise, and the demo silently degenerates into the plain arm."""
        corpus, _, _ = load_fixtures()
        self.assertTrue(all(r.get("thread") for r in corpus))
        self.assertGreater(len({r["thread"] for r in corpus}), 1)

    def test_questions_do_not_quote_their_answer_verbatim(self):
        """Leakage guard: a question containing its answer's opening words tests
        string matching, not retrieval."""
        corpus, _, questions = load_fixtures()
        by_id = {r["message_id"]: r["body"] for r in corpus}
        leaked = [
            q["query"]
            for q in questions
            if (body := (by_id.get(q["gold"]) or "").strip().lower()[:40])
            and body in q["query"].lower()
        ]
        self.assertEqual(leaked, [], f"{len(leaked)} question(s) quote their answer")


if __name__ == "__main__":
    unittest.main()
