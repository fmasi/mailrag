"""Tests for HyDE query-side helpers (issue #16)."""

import unittest

from src.query.hyde import build_hyde_prompt, build_hyde_prompt_anchored, combine_query


class TestBuildHydePrompt(unittest.TestCase):
    def test_embeds_query(self):
        self.assertIn("the QUERY", build_hyde_prompt("the QUERY"))

    def test_asks_for_short_answer(self):
        p = build_hyde_prompt("q").lower()
        self.assertIn("answer", p)
        self.assertTrue("sentence" in p or "short" in p)


class TestBuildHydePromptAnchored(unittest.TestCase):
    def test_embeds_query(self):
        self.assertIn("the QUERY", build_hyde_prompt_anchored("the QUERY"))

    def test_answer_shaped(self):
        p = build_hyde_prompt_anchored("q").lower()
        self.assertIn("answer", p)

    def test_forbids_inventing_specifics(self):
        # The whole point: keep the query's real anchors, do NOT fabricate new ones.
        p = build_hyde_prompt_anchored("q").lower()
        self.assertTrue("do not invent" in p or "do not add" in p or "no new" in p)
        # must mention the categories of specifics it must not fabricate
        self.assertTrue("name" in p and ("date" in p or "number" in p))

    def test_preserves_query_terms(self):
        p = build_hyde_prompt_anchored("q").lower()
        self.assertTrue("keep" in p or "preserve" in p or "exactly" in p or "verbatim" in p)


class TestCombineQuery(unittest.TestCase):
    def test_pure_returns_hypothetical(self):
        self.assertEqual(combine_query("q", "hypo answer", "pure"), "hypo answer")

    def test_augment_contains_both(self):
        out = combine_query("the query", "the hypo", "augment")
        self.assertIn("the query", out)
        self.assertIn("the hypo", out)

    def test_empty_hypothetical_falls_back_to_query_pure(self):
        self.assertEqual(combine_query("q", "", "pure"), "q")

    def test_empty_hypothetical_falls_back_to_query_augment(self):
        self.assertEqual(combine_query("q", "", "augment"), "q")

    def test_whitespace_hypothetical_falls_back(self):
        self.assertEqual(combine_query("q", "   \n ", "pure"), "q")

    def test_unknown_mode_returns_query(self):
        self.assertEqual(combine_query("q", "hypo", "weird"), "q")


if __name__ == "__main__":
    unittest.main()
