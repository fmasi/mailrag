"""Tests for summary-aware rerank helpers."""
import unittest

from src.query.summary_rerank import build_rerank_text, rerank_by_scores


class _Node:
    def __init__(self, body, summary=None, score=0.0):
        self.metadata = {} if summary is None else {"summary": summary}
        self._body = body
        self.score = score

    def get_content(self):
        return self._body


class TestBuildRerankText(unittest.TestCase):
    def test_with_summary_prepends(self):
        self.assertEqual(build_rerank_text(_Node("BODY", "SUMM")), "SUMM\n\nBODY")

    def test_without_summary_is_body(self):
        self.assertEqual(build_rerank_text(_Node("BODY")), "BODY")

    def test_blank_summary_is_body(self):
        self.assertEqual(build_rerank_text(_Node("BODY", "   ")), "BODY")


class TestRerankByScores(unittest.TestCase):
    def test_sorts_desc_sets_scores_truncates(self):
        a, b, c = _Node("a"), _Node("b"), _Node("c")
        out = rerank_by_scores([a, b, c], [0.2, 0.9, 0.5], top_n=2)
        self.assertEqual(out, [b, c])
        self.assertEqual(b.score, 0.9)
        self.assertEqual(c.score, 0.5)
        self.assertEqual(len(out), 2)

    def test_empty(self):
        self.assertEqual(rerank_by_scores([], [], top_n=5), [])


if __name__ == "__main__":
    unittest.main()
