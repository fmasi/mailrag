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


from unittest.mock import MagicMock
from src.query.summary_rerank import SummaryAwareReranker


class TestSummaryAwareReranker(unittest.TestCase):
    def test_scores_on_summary_plus_body_and_reorders(self):
        mock_model = MagicMock()
        mock_model.compute_score.return_value = [0.1, 0.9]
        rr = SummaryAwareReranker(top_n=2, _reranker=mock_model)
        n1, n2 = _Node("B1", "S1"), _Node("B2", "S2")
        out = rr.postprocess_nodes([n1, n2], query_str="q")
        mock_model.compute_score.assert_called_once_with([["q", "S1\n\nB1"], ["q", "S2\n\nB2"]])
        self.assertEqual(out, [n2, n1])  # 0.9 before 0.1
        self.assertEqual(n2.score, 0.9)

    def test_single_pair_float_score(self):
        mock_model = MagicMock()
        mock_model.compute_score.return_value = 0.7  # FlagReranker returns float for one pair
        rr = SummaryAwareReranker(top_n=5, _reranker=mock_model)
        out = rr.postprocess_nodes([_Node("B", "S")], query_str="q")
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].score, 0.7)

    def test_empty_nodes(self):
        rr = SummaryAwareReranker(top_n=5, _reranker=MagicMock())
        self.assertEqual(rr.postprocess_nodes([], query_str="q"), [])


if __name__ == "__main__":
    unittest.main()
