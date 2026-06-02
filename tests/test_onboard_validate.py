import json
import os
import tempfile
import unittest
from unittest import mock

from src import onboard
from src.onboard import _coverage_at3, validate_coverage


class _Node:
    """Minimal node whose metadata _node_metadata can read."""
    def __init__(self, thread_id):
        self.metadata = {"thread_id": thread_id, "message_id": "m"}


class _Searcher:
    def __init__(self, ranking):
        self._ranking = ranking  # query -> list of thread_ids in rank order

    def search(self, query):
        return [_Node(t) for t in self._ranking[query]]


class TestCoverage(unittest.TestCase):
    def test_coverage_at3_counts_top3_distinct(self):
        s = _Searcher({
            "q1": ["t1", "tx", "ty"],         # gold t1 at distinct-rank 0 -> covered
            "q2": ["ta", "tb", "tc", "t2"],   # gold t2 at distinct-rank 3 -> not covered
        })
        queries = [{"query": "q1", "thread_id": "t1"},
                   {"query": "q2", "thread_id": "t2"}]
        cov, n = _coverage_at3(s, queries)
        self.assertEqual(n, 2)
        self.assertAlmostEqual(cov, 0.5)

    def test_validate_uses_queries_file(self):
        d = tempfile.mkdtemp()
        qp = os.path.join(d, "q.jsonl")
        with open(qp, "w") as fh:
            fh.write(json.dumps({"query": "q1", "thread_id": "t1"}) + "\n")
        s = _Searcher({"q1": ["t1"]})
        cov, n = validate_coverage("coll", searcher=s, queries_path=qp)
        self.assertEqual((cov, n), (1.0, 1))

    def test_validate_returns_none_on_failure(self):
        # No queries file and gen_queries.run raises -> graceful (None, 0)
        with mock.patch("scripts.eval.gen_queries.run", side_effect=RuntimeError("x")):
            cov, n = validate_coverage("coll", searcher=_Searcher({}))
        self.assertEqual((cov, n), (None, 0))


if __name__ == "__main__":
    unittest.main()
