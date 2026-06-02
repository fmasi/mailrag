import unittest
from unittest import mock


class TestMainDemo(unittest.TestCase):
    def test_run_demo_builds_then_queries(self):
        import main as m
        fake_emails = [object()]
        with mock.patch.object(m, "_load_demo_emails", return_value=fake_emails), \
             mock.patch.object(m, "generate_thread_summaries", return_value={}), \
             mock.patch.object(m, "build_contextual_index") as build, \
             mock.patch.object(m, "build_hybrid_searcher") as mk_searcher, \
             mock.patch.object(m, "_answer", return_value="ANSWER"), \
             mock.patch.object(m, "_make_embedder", return_value=mock.Mock()), \
             mock.patch.object(m, "_require_qdrant", return_value=None), \
             mock.patch.object(m, "_init_settings", return_value=None):
            searcher = mk_searcher.return_value
            searcher.search_threads.return_value = []
            m.run_demo(num_samples=3, queries=["q1"])
        build.assert_called_once()
        searcher.search_threads.assert_called_with("q1")
