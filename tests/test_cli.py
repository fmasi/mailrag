import unittest
from unittest import mock

from src import cli
from src.onboard import OnboardReport


def _report():
    return OnboardReport(collection="mailrag-x", kept=5, noise_dropped=1,
                         llm_failures=0, chunks=6, chunk_size=512, coverage_at3=0.8,
                         n_queries=5, validated=True)


class TestCli(unittest.TestCase):
    def test_onboard_routes_and_parses(self):
        with mock.patch("src.onboard.run_onboard", return_value=_report()) as run:
            rc = cli.main(["onboard", "/maildir", "--collection", "c",
                           "--no-validate", "--limit", "20"])
        self.assertEqual(rc, 0)
        kw = run.call_args.kwargs
        self.assertEqual(kw["collection"], "c")
        self.assertEqual(kw["validate"], False)
        self.assertEqual(kw["limit"], 20)

    def test_onboard_value_error_exits_1(self):
        with mock.patch("src.onboard.run_onboard", side_effect=ValueError("bad")):
            self.assertEqual(cli.main(["onboard", "/x"]), 1)

    def test_query_without_collection_or_manifest_exits_2(self):
        with mock.patch("src.onboard.latest_manifest_collection", return_value=None):
            self.assertEqual(cli.main(["query", "hello?"]), 2)

    def test_query_routes(self):
        searcher = mock.Mock()
        searcher.search_threads.return_value = ["CTX"]
        with mock.patch("src.onboard.latest_manifest_collection", return_value="c"), \
             mock.patch("src.query.hybrid.build_hybrid_searcher",
                        return_value=searcher), \
             mock.patch("src.llm.answer.answer_from_threads", return_value="A") as ans:
            rc = cli.main(["query", "hello?", "--k", "2"])
        self.assertEqual(rc, 0)
        ans.assert_called_once_with("hello?", ["CTX"], k=2)

    def test_main_loads_dotenv_before_dispatch(self):
        with mock.patch("src.cli.load_dotenv") as ld, \
             mock.patch("src.onboard.run_onboard", return_value=_report()):
            rc = cli.main(["onboard", "/x"])
        self.assertEqual(rc, 0)
        ld.assert_called_once()


if __name__ == "__main__":
    unittest.main()
