import unittest
from unittest import mock

from src import cli
from src.onboard import OnboardReport


def _report():
    return OnboardReport(
        collection="mailrag-x",
        kept=5,
        noise_dropped=1,
        llm_failures=0,
        chunks=6,
        chunk_size=512,
        coverage_at3=0.8,
        n_queries=5,
        validated=True,
    )


class TestCli(unittest.TestCase):
    def test_onboard_routes_and_parses(self):
        with mock.patch("src.onboard.run_onboard", return_value=_report()) as run:
            rc = cli.main(
                ["onboard", "/maildir", "--collection", "c", "--no-validate", "--limit", "20"]
            )
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
        with (
            mock.patch("src.onboard.latest_manifest_collection", return_value="c"),
            mock.patch("src.query.hybrid.build_hybrid_searcher", return_value=searcher),
            mock.patch("src.llm.answer.answer_from_threads", return_value="A") as ans,
        ):
            rc = cli.main(["query", "hello?", "--k", "2"])
        self.assertEqual(rc, 0)
        ans.assert_called_once_with("hello?", ["CTX"], k=2)

    def test_query_defaults_qdrant_url_to_localhost(self):
        """The #29 fix: query must not inherit the container QDRANT_URL from .env."""
        searcher = mock.Mock()
        searcher.search_threads.return_value = ["CTX"]
        with (
            mock.patch("src.onboard.latest_manifest_collection", return_value="c"),
            mock.patch("src.query.hybrid.build_hybrid_searcher", return_value=searcher) as bhs,
            mock.patch("src.llm.answer.answer_from_threads", return_value="A"),
        ):
            rc = cli.main(["query", "hello?"])
        self.assertEqual(rc, 0)
        self.assertEqual(bhs.call_args.kwargs["qdrant_url"], "http://localhost:6333")

    def test_query_uses_profile_collection_and_qdrant_url(self):
        searcher = mock.Mock()
        searcher.search_threads.return_value = ["CTX"]
        prof = mock.Mock(collection="pc", qdrant_url="http://prof-host:6333")
        with (
            mock.patch("src.cli.CorpusProfile.load", return_value=prof),
            mock.patch("src.query.hybrid.build_hybrid_searcher", return_value=searcher) as bhs,
            mock.patch("src.llm.answer.answer_from_threads", return_value="A"),
        ):
            rc = cli.main(["query", "hello?", "--profile", "/p.json"])
        self.assertEqual(rc, 0)
        bhs.assert_called_once_with("pc", mode="hybrid", qdrant_url="http://prof-host:6333")

    def test_query_qdrant_url_flag_overrides_profile(self):
        searcher = mock.Mock()
        searcher.search_threads.return_value = ["CTX"]
        prof = mock.Mock(collection="pc", qdrant_url="http://prof-host:6333")
        with (
            mock.patch("src.cli.CorpusProfile.load", return_value=prof),
            mock.patch("src.query.hybrid.build_hybrid_searcher", return_value=searcher) as bhs,
            mock.patch("src.llm.answer.answer_from_threads", return_value="A"),
        ):
            rc = cli.main(
                ["query", "hello?", "--profile", "/p.json", "--qdrant-url", "http://flag-host:6333"]
            )
        self.assertEqual(rc, 0)
        self.assertEqual(bhs.call_args.kwargs["qdrant_url"], "http://flag-host:6333")

    def test_main_loads_dotenv_before_dispatch(self):
        with (
            mock.patch("src.cli.load_dotenv") as ld,
            mock.patch("src.onboard.run_onboard", return_value=_report()),
        ):
            rc = cli.main(["onboard", "/x"])
        self.assertEqual(rc, 0)
        ld.assert_called_once()

    def test_explore_routes_and_prints(self):
        from src.cluster.noise_pockets import ClusterReport

        report = ClusterReport(
            k=2,
            seed=11,
            n_threads=4,
            n_emails=6,
            corpus_baseline_tag_rate=0.5,
            clusters=[],
            vector_source="fresh",
        )
        prof = mock.Mock(collection="c", qdrant_url="http://localhost:6333")
        with (
            mock.patch("src.cli.CorpusProfile.load", return_value=prof),
            mock.patch("src.pipeline.explore.run", return_value=report) as run,
            mock.patch("src.cli.explore_stage.format_report", return_value="TABLE"),
        ):
            rc = cli.main(
                [
                    "explore",
                    "--profile",
                    "p.json",
                    "--clusters",
                    "2",
                    "--json",
                    "/tmp/e.json",
                    "--fresh",
                ]
            )
        self.assertEqual(rc, 0)
        kw = run.call_args.kwargs
        self.assertEqual(kw["clusters"], 2)
        self.assertEqual(kw["json_path"], "/tmp/e.json")
        self.assertEqual(kw["force_fresh"], True)

    def test_explore_value_error_exits_2(self):
        prof = mock.Mock(collection="c", qdrant_url="http://localhost:6333")
        with (
            mock.patch("src.cli.CorpusProfile.load", return_value=prof),
            mock.patch("src.pipeline.explore.run", side_effect=ValueError("no files selected")),
        ):
            rc = cli.main(["explore", "--profile", "p.json"])
        self.assertEqual(rc, 2)


if __name__ == "__main__":
    unittest.main()
