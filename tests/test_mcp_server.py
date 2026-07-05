"""Tests for the MCP server (src/mcp_server) over the email-RAG query path.

The retrieval layer is fully mocked (a fake searcher), so no live Qdrant or LLM
is needed: we assert the tools call into ``search_threads`` / ``answer_from_threads``
correctly, map the results, resolve config, and register the expected tool surface.
"""
import asyncio
import unittest
from unittest import mock

from src.mcp_server import server


class _FakeEmail:
    def __init__(self, mid="m"):
        self.message_id = mid


class _FakeThread:
    """Stand-in for query.thread_expand.ThreadContext."""

    def __init__(self, thread_id, subject, text, n_emails=1):
        self.thread_id = thread_id
        self.subject = subject
        self.text = text
        self.emails = [_FakeEmail(f"{thread_id}-{i}") for i in range(n_emails)]


class _FakeSearcher:
    def __init__(self, contexts):
        self._contexts = contexts
        self.calls = []

    def search_threads(self, query):
        self.calls.append(query)
        return self._contexts


def _threads():
    return [
        _FakeThread("t1", "Invoice March", "thread one body", n_emails=2),
        _FakeThread("t2", "Invoice April", "thread two body", n_emails=1),
        _FakeThread("t3", "Invoice May", "thread three body", n_emails=3),
    ]


class TestResolveConfig(unittest.TestCase):
    def test_collection_arg_wins(self):
        self.assertEqual(server.resolve_collection("explicit"), "explicit")

    def test_collection_from_env(self):
        with mock.patch.dict("os.environ", {"MAILRAG_COLLECTION": "envcol"}, clear=False):
            self.assertEqual(server.resolve_collection(), "envcol")

    def test_collection_falls_back_to_manifest(self):
        with mock.patch.dict("os.environ", {}, clear=True), \
             mock.patch("src.mcp_server.server.latest_manifest_collection",
                        return_value="manifestcol"):
            self.assertEqual(server.resolve_collection(), "manifestcol")

    def test_collection_missing_raises_clear_error(self):
        with mock.patch.dict("os.environ", {}, clear=True), \
             mock.patch("src.mcp_server.server.latest_manifest_collection",
                        return_value=None):
            with self.assertRaises(ValueError) as ctx:
                server.resolve_collection()
        self.assertIn("MAILRAG_COLLECTION", str(ctx.exception))

    def test_qdrant_url_precedence(self):
        self.assertEqual(server.resolve_qdrant_url("http://arg:6333"), "http://arg:6333")
        with mock.patch.dict("os.environ",
                             {"MAILRAG_QDRANT_URL": "http://mr:6333",
                              "QDRANT_URL": "http://container:6333"}, clear=False):
            self.assertEqual(server.resolve_qdrant_url(), "http://mr:6333")
        with mock.patch.dict("os.environ", {"QDRANT_URL": "http://container:6333"},
                             clear=True):
            self.assertEqual(server.resolve_qdrant_url(), "http://container:6333")
        with mock.patch.dict("os.environ", {}, clear=True):
            self.assertEqual(server.resolve_qdrant_url(), server.DEFAULT_QDRANT_URL)


class TestGetSearcher(unittest.TestCase):
    def setUp(self):
        server._SEARCHER_CACHE.clear()
        self.addCleanup(server._SEARCHER_CACHE.clear)

    def test_builds_via_factory_in_hybrid_mode_and_caches(self):
        fake = _FakeSearcher([])
        factory = mock.Mock(return_value=fake)
        s1 = server.get_searcher("col", "http://q:6333", factory=factory)
        s2 = server.get_searcher("col", "http://q:6333", factory=factory)
        self.assertIs(s1, fake)
        self.assertIs(s2, fake)
        factory.assert_called_once_with("col", mode="hybrid", qdrant_url="http://q:6333")


class TestSearchEmail(unittest.TestCase):
    def test_maps_threads_and_respects_top_k(self):
        searcher = _FakeSearcher(_threads())
        rows = server.search_email("invoices?", top_k=2, searcher=searcher)
        self.assertEqual(searcher.calls, ["invoices?"])
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0],
                         {"thread_id": "t1", "subject": "Invoice March",
                          "num_emails": 2, "text": "thread one body"})
        self.assertEqual(rows[1]["thread_id"], "t2")

    def test_empty_results_return_empty_list(self):
        rows = server.search_email("nothing", searcher=_FakeSearcher([]))
        self.assertEqual(rows, [])

    def test_blank_query_rejected(self):
        with self.assertRaises(ValueError):
            server.search_email("   ", searcher=_FakeSearcher([]))

    def test_bad_top_k_rejected(self):
        with self.assertRaises(ValueError):
            server.search_email("q", top_k=0, searcher=_FakeSearcher([]))

    def test_uses_get_searcher_when_not_injected(self):
        searcher = _FakeSearcher(_threads())
        with mock.patch("src.mcp_server.server.get_searcher", return_value=searcher):
            rows = server.search_email("q", top_k=1)
        self.assertEqual(len(rows), 1)


class TestAnswerQuestion(unittest.TestCase):
    def test_calls_answer_from_threads_with_top_k(self):
        searcher = _FakeSearcher(_threads())
        with mock.patch("src.mcp_server.server.answer_from_threads",
                        return_value="GROUNDED") as answer:
            out = server.answer_question("when due?", k=2, searcher=searcher)
        self.assertEqual(searcher.calls, ["when due?"])
        # answer_from_threads receives the full context list + k (it truncates).
        args, kwargs = answer.call_args
        self.assertEqual(args[0], "when due?")
        self.assertEqual(len(args[1]), 3)
        self.assertEqual(kwargs["k"], 2)
        self.assertEqual(out["answer"], "GROUNDED")
        self.assertEqual([s["thread_id"] for s in out["sources"]], ["t1", "t2"])

    def test_no_results_still_returns_answer_shape(self):
        with mock.patch("src.mcp_server.server.answer_from_threads",
                        return_value="No relevant threads retrieved.") as answer:
            out = server.answer_question("q", searcher=_FakeSearcher([]))
        answer.assert_called_once()
        self.assertEqual(out["answer"], "No relevant threads retrieved.")
        self.assertEqual(out["sources"], [])

    def test_blank_query_rejected(self):
        with self.assertRaises(ValueError):
            server.answer_question("", searcher=_FakeSearcher([]))

    def test_bad_k_rejected(self):
        with self.assertRaises(ValueError):
            server.answer_question("q", k=0, searcher=_FakeSearcher([]))


class TestServerRegistration(unittest.TestCase):
    def test_tools_registered_with_expected_names_and_schema(self):
        srv = server.build_server()
        tools = asyncio.run(srv.list_tools())
        by_name = {t.name: t for t in tools}
        self.assertEqual(set(by_name), {"search_email", "answer_question"})
        self.assertEqual(
            set(by_name["search_email"].inputSchema["properties"]),
            {"query", "top_k"})
        self.assertEqual(
            set(by_name["answer_question"].inputSchema["properties"]),
            {"query", "k"})

    def test_call_tool_dispatches_into_search_email(self):
        srv = server.build_server()
        searcher = _FakeSearcher(_threads())
        with mock.patch("src.mcp_server.server.get_searcher", return_value=searcher):
            result = asyncio.run(srv.call_tool("search_email",
                                               {"query": "invoices", "top_k": 1}))
        # FastMCP (this SDK version) returns (content_blocks, structured_result).
        content_blocks, structured = result
        rows = structured["result"]
        self.assertEqual(rows[0]["thread_id"], "t1")
        # The text content mirrors the same payload.
        self.assertIn("t1", content_blocks[0].text)


class TestCliWiring(unittest.TestCase):
    def test_mcp_verb_invokes_serve_and_sets_env(self):
        from src import cli
        with mock.patch("src.mcp_server.server.serve") as serve, \
             mock.patch.dict("os.environ", {}, clear=True):
            rc = cli.main(["mcp", "--collection", "work-rag",
                           "--qdrant-url", "http://localhost:6333"])
            self.assertEqual(rc, 0)
            serve.assert_called_once()
            import os
            self.assertEqual(os.environ["MAILRAG_COLLECTION"], "work-rag")
            self.assertEqual(os.environ["MAILRAG_QDRANT_URL"], "http://localhost:6333")


if __name__ == "__main__":
    unittest.main()
