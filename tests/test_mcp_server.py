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
        with (
            mock.patch.dict("os.environ", {}, clear=True),
            mock.patch(
                "src.mcp_server.server.latest_manifest_collection", return_value="manifestcol"
            ),
        ):
            self.assertEqual(server.resolve_collection(), "manifestcol")

    def test_collection_missing_raises_clear_error(self):
        with (
            mock.patch.dict("os.environ", {}, clear=True),
            mock.patch("src.mcp_server.server.latest_manifest_collection", return_value=None),
        ):
            with self.assertRaises(ValueError) as ctx:
                server.resolve_collection()
        self.assertIn("MAILRAG_COLLECTION", str(ctx.exception))

    def test_qdrant_url_precedence(self):
        self.assertEqual(server.resolve_qdrant_url("http://arg:6333"), "http://arg:6333")
        with mock.patch.dict(
            "os.environ",
            {"MAILRAG_QDRANT_URL": "http://mr:6333", "QDRANT_URL": "http://container:6333"},
            clear=False,
        ):
            self.assertEqual(server.resolve_qdrant_url(), "http://mr:6333")
        with mock.patch.dict("os.environ", {"QDRANT_URL": "http://container:6333"}, clear=True):
            self.assertEqual(server.resolve_qdrant_url(), "http://container:6333")
        with mock.patch.dict("os.environ", {}, clear=True):
            self.assertEqual(server.resolve_qdrant_url(), server.DEFAULT_QDRANT_URL)


class TestResolveAttachStore(unittest.TestCase):
    def test_arg_wins(self):
        self.assertEqual(server.resolve_attach_store("/tmp/store"), "/tmp/store")

    def test_env_override(self):
        with mock.patch.dict("os.environ", {"RAG_ATTACH_STORE": "/env/store"}, clear=False):
            self.assertEqual(server.resolve_attach_store(), "/env/store")

    def test_default(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            self.assertEqual(server.resolve_attach_store(), server.DEFAULT_ATTACH_STORE)


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

    def test_mode_is_part_of_cache_key_and_passed_to_factory(self):
        hybrid, dense = _FakeSearcher([]), _FakeSearcher([])
        factory = mock.Mock(side_effect=[hybrid, dense])
        s1 = server.get_searcher("col", "http://q:6333", "hybrid", factory=factory)
        s2 = server.get_searcher("col", "http://q:6333", "dense", factory=factory)
        self.assertIs(s1, hybrid)
        self.assertIs(s2, dense)
        self.assertEqual(factory.call_count, 2)
        self.assertEqual(factory.call_args_list[1].kwargs["mode"], "dense")

    def test_invalid_mode_rejected(self):
        with self.assertRaises(ValueError):
            server.get_searcher("col", "http://q:6333", "bogus", factory=mock.Mock())


class TestSearchEmail(unittest.TestCase):
    def test_maps_threads_and_respects_top_k(self):
        searcher = _FakeSearcher(_threads())
        rows = server.search_email("invoices?", top_k=2, searcher=searcher)
        self.assertEqual(searcher.calls, ["invoices?"])
        self.assertEqual(len(rows), 2)
        self.assertEqual(
            rows[0],
            {
                "thread_id": "t1",
                "subject": "Invoice March",
                "num_emails": 2,
                "text": "thread one body",
            },
        )
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

    def test_collection_and_mode_flow_to_get_searcher(self):
        searcher = _FakeSearcher(_threads())
        with mock.patch(
            "src.mcp_server.server.get_searcher", return_value=searcher
        ) as get_searcher:
            server.search_email("q", collection="work-rag", mode="dense")
        get_searcher.assert_called_once_with("work-rag", mode="dense")

    def test_invalid_mode_rejected(self):
        with self.assertRaises(ValueError):
            server.search_email("q", mode="bogus", searcher=_FakeSearcher([]))


class TestAnswerQuestion(unittest.TestCase):
    def test_calls_answer_from_threads_with_top_k(self):
        searcher = _FakeSearcher(_threads())
        with mock.patch(
            "src.mcp_server.server.answer_from_threads", return_value="GROUNDED"
        ) as answer:
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
        with mock.patch(
            "src.mcp_server.server.answer_from_threads",
            return_value="No relevant threads retrieved.",
        ) as answer:
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

    def test_collection_flows_to_get_searcher(self):
        searcher = _FakeSearcher(_threads())
        with (
            mock.patch("src.mcp_server.server.get_searcher", return_value=searcher) as get_searcher,
            mock.patch("src.mcp_server.server.answer_from_threads", return_value="A"),
        ):
            server.answer_question("q", collection="work-rag", k=1)
        get_searcher.assert_called_once_with("work-rag")


class _FakeCollDesc:
    def __init__(self, name):
        self.name = name


class _FakeCollInfo:
    def __init__(self, points_count):
        self.points_count = points_count


class _FakeQdrant:
    """Minimal stand-in for QdrantClient used by list_collections."""

    def __init__(self, names, counts=None, raise_on_get_collections=False, bad_info=()):
        self._names = names
        self._counts = counts or {}
        self._raise = raise_on_get_collections
        self._bad_info = set(bad_info)

    def get_collections(self):
        if self._raise:
            raise ConnectionError("refused")
        return mock.Mock(collections=[_FakeCollDesc(n) for n in self._names])

    def get_collection(self, name):
        if name in self._bad_info:
            raise RuntimeError("no such collection")
        return _FakeCollInfo(self._counts.get(name))


class TestListCollections(unittest.TestCase):
    def test_maps_rows_and_marks_default(self):
        client = _FakeQdrant(["work-rag", "personal-rag"], {"work-rag": 12, "personal-rag": 7})
        with mock.patch.dict("os.environ", {"MAILRAG_COLLECTION": "personal-rag"}, clear=True):
            rows = server.list_collections(client=client)
        self.assertEqual(
            rows,
            [
                {"name": "work-rag", "points_count": 12, "is_default": False},
                {"name": "personal-rag", "points_count": 7, "is_default": True},
            ],
        )

    def test_default_falls_back_to_manifest(self):
        client = _FakeQdrant(["a", "b"], {"a": 1, "b": 2})
        with (
            mock.patch.dict("os.environ", {}, clear=True),
            mock.patch("src.mcp_server.server.latest_manifest_collection", return_value="b"),
        ):
            rows = server.list_collections(client=client)
        self.assertTrue(rows[1]["is_default"])
        self.assertFalse(rows[0]["is_default"])

    def test_points_count_none_when_info_fails(self):
        client = _FakeQdrant(["a"], {"a": 5}, bad_info={"a"})
        with mock.patch.dict("os.environ", {}, clear=True):
            rows = server.list_collections(client=client)
        self.assertIsNone(rows[0]["points_count"])

    def test_unreachable_qdrant_raises_clear_error(self):
        client = _FakeQdrant([], raise_on_get_collections=True)
        with self.assertRaises(ValueError) as ctx:
            server.list_collections(client=client)
        self.assertIn("cannot list collections", str(ctx.exception))

    def test_builds_client_when_not_injected(self):
        client = _FakeQdrant(["a"], {"a": 4})
        with (
            mock.patch.dict("os.environ", {}, clear=True),
            mock.patch("src.config.qdrant.get_qdrant_client", return_value=client) as get_client,
        ):
            rows = server.list_collections()
        get_client.assert_called_once_with(url=server.DEFAULT_QDRANT_URL)
        self.assertEqual(rows[0]["name"], "a")

    def test_manifest_lookup_failure_degrades_to_no_default(self):
        client = _FakeQdrant(["a"], {"a": 1})
        with (
            mock.patch.dict("os.environ", {}, clear=True),
            mock.patch(
                "src.mcp_server.server.latest_manifest_collection",
                side_effect=RuntimeError("boom"),
            ),
        ):
            rows = server.list_collections(client=client)
        self.assertFalse(rows[0]["is_default"])


class _FakeMeta:
    def __init__(self, sha256, filename, mime, size, thread_id, message_id, inline=False):
        self.sha256 = sha256
        self.filename = filename
        self.mime = mime
        self.size = size
        self.thread_id = thread_id
        self.message_id = message_id
        self.inline = inline


class _FakeStore:
    def __init__(self, metas=None, fetch_result=None, fetch_raises=False):
        self._metas = metas or []
        self._fetch_result = fetch_result
        self._fetch_raises = fetch_raises
        self.closed = False
        self.list_calls = []
        self.fetch_calls = []

    def list_for(self, *, thread_id=None, message_id=None):
        self.list_calls.append((thread_id, message_id))
        return self._metas

    def fetch(self, sha256, *, extractor=None, force=False):
        self.fetch_calls.append((sha256, extractor))
        if self._fetch_raises:
            raise KeyError(f"unknown attachment {sha256}")
        return self._fetch_result

    def close(self):
        self.closed = True


class TestListAttachments(unittest.TestCase):
    def test_maps_rows_for_thread(self):
        store = _FakeStore(
            [_FakeMeta("abc", "report.pdf", "application/pdf", 1024, "t1", "m1", inline=False)]
        )
        rows = server.list_attachments(thread_id="t1", store=store)
        self.assertEqual(store.list_calls, [("t1", None)])
        self.assertEqual(
            rows,
            [
                {
                    "sha256": "abc",
                    "filename": "report.pdf",
                    "mime": "application/pdf",
                    "size": 1024,
                    "thread_id": "t1",
                    "message_id": "m1",
                    "inline": False,
                }
            ],
        )

    def test_message_id_passthrough(self):
        store = _FakeStore([])
        server.list_attachments(message_id="m9", store=store)
        self.assertEqual(store.list_calls, [(None, "m9")])

    def test_requires_an_identifier(self):
        with self.assertRaises(ValueError):
            server.list_attachments(store=_FakeStore([]))

    def test_builds_and_closes_store_when_not_injected(self):
        store = _FakeStore([])
        with mock.patch("src.mcp_server.server.AttachmentStore", return_value=store) as ctor:
            server.list_attachments(thread_id="t1")
        ctor.assert_called_once()
        self.assertTrue(store.closed)


class TestGetAttachment(unittest.TestCase):
    def test_returns_text_and_metadata_no_bytes(self):
        store = _FakeStore(
            fetch_result={
                "sha256": "abc",
                "filename": "report.pdf",
                "mime": "application/pdf",
                "size": 1024,
                "text": "extracted body",
                "text_status": "ok",
                "path": "/blobs/ab/abc",
            }
        )
        out = server.get_attachment("abc", ocr="tesseract", store=store)
        self.assertEqual(store.fetch_calls, [("abc", "tesseract")])
        self.assertEqual(
            out,
            {
                "sha256": "abc",
                "filename": "report.pdf",
                "mime": "application/pdf",
                "size": 1024,
                "text": "extracted body",
                "text_status": "ok",
            },
        )
        self.assertNotIn("path", out)  # no raw bytes / local path leaked

    def test_unknown_sha_raises_clear_value_error(self):
        store = _FakeStore(fetch_raises=True)
        with self.assertRaises(ValueError) as ctx:
            server.get_attachment("deadbeef", store=store)
        self.assertIn("unknown attachment", str(ctx.exception))

    def test_blank_sha_rejected(self):
        with self.assertRaises(ValueError):
            server.get_attachment("  ", store=_FakeStore())

    def test_builds_and_closes_store_when_not_injected(self):
        store = _FakeStore(
            fetch_result={
                "sha256": "abc",
                "filename": "f",
                "mime": "text/plain",
                "size": 1,
                "text": "t",
                "text_status": "ok",
                "path": "/x",
            }
        )
        with mock.patch("src.mcp_server.server.AttachmentStore", return_value=store):
            server.get_attachment("abc")
        self.assertTrue(store.closed)


class TestServerRegistration(unittest.TestCase):
    def test_tools_registered_with_expected_names_and_schema(self):
        srv = server.build_server()
        tools = asyncio.run(srv.list_tools())
        by_name = {t.name: t for t in tools}
        self.assertEqual(
            set(by_name),
            {
                "list_collections",
                "search_email",
                "answer_question",
                "list_attachments",
                "get_attachment",
            },
        )
        self.assertEqual(set(by_name["list_collections"].inputSchema.get("properties", {})), set())
        self.assertEqual(
            set(by_name["search_email"].inputSchema["properties"]),
            {"query", "collection", "top_k", "mode"},
        )
        self.assertEqual(
            set(by_name["answer_question"].inputSchema["properties"]),
            {"query", "collection", "k"},
        )
        self.assertEqual(
            set(by_name["list_attachments"].inputSchema["properties"]),
            {"thread_id", "message_id", "collection"},
        )
        self.assertEqual(
            set(by_name["get_attachment"].inputSchema["properties"]),
            {"sha256", "ocr"},
        )

    def test_call_tool_dispatches_into_search_email(self):
        srv = server.build_server()
        searcher = _FakeSearcher(_threads())
        with mock.patch("src.mcp_server.server.get_searcher", return_value=searcher):
            result = asyncio.run(srv.call_tool("search_email", {"query": "invoices", "top_k": 1}))
        # FastMCP (this SDK version) returns (content_blocks, structured_result).
        content_blocks, structured = result
        rows = structured["result"]
        self.assertEqual(rows[0]["thread_id"], "t1")
        # The text content mirrors the same payload.
        self.assertIn("t1", content_blocks[0].text)

    def test_call_tool_dispatches_into_list_collections(self):
        srv = server.build_server()
        rows = [{"name": "work-rag", "points_count": 3, "is_default": True}]
        with mock.patch("src.mcp_server.server.list_collections", return_value=rows):
            _, structured = asyncio.run(srv.call_tool("list_collections", {}))
        self.assertEqual(structured["result"][0]["name"], "work-rag")

    def test_call_tool_dispatches_into_answer_question(self):
        srv = server.build_server()
        searcher = _FakeSearcher(_threads())
        with (
            mock.patch("src.mcp_server.server.get_searcher", return_value=searcher),
            mock.patch("src.mcp_server.server.answer_from_threads", return_value="A"),
        ):
            _, structured = asyncio.run(
                srv.call_tool("answer_question", {"query": "when?", "k": 1})
            )
        self.assertEqual(structured["result"]["answer"], "A")

    def test_call_tool_dispatches_into_list_attachments(self):
        srv = server.build_server()
        store = _FakeStore([_FakeMeta("abc", "f.pdf", "application/pdf", 1, "t1", "m1")])
        with mock.patch("src.mcp_server.server.AttachmentStore", return_value=store):
            _, structured = asyncio.run(srv.call_tool("list_attachments", {"thread_id": "t1"}))
        self.assertEqual(structured["result"][0]["sha256"], "abc")

    def test_call_tool_dispatches_into_get_attachment(self):
        srv = server.build_server()
        store = _FakeStore(
            fetch_result={
                "sha256": "abc",
                "filename": "f.pdf",
                "mime": "application/pdf",
                "size": 1,
                "text": "body",
                "text_status": "ok",
                "path": "/x",
            }
        )
        with mock.patch("src.mcp_server.server.AttachmentStore", return_value=store):
            _, structured = asyncio.run(srv.call_tool("get_attachment", {"sha256": "abc"}))
        self.assertEqual(structured["result"]["text"], "body")


class TestCliWiring(unittest.TestCase):
    def test_mcp_verb_invokes_serve_and_sets_env(self):
        from src import cli

        with (
            mock.patch("src.mcp_server.server.serve") as serve,
            mock.patch.dict("os.environ", {}, clear=True),
        ):
            rc = cli.main(
                ["mcp", "--collection", "work-rag", "--qdrant-url", "http://localhost:6333"]
            )
            self.assertEqual(rc, 0)
            serve.assert_called_once()
            import os

            self.assertEqual(os.environ["MAILRAG_COLLECTION"], "work-rag")
            self.assertEqual(os.environ["MAILRAG_QDRANT_URL"], "http://localhost:6333")


if __name__ == "__main__":
    unittest.main()
