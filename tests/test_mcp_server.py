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

    def thread_by_id(self, thread_id):
        """Exact lookup by id — deliberately NOT recorded in ``calls``.

        Mirrors the real ``HybridSearcher``: a key fetch, independent of search.
        Keeping it out of ``calls`` is what lets tests assert that resolving a
        thread never touched retrieval.
        """
        return next((c for c in self._contexts if c.thread_id == thread_id), None)


class _StoreBackedSearcher(_FakeSearcher):
    """A searcher that can look a thread up by id WITHOUT going through search.

    Models the real world in the way ``_FakeSearcher`` does not: ``search_threads``
    returns whatever semantic retrieval happens to rank highly (which for an
    opaque message-id query is usually not the thread you asked for), while the
    thread itself is present in the store and reachable by exact key. See #109.
    """

    def __init__(self, search_returns, store):
        super().__init__(search_returns)
        self._store = store

    def thread_by_id(self, thread_id):
        return self._store.get(thread_id)


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
        # Bounded shape (issue #84): a snippet + metadata, not the full body.
        self.assertEqual(rows[0]["thread_id"], "t1")
        self.assertEqual(rows[0]["subject"], "Invoice March")
        self.assertEqual(rows[0]["num_emails"], 2)
        self.assertEqual(rows[0]["snippet"], "thread one body")  # short body: verbatim
        self.assertNotIn("text", rows[0])  # full body is opt-in only
        self.assertIn("attachment_names", rows[0])
        self.assertIn("message_ids", rows[0])
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

    def test_long_body_is_bounded_to_snippet_window(self):
        # A 5000-char body must be windowed to max_chars around the match.
        body = ("x" * 2000) + " TARGETWORD " + ("y" * 3000)
        thread = _FakeThread("t1", "S", body)
        rows = server.search_email("targetword", max_chars=200, searcher=_FakeSearcher([thread]))
        snippet = rows[0]["snippet"]
        self.assertLessEqual(len(snippet), 200 + 2)  # + ellipsis markers
        self.assertIn("targetword", snippet.lower())

    def test_max_chars_is_hard_capped(self):
        body = "z" * 20000
        thread = _FakeThread("t1", "S", body)
        rows = server.search_email("nomatch", max_chars=999999, searcher=_FakeSearcher([thread]))
        # Even a huge requested max_chars is clamped to the hard cap.
        self.assertLessEqual(len(rows[0]["snippet"]), server.HARD_SEARCH_MAX_CHARS + 1)

    def test_bad_max_chars_rejected(self):
        with self.assertRaises(ValueError):
            server.search_email(
                "q", max_chars=0, searcher=_FakeSearcher([_FakeThread("t", "s", "b")])
            )

    def test_full_returns_whole_body(self):
        body = "full body here " * 100
        thread = _FakeThread("t1", "S", body)
        rows = server.search_email("body", full=True, searcher=_FakeSearcher([thread]))
        self.assertEqual(rows[0]["text"], body)
        self.assertNotIn("snippet", rows[0])


class TestGetThread(unittest.TestCase):
    def test_returns_full_body_for_matching_thread(self):
        threads = _threads()
        out = server.get_thread("t2", searcher=_FakeSearcher(threads))
        self.assertEqual(out["thread_id"], "t2")
        self.assertEqual(out["text"], "thread two body")
        self.assertIn("attachment_names", out)

    def test_resolves_a_thread_that_semantic_search_never_returns(self):
        """Issue #109: a thread_id is a key, not a query.

        The old implementation resolved a thread by embedding the thread_id and
        running retrieval with it, then scanning the hits for an exact id match.
        A thread_id is an opaque message-id, so whether the owning thread ranks
        in the top-k is chance — measured at 3/12 (25%) against the live
        personal-rag collection, including an ordinary thread titled
        "RE: Re: Meeting next Monday".

        ``_FakeSearcher`` hid this because it returns every thread for every
        query, so the id match always succeeded. ``_StoreBackedSearcher`` models
        the real shape instead: the thread exists in the store, and semantic
        search does not surface it. Resolution must not depend on retrieval.
        """
        target = _FakeThread("needle", "Found me", "the full body", n_emails=2)
        searcher = _StoreBackedSearcher(search_returns=_threads(), store={"needle": target})
        out = server.get_thread("needle", searcher=searcher)
        self.assertEqual(out["thread_id"], "needle")
        self.assertEqual(out["text"], "the full body")
        self.assertEqual(out["num_emails"], 2)
        # And it must not have gone anywhere near retrieval to do it.
        self.assertEqual(searcher.calls, [])

    def test_unknown_thread_raises(self):
        with self.assertRaises(ValueError):
            server.get_thread("nope", searcher=_StoreBackedSearcher(_threads(), store={}))

    def test_bracket_only_id_reports_a_blank_id_not_a_missing_thread(self):
        """`<>` normalises to empty, so it is a malformed id — not a lookup miss.

        Regression guard for the normalisation added alongside the #109 fix:
        stripping brackets after the emptiness check let `<>` slip through as a
        real id, then fail with `unknown thread ''` — which sends the caller
        hunting for a thread that was never named.
        """
        for blank in ("<>", " <> ", "  ", ""):
            with self.subTest(thread_id=blank):
                with self.assertRaises(ValueError) as ctx:
                    server.get_thread(blank, searcher=_StoreBackedSearcher([], store={}))
                self.assertIn("non-empty", str(ctx.exception))

    def test_angle_bracketed_id_is_normalised(self):
        """`<abc@host>` and `abc@host` name the same thread.

        Stored thread_ids carry no angle brackets but message_id does, and
        search_email surfaces both fields — so a caller copying the wrong one
        would otherwise get `unknown thread` for a thread that is right there.
        """
        target = _FakeThread("abc@host", "S", "body")
        searcher = _StoreBackedSearcher(search_returns=[], store={"abc@host": target})
        out = server.get_thread("<abc@host>", searcher=searcher)
        self.assertEqual(out["thread_id"], "abc@host")

    def test_mode_does_not_change_the_result(self):
        """A key fetch has no ranking, so every mode must agree.

        `mode` is kept for backward compatibility; this pins that it cannot
        influence which thread comes back.
        """
        target = _FakeThread("t9", "S", "same body either way")
        searcher = _StoreBackedSearcher(search_returns=_threads(), store={"t9": target})
        results = [
            server.get_thread("t9", mode=m, searcher=searcher)
            for m in ("hybrid", "dense", "sparse")
        ]
        self.assertEqual([r["text"] for r in results], ["same body either way"] * 3)

    def test_blank_thread_id_rejected(self):
        with self.assertRaises(ValueError):
            server.get_thread("  ", searcher=_FakeSearcher(_threads()))


class TestGrepEmailTool(unittest.TestCase):
    _RESULT = {"matches": [{"subject": "hit"}], "complete": True, "scanned": 5}

    def test_delegates_to_grep_module(self):
        with mock.patch("src.mcp_server.server._grep_email", return_value=self._RESULT) as grep:
            res = server.grep_email("210,000,000", max_matches=5, regex=False)
        grep.assert_called_once_with("210,000,000", collection=None, max_matches=5, regex=False)
        self.assertEqual(res["matches"][0]["subject"], "hit")

    def test_scan_report_is_passed_through(self):
        # The caller needs `complete` to read an empty result correctly; the
        # wrapper must not flatten the report back down to a bare row list.
        with mock.patch("src.mcp_server.server._grep_email", return_value=self._RESULT):
            res = server.grep_email("nothing")
        self.assertTrue(res["complete"])
        self.assertEqual(res["scanned"], 5)

    def test_work_bounds_are_forwarded_when_given(self):
        with mock.patch("src.mcp_server.server._grep_email", return_value=self._RESULT) as grep:
            server.grep_email("x", max_files=100, max_seconds=5)
        self.assertEqual(grep.call_args.kwargs["max_files"], 100)
        self.assertEqual(grep.call_args.kwargs["max_seconds"], 5)

    def test_unset_bounds_are_left_to_the_grep_defaults(self):
        # Passing max_seconds=None through would disable the deadline outright,
        # which is the opposite of what an unset argument should mean here.
        with mock.patch("src.mcp_server.server._grep_email", return_value=self._RESULT) as grep:
            server.grep_email("x")
        self.assertNotIn("max_seconds", grep.call_args.kwargs)
        self.assertNotIn("max_files", grep.call_args.kwargs)


class TestListAttachmentsBoilerplateDefault(unittest.TestCase):
    """The MCP tool must filter decoration by default; the store must not.

    73% of rows on a real corpus are recurring inline images, so an unfiltered
    listing buries the documents. But the store stays a faithful record — the
    opinion belongs to the agent-facing tool, not the storage layer.
    """

    def test_tool_defaults_to_filtering(self):
        store = _FakeStore(
            [_FakeMeta("a", "d.pdf", "application/pdf", 9, "t1", "m1", inline=False)]
        )
        server.list_attachments(thread_id="t1", store=store)
        # Raw first (cheap exit when a thread has nothing), then the filtered
        # view only if there was something to filter.
        self.assertEqual(store.boilerplate_calls, [True, False])

    def test_tool_can_request_the_raw_list(self):
        store = _FakeStore(
            [_FakeMeta("a", "d.pdf", "application/pdf", 9, "t1", "m1", inline=False)]
        )
        server.list_attachments(thread_id="t1", include_boilerplate=True, store=store)
        self.assertEqual(store.boilerplate_calls, [True])  # raw request needs no second pass


class TestAllDecorationIsNotNoAttachments(unittest.TestCase):
    """A third kind of nothing, which must not read like the other two.

    "No attachments here", "the store was never built" and "everything here was
    decoration" are different answers. A caller told the first will stop
    looking, which is the whole failure this work exists to remove.
    """

    class _Store:
        root = "/tmp/attachments"

        def __init__(self, raw):
            self._raw = raw

        def count(self):
            return 12

        def list_for(self, *, thread_id=None, message_id=None, include_boilerplate=True):
            return self._raw if include_boilerplate else []

        def close(self):
            pass

    def test_all_decoration_raises_rather_than_returning_empty(self):
        store = self._Store([_FakeMeta("a", "logo.png", "image/png", 900, "t1", "m1", inline=True)])
        with self.assertRaises(ValueError) as ctx:
            server.list_attachments(thread_id="t1", store=store)
        msg = str(ctx.exception)
        self.assertIn("decoration", msg)
        self.assertIn("include_boilerplate", msg)

    def test_thread_with_genuinely_no_attachments_returns_empty(self):
        # Nothing raw either — so the honest answer really is "nothing here".
        self.assertEqual(server.list_attachments(thread_id="t1", store=self._Store([])), [])

    def test_include_boilerplate_true_never_raises(self):
        store = self._Store([_FakeMeta("a", "logo.png", "image/png", 900, "t1", "m1", inline=True)])
        rows = server.list_attachments(thread_id="t1", include_boilerplate=True, store=store)
        self.assertEqual([r["filename"] for r in rows], ["logo.png"])


class TestAttachmentStoreNeverBuilt(unittest.TestCase):
    """An un-ingested store must not answer like a thread with no attachments.

    The real incident: attachment text was searchable (indexing extracts it down
    its own path), so the corpus looked complete — while every list_attachments
    call returned [] because `attachments build` had never run. Absence and
    never-looked must not share a representation.
    """

    class _EmptyStore:
        root = "/tmp/attachments"

        def count(self):
            return 0

        def list_for(self, **kw):
            return []

        def fetch(self, sha256, extractor=None):
            raise KeyError(sha256)

        def close(self):
            pass

    class _PopulatedStore(_EmptyStore):
        def count(self):
            return 12

    def test_empty_store_raises_actionable_error(self):
        with self.assertRaises(ValueError) as ctx:
            server.list_attachments(thread_id="t1", store=self._EmptyStore())
        msg = str(ctx.exception)
        self.assertIn("empty", msg)
        self.assertIn("attachments build", msg)

    def test_populated_store_returns_empty_for_a_thread_without_attachments(self):
        # The guard must not turn a legitimate "no attachments here" into an error.
        self.assertEqual(server.list_attachments(thread_id="t1", store=self._PopulatedStore()), [])

    def test_get_attachment_on_empty_store_names_the_cause(self):
        with self.assertRaises(ValueError) as ctx:
            server.get_attachment("deadbeef", store=self._EmptyStore())
        self.assertIn("attachments build", str(ctx.exception))

    def test_get_attachment_on_populated_store_reports_unknown_sha(self):
        with self.assertRaises(ValueError) as ctx:
            server.get_attachment("deadbeef", store=self._PopulatedStore())
        self.assertIn("unknown attachment", str(ctx.exception))


class TestAnswerQuestionHealthcheck(unittest.TestCase):
    def test_healthcheck_runs_before_llm_by_default(self):
        searcher = _FakeSearcher(_threads())
        with (
            mock.patch("src.llm.client.healthcheck", return_value=None) as hc,
            mock.patch("src.mcp_server.server.answer_from_threads", return_value="A"),
        ):
            server.answer_question("q", searcher=searcher)
        hc.assert_called_once()

    def test_healthcheck_failure_surfaces_clear_error(self):
        from src.llm.client import LLMHealthcheckError

        searcher = _FakeSearcher(_threads())
        with (
            mock.patch(
                "src.llm.client.healthcheck",
                side_effect=LLMHealthcheckError("set RAG_LLM_API_KEY"),
            ),
            mock.patch("src.mcp_server.server.answer_from_threads") as answer,
        ):
            with self.assertRaises(LLMHealthcheckError) as ctx:
                server.answer_question("q", searcher=searcher)
        self.assertIn("RAG_LLM_API_KEY", str(ctx.exception))
        answer.assert_not_called()  # never reached the LLM

    def test_healthcheck_can_be_disabled(self):
        searcher = _FakeSearcher(_threads())
        with (
            mock.patch("src.llm.client.healthcheck") as hc,
            mock.patch("src.mcp_server.server.answer_from_threads", return_value="A"),
        ):
            server.answer_question("q", searcher=searcher, healthcheck=False)
        hc.assert_not_called()


class TestSearchAnswerParity(unittest.TestCase):
    """search_email and answer_question must retrieve the SAME threads.

    Regression for the reported bug where search_email returned hits but
    answer_question returned 0 sources / an empty answer for the same
    query+collection.
    """

    def test_same_query_yields_same_thread_set(self):
        threads = _threads()
        # Both call sites resolve to the SAME searcher (same collection/mode), so
        # search_email and answer_question retrieve the same thread set.
        fake = _FakeSearcher(threads)
        with (
            mock.patch("src.mcp_server.server.get_searcher", return_value=fake) as gs,
            mock.patch("src.mcp_server.server.answer_from_threads", return_value="grounded"),
        ):
            hits = server.search_email("project deadline", top_k=3)
            ans = server.answer_question("project deadline", k=3, healthcheck=False)
        # Both resolved a searcher for the same (default) collection + hybrid mode.
        self.assertEqual(
            gs.call_args_list,
            [mock.call(None, mode="hybrid"), mock.call(None, mode="hybrid")],
        )
        search_ids = {r["thread_id"] for r in hits}
        source_ids = {s["thread_id"] for s in ans["sources"]}
        self.assertTrue(source_ids)  # not empty when search_email found hits
        self.assertTrue(source_ids <= search_ids)  # sources are a subset of hits

    def test_sources_derive_from_materialised_list_not_generator(self):
        # A generator-returning searcher must not leave sources empty: the fix
        # materialises contexts before both consuming and slicing them.
        threads = _threads()

        class _GenSearcher:
            def search_threads(self, query):
                return (t for t in threads)  # a one-shot generator

        with mock.patch("src.mcp_server.server.answer_from_threads", return_value="A"):
            out = server.answer_question("q", k=2, searcher=_GenSearcher(), healthcheck=False)
        self.assertEqual([s["thread_id"] for s in out["sources"]], ["t1", "t2"])


class TestAnswerQuestion(unittest.TestCase):
    def test_calls_answer_from_threads_with_top_k(self):
        searcher = _FakeSearcher(_threads())
        with mock.patch(
            "src.mcp_server.server.answer_from_threads", return_value="GROUNDED"
        ) as answer:
            out = server.answer_question("when due?", k=2, searcher=searcher, healthcheck=False)
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
            out = server.answer_question("q", searcher=_FakeSearcher([]), healthcheck=False)
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
            server.answer_question("q", collection="work-rag", k=1, healthcheck=False)
        get_searcher.assert_called_once_with("work-rag", mode="hybrid")


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
    # `count` defaults to a populated store: these tests exercise lookups against
    # a store that HAS been ingested, so an empty result means "no attachments on
    # this thread" rather than "never built" (see TestAttachmentStoreNeverBuilt).
    root = "/tmp/attachments"

    def __init__(self, metas=None, fetch_result=None, fetch_raises=False, count=7):
        self._count = count
        self._metas = metas or []
        self._fetch_result = fetch_result
        self._fetch_raises = fetch_raises
        self.closed = False
        self.list_calls = []
        self.boilerplate_calls = []
        self.fetch_calls = []

    def count(self):
        return self._count

    def list_for(self, *, thread_id=None, message_id=None, include_boilerplate=True):
        self.list_calls.append((thread_id, message_id))
        self.boilerplate_calls.append(include_boilerplate)
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
        # Raw read, then the filtered view: the second only happens because this
        # thread has attachments to filter.
        self.assertEqual(store.list_calls, [("t1", None), ("t1", None)])
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
        # One call: the raw read came back empty, so there was nothing to filter
        # and no second query.
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
                # Reported alongside the text so a caller can see how much of the
                # document actually arrived (a 22MB deck can extract to 1.4k
                # chars and still say "extracted").
                "chars": 14,
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
                "get_thread",
                "grep_email",
                "answer_question",
                "list_attachments",
                "get_attachment",
            },
        )
        self.assertEqual(set(by_name["list_collections"].input_schema.get("properties", {})), set())
        self.assertEqual(
            set(by_name["search_email"].input_schema["properties"]),
            {"query", "collection", "top_k", "mode", "max_chars", "full"},
        )
        self.assertEqual(
            set(by_name["get_thread"].input_schema["properties"]),
            {"thread_id", "collection", "mode"},
        )
        self.assertEqual(
            set(by_name["grep_email"].input_schema["properties"]),
            # The work bounds are part of the tool contract: without them an
            # agent has no way to stop a scan that will not find anything.
            {"pattern", "collection", "max_matches", "regex", "max_files", "max_seconds"},
        )
        self.assertEqual(
            set(by_name["answer_question"].input_schema["properties"]),
            {"query", "collection", "k"},
        )
        self.assertEqual(
            set(by_name["list_attachments"].input_schema["properties"]),
            {"thread_id", "message_id", "collection", "include_boilerplate"},
        )
        self.assertEqual(
            set(by_name["get_attachment"].input_schema["properties"]),
            {"sha256", "ocr"},
        )

    def test_call_tool_dispatches_into_search_email(self):
        srv = server.build_server()
        searcher = _FakeSearcher(_threads())
        with mock.patch("src.mcp_server.server.get_searcher", return_value=searcher):
            result = asyncio.run(srv.call_tool("search_email", {"query": "invoices", "top_k": 1}))
        # SDK v2 returns a CallToolResult model. v1's FastMCP returned a bare
        # (content_blocks, structured_result) 2-tuple, so the old unpacking here
        # silently became "iterate the model's fields" — which is why the v1
        # form failed loudly on the upgrade rather than passing on wrong data.
        rows = result.structured_content["result"]
        self.assertEqual(rows[0]["thread_id"], "t1")
        # The text content mirrors the same payload.
        self.assertIn("t1", result.content[0].text)
        # A successful call must not be flagged as an error.
        self.assertFalse(result.is_error)

    def test_call_tool_dispatches_into_list_collections(self):
        srv = server.build_server()
        rows = [{"name": "work-rag", "points_count": 3, "is_default": True}]
        with mock.patch("src.mcp_server.server.list_collections", return_value=rows):
            result = asyncio.run(srv.call_tool("list_collections", {}))
        self.assertEqual(result.structured_content["result"][0]["name"], "work-rag")
        self.assertFalse(result.is_error)

    def test_call_tool_dispatches_into_answer_question(self):
        srv = server.build_server()
        searcher = _FakeSearcher(_threads())
        with (
            mock.patch("src.mcp_server.server.get_searcher", return_value=searcher),
            mock.patch("src.mcp_server.server.answer_from_threads", return_value="A"),
            mock.patch("src.llm.client.healthcheck", return_value=None),
        ):
            result = asyncio.run(srv.call_tool("answer_question", {"query": "when?", "k": 1}))
        self.assertEqual(result.structured_content["result"]["answer"], "A")
        self.assertFalse(result.is_error)

    def test_call_tool_dispatches_into_list_attachments(self):
        srv = server.build_server()
        store = _FakeStore([_FakeMeta("abc", "f.pdf", "application/pdf", 1, "t1", "m1")])
        with mock.patch("src.mcp_server.server.AttachmentStore", return_value=store):
            result = asyncio.run(srv.call_tool("list_attachments", {"thread_id": "t1"}))
        self.assertEqual(result.structured_content["result"][0]["sha256"], "abc")
        self.assertFalse(result.is_error)

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
            result = asyncio.run(srv.call_tool("get_attachment", {"sha256": "abc"}))
        self.assertEqual(result.structured_content["result"]["text"], "body")
        self.assertFalse(result.is_error)

    def test_call_tool_dispatches_into_grep_email(self):
        srv = server.build_server()
        rows = {
            "matches": [{"subject": "hit", "thread_id": "t1", "line": "an invoice line"}],
            "complete": True,
        }
        with mock.patch("src.mcp_server.server._grep_email", return_value=rows) as grep:
            result = asyncio.run(srv.call_tool("grep_email", {"pattern": "invoice"}))
        payload = result.structured_content["result"]
        self.assertEqual(payload["matches"][0]["subject"], "hit")
        self.assertTrue(payload["complete"])
        self.assertFalse(result.is_error)
        # The pattern reaches the grep layer rather than being dropped or reused
        # as a semantic query — grep_email is the no-embeddings path.
        self.assertEqual(grep.call_args.args[0], "invoice")

    def test_call_tool_dispatches_into_get_thread(self):
        srv = server.build_server()
        searcher = _FakeSearcher(_threads())
        with mock.patch("src.mcp_server.server.get_searcher", return_value=searcher):
            result = asyncio.run(srv.call_tool("get_thread", {"thread_id": "t1"}))
        self.assertFalse(result.is_error)
        row = result.structured_content["result"]
        self.assertEqual(row["thread_id"], "t1")
        # get_thread is the FULL-body companion to the bounded search_email, so
        # the whole text must come back, not a snippet.
        self.assertEqual(row["text"], "thread one body")

    def test_invalid_argument_round_trips_as_a_protocol_error(self):
        """A rejected argument must reach the client as ``is_error``, not a crash.

        Driven through the SDK v2 in-memory ``Client`` rather than
        ``srv.call_tool`` directly, because the two behave differently: the
        direct call raises ``ToolError`` in-process, and it is the protocol
        layer that converts that into ``CallToolResult(is_error=True)``. Only
        the client path exercises what a real MCP consumer actually sees.

        The happy-path tests above pin ``is_error is False``, which alone would
        still pass if the flag were hard-wired False; this pins the other
        direction.
        """
        from mcp import Client

        srv = server.build_server()

        async def run():
            async with Client(srv) as client:
                return await client.call_tool("search_email", {"query": "   "})

        # No get_searcher mock: search_email validates the query before it ever
        # builds a searcher, so on this path a mock would be inert and would only
        # suggest the searcher is involved in the failure.
        result = asyncio.run(run())
        self.assertTrue(result.is_error)
        # Some reason reaches the caller rather than being swallowed — but the
        # assertion deliberately stops at "non-empty text". The wording of the
        # message is ours, and the "Error executing tool <name>:" framing around
        # it is the SDK's error-formatting choice; neither is part of the MCP
        # protocol contract, so matching either would couple this test to a
        # string that can change while the behaviour it checks stays correct.
        self.assertTrue(result.content)
        # Assert the block type rather than assuming it: content is a union of
        # text/image/resource blocks, so reaching .text blindly would fail as an
        # AttributeError instead of a readable assertion.
        self.assertEqual(result.content[0].type, "text")
        self.assertTrue(result.content[0].text)

    def test_session_survives_an_errored_call(self):
        """An errored call must leave the session able to serve the next one.

        Guards the failure mode where an exception escapes the tool wrapper and
        tears down the connection — which no single-call test can detect. This
        is the unit-level counterpart of the live stdio smoke test.
        """
        from mcp import Client

        srv = server.build_server()

        async def run():
            async with Client(srv) as client:
                bad = await client.call_tool("search_email", {"query": "   "})
                good = await client.call_tool("search_email", {"query": "invoices", "top_k": 1})
                return bad, good

        # The mock is load-bearing only for the second (valid) call: the first
        # fails at search_email's blank-query guard before any searcher is built,
        # so it is not suppressing any part of the error path under test.
        with mock.patch(
            "src.mcp_server.server.get_searcher", return_value=_FakeSearcher(_threads())
        ):
            bad, good = asyncio.run(run())
        self.assertTrue(bad.is_error)
        self.assertFalse(good.is_error)
        self.assertEqual(good.structured_content["result"][0]["thread_id"], "t1")


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
