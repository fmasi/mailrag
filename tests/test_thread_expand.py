"""Tests for thread-aware retrieval expansion."""

import unittest
from unittest.mock import MagicMock

from llama_index.core.schema import NodeWithScore, TextNode

from src.query import thread_expand as te


class TestDataTypes(unittest.TestCase):
    def test_thread_email_holds_fields(self):
        e = te.ThreadEmail(
            message_id="m1",
            sender="a@x",
            to="b@y",
            cc="",
            date="2024-05-03T14:12:53+00:00",
            subject="Re: hi",
            body="Lets do it",
            summary="agree to meet",
        )
        self.assertEqual(e.message_id, "m1")
        self.assertEqual(e.body, "Lets do it")

    def test_thread_context_defaults(self):
        ctx = te.ThreadContext(thread_id="t1", subject="hi", emails=[], text="")
        self.assertEqual(ctx.thread_id, "t1")
        self.assertEqual(ctx.emails, [])
        self.assertFalse(ctx.bounded)


class TestExtractThreadIds(unittest.TestCase):
    def _node(self, tid):
        return NodeWithScore(node=TextNode(text="b", metadata={"thread_id": tid}), score=1.0)

    def test_dedups_and_preserves_order(self):
        nodes = [self._node("t1"), self._node("t2"), self._node("t1")]
        self.assertEqual(te.extract_thread_ids(nodes), ["t1", "t2"])

    def test_skips_missing_thread_id(self):
        nodes = [self._node("t1"), NodeWithScore(node=TextNode(text="b", metadata={}), score=1.0)]
        self.assertEqual(te.extract_thread_ids(nodes), ["t1"])


class TestFetchThreadPayloads(unittest.TestCase):
    def _pt(self, mid, tid):
        p = MagicMock()
        p.payload = {"message_id": mid, "thread_id": tid, "text": "b"}
        return p

    def test_scrolls_and_collects_payloads_with_pagination(self):
        client = MagicMock()
        # First page returns a next-offset, second page ends (None).
        client.scroll.side_effect = [
            ([self._pt("m1", "t1")], "next"),
            ([self._pt("m2", "t1")], None),
        ]
        out = te.fetch_thread_payloads(client, "work-rag", ["t1"])
        self.assertEqual([p["message_id"] for p in out], ["m1", "m2"])
        self.assertEqual(client.scroll.call_count, 2)
        # Collection name forwarded.
        self.assertEqual(client.scroll.call_args_list[0].kwargs["collection_name"], "work-rag")

    def test_empty_thread_ids_returns_empty(self):
        client = MagicMock()
        self.assertEqual(te.fetch_thread_payloads(client, "work-rag", []), [])
        client.scroll.assert_not_called()


class TestGroupIntoEmails(unittest.TestCase):
    def test_single_chunk_emails(self):
        payloads = [
            {
                "message_id": "m1",
                "sender": "a",
                "to": "b",
                "cc": "",
                "date": "2024-05-01T00:00:00+00:00",
                "subject": "hi",
                "text": "first",
                "summary": "s1",
            },
            {
                "message_id": "m2",
                "sender": "b",
                "to": "a",
                "cc": "c",
                "date": "2024-05-02T00:00:00+00:00",
                "subject": "Re: hi",
                "text": "second",
                "summary": "s2",
            },
        ]
        emails = te.group_into_emails(payloads)
        self.assertEqual({e.message_id for e in emails}, {"m1", "m2"})
        m1 = next(e for e in emails if e.message_id == "m1")
        self.assertEqual(m1.body, "first")
        self.assertEqual(m1.cc, "")

    def test_multi_chunk_concatenated(self):
        payloads = [
            {
                "message_id": "m1",
                "text": "part B",
                "date": "d",
                "sender": "a",
                "to": "b",
                "cc": "",
                "subject": "hi",
                "summary": "",
            },
            {
                "message_id": "m1",
                "text": "part A",
                "date": "d",
                "sender": "a",
                "to": "b",
                "cc": "",
                "subject": "hi",
                "summary": "",
            },
        ]
        emails = te.group_into_emails(payloads)
        self.assertEqual(len(emails), 1)
        # Both chunk texts present (order best-effort; content must not be lost).
        self.assertIn("part A", emails[0].body)
        self.assertIn("part B", emails[0].body)


class TestOrderByDate(unittest.TestCase):
    def _e(self, mid, date):
        return te.ThreadEmail(
            message_id=mid, sender="a", to="b", cc="", date=date, subject="hi", body="x"
        )

    def test_sorts_iso_dates_ascending(self):
        emails = [
            self._e("m2", "2024-05-02T00:00:00+00:00"),
            self._e("m1", "2024-05-01T00:00:00+00:00"),
        ]
        out = te.order_by_date(emails)
        self.assertEqual([e.message_id for e in out], ["m1", "m2"])

    def test_unknown_dates_sort_last_stable(self):
        emails = [
            self._e("m2", "unknown"),
            self._e("m1", "2024-05-01T00:00:00+00:00"),
        ]
        out = te.order_by_date(emails)
        self.assertEqual([e.message_id for e in out], ["m1", "m2"])


class TestRenderThread(unittest.TestCase):
    def test_renders_attribution_header_per_email(self):
        emails = [
            te.ThreadEmail(
                message_id="m1",
                sender="Anthony",
                to="Fred",
                cc="",
                date="2015-01-08T16:05:00+00:00",
                subject="viewing",
                body="Please find details",
            ),
            te.ThreadEmail(
                message_id="m2",
                sender="Fred",
                to="Anthony",
                cc="Boss",
                date="2015-01-08T16:59:00+00:00",
                subject="Re: viewing",
                body="Lets do it",
            ),
        ]
        text = te.render_thread("t1", emails)
        self.assertIn("[Thread: viewing]", text)
        self.assertIn("From: Anthony", text)
        self.assertIn("To: Fred", text)
        self.assertIn("Cc: Boss", text)
        self.assertIn("Lets do it", text)
        # Empty cc renders as an em-dash, never blank/missing.
        self.assertIn("Cc: —", text)
        # Chronological: Anthony's email appears before Fred's reply.
        self.assertLess(text.index("Please find details"), text.index("Lets do it"))


class TestBuildThreadContexts(unittest.TestCase):
    """The exact-lookup path underneath both assemble_threads and thread_by_id."""

    def _pt(self, mid, tid, text):
        p = MagicMock()
        p.payload = {
            "message_id": mid,
            "thread_id": tid,
            "sender": "a",
            "date": "2024-05-01T00:00:00+00:00",
            "text": text,
        }
        return p

    def test_fetches_by_exact_thread_id_filter_without_any_search(self):
        """Issue #109: ids are resolved by payload filter, never by similarity."""
        client = MagicMock()
        client.scroll.side_effect = [([self._pt("m1", "needle", "the body")], None)]
        ctxs = te.build_thread_contexts(client, "work-rag", ["needle"])
        self.assertEqual([c.thread_id for c in ctxs], ["needle"])
        self.assertIn("the body", ctxs[0].text)
        # The id reached Qdrant as a filter value, not as a query string.
        flt = client.scroll.call_args.kwargs["scroll_filter"]
        self.assertEqual(flt.must[0].key, "thread_id")
        self.assertEqual(list(flt.must[0].match.any), ["needle"])

    def test_preserves_requested_order(self):
        client = MagicMock()
        client.scroll.side_effect = [
            ([self._pt("m2", "t2", "second"), self._pt("m1", "t1", "first")], None)
        ]
        ctxs = te.build_thread_contexts(client, "work-rag", ["t1", "t2"])
        self.assertEqual([c.thread_id for c in ctxs], ["t1", "t2"])

    def test_id_with_no_points_is_skipped_not_fabricated(self):
        """A stale id yields fewer threads rather than an empty/wrong one."""
        client = MagicMock()
        client.scroll.side_effect = [([self._pt("m1", "t1", "only this")], None)]
        ctxs = te.build_thread_contexts(client, "work-rag", ["t1", "ghost"])
        self.assertEqual([c.thread_id for c in ctxs], ["t1"])

    def test_empty_ids_short_circuits(self):
        client = MagicMock()
        self.assertEqual(te.build_thread_contexts(client, "work-rag", []), [])
        client.scroll.assert_not_called()


class TestAssembleThreads(unittest.TestCase):
    def test_end_to_end_with_mock_client(self):
        nodes = [
            NodeWithScore(node=TextNode(text="Lets do it", metadata={"thread_id": "t1"}), score=1.0)
        ]
        client = MagicMock()
        client.scroll.side_effect = [
            (
                [
                    self._pt("m1", "a", "2024-05-01T00:00:00+00:00", "details"),
                    self._pt("m2", "b", "2024-05-02T00:00:00+00:00", "Lets do it"),
                ],
                None,
            )
        ]
        ctxs = te.assemble_threads(nodes, client, "work-rag")
        self.assertEqual(len(ctxs), 1)
        self.assertEqual(ctxs[0].thread_id, "t1")
        self.assertEqual([e.message_id for e in ctxs[0].emails], ["m1", "m2"])
        self.assertIn("details", ctxs[0].text)
        self.assertIn("Lets do it", ctxs[0].text)

    def _pt(self, mid, sender, date, text):
        p = MagicMock()
        p.payload = {
            "message_id": mid,
            "thread_id": "t1",
            "sender": sender,
            "to": "x",
            "cc": "",
            "date": date,
            "subject": "hi",
            "text": text,
            "summary": "",
        }
        return p

    def test_no_thread_ids_returns_empty(self):
        nodes = [NodeWithScore(node=TextNode(text="b", metadata={}), score=1.0)]
        self.assertEqual(te.assemble_threads(nodes, MagicMock(), "work-rag"), [])


class TestBoundThread(unittest.TestCase):
    def _ctx(self, n_emails, body):
        emails = [
            te.ThreadEmail(
                message_id=f"m{i}",
                sender="a",
                to="b",
                cc="",
                date=f"2024-05-{i + 1:02d}T00:00:00+00:00",
                subject="hi",
                body=body,
            )
            for i in range(n_emails)
        ]
        return te.ThreadContext(
            thread_id="t1", subject="hi", emails=emails, text=te.render_thread("t1", emails)
        )

    def test_under_budget_is_unchanged(self):
        ctx = self._ctx(2, "short")
        out = te.bound_thread(ctx, max_tokens=10_000)
        self.assertFalse(out.bounded)
        self.assertEqual(out.text, ctx.text)

    def test_over_budget_with_summarizer_summarizes_tail(self):
        ctx = self._ctx(6, "x" * 400)
        called = {}

        def fake_summarizer(text: str) -> str:
            called["yes"] = True
            return "SUMMARY OF EARLIER"

        out = te.bound_thread(ctx, max_tokens=200, summarizer=fake_summarizer)
        self.assertTrue(out.bounded)
        self.assertIn("SUMMARY OF EARLIER", out.text)
        self.assertTrue(called.get("yes"))
        # Most recent email kept verbatim.
        self.assertIn(ctx.emails[-1].body, out.text)

    def test_over_budget_without_summarizer_elides_middle(self):
        ctx = self._ctx(6, "x" * 400)
        out = te.bound_thread(ctx, max_tokens=200, summarizer=None)
        self.assertTrue(out.bounded)
        self.assertIn("omitted", out.text.lower())
        # Root (first) and latest (last) kept verbatim.
        self.assertIn(ctx.emails[0].body, out.text)
        self.assertIn(ctx.emails[-1].body, out.text)
