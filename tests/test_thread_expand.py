"""Tests for thread-aware retrieval expansion."""
import unittest
from unittest.mock import MagicMock, patch
from src.query import thread_expand as te
from llama_index.core.schema import TextNode, NodeWithScore


class TestDataTypes(unittest.TestCase):
    def test_thread_email_holds_fields(self):
        e = te.ThreadEmail(
            message_id="m1", sender="a@x", to="b@y", cc="",
            date="2024-05-03T14:12:53+00:00", subject="Re: hi",
            body="Lets do it", summary="agree to meet",
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
            {"message_id": "m1", "sender": "a", "to": "b", "cc": "",
             "date": "2024-05-01T00:00:00+00:00", "subject": "hi", "text": "first",
             "summary": "s1"},
            {"message_id": "m2", "sender": "b", "to": "a", "cc": "c",
             "date": "2024-05-02T00:00:00+00:00", "subject": "Re: hi", "text": "second",
             "summary": "s2"},
        ]
        emails = te.group_into_emails(payloads)
        self.assertEqual({e.message_id for e in emails}, {"m1", "m2"})
        m1 = next(e for e in emails if e.message_id == "m1")
        self.assertEqual(m1.body, "first")
        self.assertEqual(m1.cc, "")

    def test_multi_chunk_concatenated(self):
        payloads = [
            {"message_id": "m1", "text": "part B", "date": "d", "sender": "a",
             "to": "b", "cc": "", "subject": "hi", "summary": ""},
            {"message_id": "m1", "text": "part A", "date": "d", "sender": "a",
             "to": "b", "cc": "", "subject": "hi", "summary": ""},
        ]
        emails = te.group_into_emails(payloads)
        self.assertEqual(len(emails), 1)
        # Both chunk texts present (order best-effort; content must not be lost).
        self.assertIn("part A", emails[0].body)
        self.assertIn("part B", emails[0].body)
