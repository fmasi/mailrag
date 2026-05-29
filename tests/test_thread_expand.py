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
