"""Tests for thread-aware retrieval expansion."""
import unittest
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
