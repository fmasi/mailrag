"""Tests for thread-aware retrieval expansion."""
import unittest
from src.query import thread_expand as te


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
