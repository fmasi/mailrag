"""Tests for email thread-id computation (stdlib-only, host-runnable)."""
import unittest

from src.data import threading


class TestNormalizeMessageId(unittest.TestCase):
    def test_strips_angle_brackets_and_whitespace(self):
        self.assertEqual(threading.normalize_message_id("  <abc@x>  "), "abc@x")

    def test_keeps_bare_id(self):
        self.assertEqual(threading.normalize_message_id("abc@x"), "abc@x")

    def test_empty_and_none_become_empty(self):
        self.assertEqual(threading.normalize_message_id(""), "")
        self.assertEqual(threading.normalize_message_id(None), "")


class TestComputeThreadId(unittest.TestCase):
    def test_uses_references_root_when_present(self):
        tid = threading.compute_thread_id(
            message_id="<c@x>",
            in_reply_to="<b@x>",
            references="<root@x> <b@x>",
        )
        self.assertEqual(tid, "root@x")

    def test_falls_back_to_in_reply_to_when_no_references(self):
        self.assertEqual(
            threading.compute_thread_id("<c@x>", "<b@x>", ""), "b@x"
        )
        self.assertEqual(
            threading.compute_thread_id("<c@x>", "<b@x>", None), "b@x"
        )

    def test_falls_back_to_own_message_id_at_thread_root(self):
        self.assertEqual(
            threading.compute_thread_id("<root@x>", "", ""), "root@x"
        )

    def test_root_email_and_its_reply_share_thread_id(self):
        root = threading.compute_thread_id("<root@x>", "", "")
        reply = threading.compute_thread_id("<c@x>", "<root@x>", "<root@x>")
        self.assertEqual(root, reply)

    def test_empty_when_no_identifiers(self):
        self.assertEqual(threading.compute_thread_id("", "", ""), "")


if __name__ == "__main__":
    unittest.main()
