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


class TestSubjectSlug(unittest.TestCase):
    def test_strips_prefixes_and_normalizes(self):
        from src.data.threading import subject_slug
        self.assertEqual(subject_slug("Re: FW:  The  Plan "), "the plan")
        self.assertEqual(subject_slug("Fwd: Re: Budget"), "budget")

    def test_bare_subject_lowercased(self):
        from src.data.threading import subject_slug
        self.assertEqual(subject_slug("Monthly Report"), "monthly report")

    def test_empty_and_none_return_empty(self):
        from src.data.threading import subject_slug
        self.assertEqual(subject_slug(""), "")
        self.assertEqual(subject_slug(None), "")

    def test_collapses_internal_whitespace(self):
        from src.data.threading import subject_slug
        self.assertEqual(subject_slug("  Hello   World  "), "hello world")


class TestComputeThreadIdSubjectFallback(unittest.TestCase):
    def test_header_less_falls_back_to_subject(self):
        """Both Re:-prefixed and bare subject must map to the same thread."""
        a = threading.compute_thread_id("", "", "", subject="Re: The Plan")
        b = threading.compute_thread_id("", "", "", subject="The Plan")
        self.assertEqual(a, b)
        self.assertNotEqual(a, "")

    def test_subject_fallback_uses_subj_prefix(self):
        """Thread id must start with 'subj:' for header-less grouping."""
        tid = threading.compute_thread_id("", "", "", subject="Budget Review")
        self.assertTrue(tid.startswith("subj:"), tid)

    def test_existing_3arg_behaviour_unchanged(self):
        """Passing subject= must not alter the result when headers produce a key."""
        without = threading.compute_thread_id("<1>", "", "")
        with_subj = threading.compute_thread_id("<1>", "", "", subject="Whatever")
        self.assertEqual(without, with_subj)

    def test_no_subject_still_returns_empty(self):
        """When headers are empty AND subject is empty, result is still ''."""
        self.assertEqual(threading.compute_thread_id("", "", "", subject=""), "")
        self.assertEqual(threading.compute_thread_id("", "", ""), "")

    def test_subject_with_only_prefix_returns_empty(self):
        """A subject that normalises to '' (e.g. 'Re:') should not form a key."""
        tid = threading.compute_thread_id("", "", "", subject="Re:")
        self.assertEqual(tid, "")


if __name__ == "__main__":
    unittest.main()
