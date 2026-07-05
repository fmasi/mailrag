import unittest

from src.data.threading import assign_subject_fallback_thread_ids


class _Email:
    def __init__(self, subject="", thread_id=None, message_id="", in_reply_to="", references=""):
        self.subject = subject
        self.thread_id = thread_id
        self.message_id = message_id
        self.in_reply_to = in_reply_to
        self.references = references


class TestAssignThreadIds(unittest.TestCase):
    def test_assigns_when_missing_and_groups_by_subject(self):
        a = _Email(subject="Re: Lunch plans")
        b = _Email(subject="Lunch plans")
        n = assign_subject_fallback_thread_ids([a, b])
        self.assertEqual(n, 2)
        self.assertTrue(a.thread_id)
        self.assertEqual(a.thread_id, b.thread_id)  # subject-slug groups them

    def test_preserves_existing_thread_id(self):
        a = _Email(subject="x", thread_id="keep-me")
        n = assign_subject_fallback_thread_ids([a])
        self.assertEqual(n, 0)
        self.assertEqual(a.thread_id, "keep-me")


if __name__ == "__main__":
    unittest.main()
