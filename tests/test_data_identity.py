# tests/test_data_identity.py
"""Tests for content-based email identity (stdlib-only).

``content_sha256`` must be stable across byte-level reformatting (line endings,
trailing whitespace, header re-encoding) so a re-exported mailbox reuses the
expensive Pass-2 cache instead of recomputing it.
"""
from datetime import datetime, timezone
import unittest

from src.data.identity import content_sha256


class TestContentSha256(unittest.TestCase):
    def test_stable_across_line_endings_and_trailing_whitespace(self):
        a = content_sha256(sender="a@x.com", subject="Hi", date=None,
                           body="line one\r\nline two   \r\n")
        b = content_sha256(sender="a@x.com", subject="Hi", date=None,
                           body="line one\nline two\n")
        self.assertEqual(a, b)

    def test_stable_across_sender_whitespace(self):
        a = content_sha256(sender="Alice  <a@x.com>", subject="Hi", date=None, body="x")
        b = content_sha256(sender="Alice <a@x.com>", subject="Hi", date=None, body="x")
        self.assertEqual(a, b)

    def test_datetime_and_its_isoformat_are_equivalent(self):
        dt = datetime(2024, 3, 27, 15, 27, tzinfo=timezone.utc)
        a = content_sha256(sender="a", subject="s", date=dt, body="b")
        b = content_sha256(sender="a", subject="s", date=dt.isoformat(), body="b")
        self.assertEqual(a, b)

    def test_changes_when_subject_changes(self):
        a = content_sha256(sender="a", subject="One", date=None, body="b")
        b = content_sha256(sender="a", subject="Two", date=None, body="b")
        self.assertNotEqual(a, b)

    def test_changes_when_body_changes(self):
        a = content_sha256(sender="a", subject="s", date=None, body="hello")
        b = content_sha256(sender="a", subject="s", date=None, body="hella")
        self.assertNotEqual(a, b)

    def test_returns_hex_digest(self):
        h = content_sha256(sender="a", subject="s", date=None, body="b")
        self.assertEqual(len(h), 64)
        int(h, 16)  # raises if not hex


class TestEmailIdentity(unittest.TestCase):
    def test_normalizes_message_id_and_hashes_content(self):
        from src.data.identity import email_identity
        mid, ch = email_identity(sender="a", subject="s", date=None, body="b",
                                 message_id="<M@x>")
        self.assertEqual(mid, "M@x")  # surrounding <> stripped
        self.assertEqual(ch, content_sha256(sender="a", subject="s", date=None, body="b"))

    def test_empty_message_id_becomes_none(self):
        from src.data.identity import email_identity
        mid, _ = email_identity(sender="a", subject="s", date=None, body="b",
                                message_id="")
        self.assertIsNone(mid)


if __name__ == "__main__":
    unittest.main()
