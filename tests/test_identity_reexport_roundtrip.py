# tests/test_identity_reexport_roundtrip.py
"""Re-export round-trip: parsing the SAME email content under different export
byte-layouts must yield the same (message_id, content_sha256).

This is the property the resilient Pass-2 cache relies on — a mailbox re-export
(different mbox preamble, reordered headers, CRLF/LF, extra X-headers, trailing
whitespace) must NOT change an email's stable identity, so the expensive LLM
summary is reused instead of recomputed.  Unlike the unit tests, this drives
the *production loader* end to end, so it runs in the full env (not stdlib).
"""
import contextlib
import io
import os
import tempfile
import unittest

from src.data.identity import email_identity


def _parse_identity(raw: bytes):
    """Write raw .eml bytes, parse with the real loader, return its identity."""
    from src.data.loaders.mail_archive_x import MailArchiveXLoader
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "m.eml")
        with open(path, "wb") as fh:
            fh.write(raw)
        with contextlib.redirect_stdout(io.StringIO()):
            email = list(MailArchiveXLoader(eml_files=[path]).load())[0]
    return email_identity(sender=email.sender or "", subject=email.subject or "",
                          date=email.date, body=email.body or "",
                          message_id=email.message_id or "")


_HEADERS = [
    b"Message-ID: <roundtrip-123@example.com>",
    b'From: "Alice Example" <alice@example.com>',
    b"To: bob@example.com",
    b"Subject: Quarterly sync notes",
    b"Date: Wed, 27 Mar 2024 15:27:20 +0000",
    b'Content-Type: text/plain; charset="utf-8"',
]
_BODY = b"Hi Bob,\n\nHere are the notes from today.\n\nThanks,\nAlice\n"


def _assemble(headers, body, preamble=b"", eol=b"\n"):
    head = eol.join(headers)
    return preamble + head + eol + eol + body.replace(b"\n", eol)


class TestReexportIdentityRoundTrip(unittest.TestCase):
    def setUp(self):
        base = _assemble(_HEADERS, _BODY,
                         preamble=b"From - Wed Mar 27 15:27:20 2024\n188035\t\n")
        self.mid, self.chash = _parse_identity(base)
        self.assertTrue(self.mid, "sanity: base email yielded a Message-ID")
        self.assertTrue(self.chash, "sanity: base email yielded a content hash")

    def _assert_same_identity(self, raw: bytes):
        mid, chash = _parse_identity(raw)
        self.assertEqual(mid, self.mid)
        self.assertEqual(chash, self.chash)

    def test_crlf_line_endings_unchanged(self):
        self._assert_same_identity(
            _assemble(_HEADERS, _BODY, preamble=b"From - x\n188035\t\n", eol=b"\r\n"))

    def test_missing_mbox_preamble_unchanged(self):
        self._assert_same_identity(_assemble(_HEADERS, _BODY, preamble=b""))

    def test_different_mbox_preamble_unchanged(self):
        self._assert_same_identity(
            _assemble(_HEADERS, _BODY, preamble=b"From 1700000000.deadbeef\n999999\t\n"))

    def test_reordered_headers_unchanged(self):
        reordered = [_HEADERS[3], _HEADERS[1], _HEADERS[4],
                     _HEADERS[0], _HEADERS[2], _HEADERS[5]]
        self._assert_same_identity(
            _assemble(reordered, _BODY, preamble=b"From - x\n188035\t\n"))

    def test_extra_x_headers_unchanged(self):
        extra = _HEADERS + [b"X-Mailer: Some Re-Export Tool 9.0",
                            b"X-Spam-Score: 0.1"]
        self._assert_same_identity(
            _assemble(extra, _BODY, preamble=b"From - x\n188035\t\n"))

    def test_trailing_body_whitespace_unchanged(self):
        body_ws = b"Hi Bob,   \n\nHere are the notes from today.\t\n\nThanks,\nAlice\n"
        self._assert_same_identity(
            _assemble(_HEADERS, body_ws, preamble=b"From - x\n188035\t\n"))


if __name__ == "__main__":
    unittest.main()
