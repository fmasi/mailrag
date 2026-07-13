"""Regression tests locking in the body-decoding contract (issue #81).

The handoff flagged that raw .eml use quoted-printable, base64, and HTML bodies and
worried the decoded layer was missing / base64 leaked in as prose. The loader in
fact already honours ``Content-Transfer-Encoding`` per MIME part (via
``get_payload(decode=True)``), prefers ``text/plain`` in ``multipart/alternative``,
strips HTML, and never emits raw base64. These tests pin that behaviour so it
cannot silently regress.
"""

import email
import unittest
from email.message import EmailMessage
from email.policy import compat32

from src.data.loaders.mail_archive_x import MailArchiveXLoader


def _body(msg_bytes: bytes) -> str:
    loader = MailArchiveXLoader(eml_files=["x"], verbose=False)
    msg = email.message_from_bytes(msg_bytes, policy=compat32)
    return loader._extract_email_body_from_message(msg)


class TestBodyDecoding(unittest.TestCase):
    def test_quoted_printable_plain_is_decoded(self):
        m = EmailMessage()
        m["Subject"] = "s"
        m.set_content("Deal size is 210,000,000 today", cte="quoted-printable")
        body = _body(bytes(m))
        self.assertIn("210,000,000", body)

    def test_base64_plain_is_decoded_not_raw(self):
        m = EmailMessage()
        m["Subject"] = "s"
        m.set_content("Secret figure 42,000", cte="base64")
        body = _body(bytes(m))
        self.assertIn("42,000", body)
        # The base64 payload characters must NOT appear as prose.
        self.assertNotIn("U2VjcmV0", body)

    def test_multipart_alternative_prefers_plain_text(self):
        m = EmailMessage()
        m["Subject"] = "s"
        m.set_content("PLAIN body 210,000,000", subtype="plain", cte="quoted-printable")
        m.add_alternative("<html><body>HTML body 999</body></html>", subtype="html", cte="base64")
        body = _body(bytes(m))
        self.assertIn("210,000,000", body)
        # HTML branch not used when a plain part exists.
        self.assertNotIn("999", body)
        self.assertNotIn("<html>", body)

    def test_html_only_base64_is_decoded_and_stripped(self):
        m = EmailMessage()
        m["Subject"] = "s"
        m.set_content(
            "<html><body><p>Revenue was 55,000 USD</p></body></html>",
            subtype="html",
            cte="base64",
        )
        body = _body(bytes(m))
        self.assertIn("55,000", body)
        self.assertNotIn("<p>", body)
        self.assertNotIn("<html>", body)


if __name__ == "__main__":
    unittest.main()
