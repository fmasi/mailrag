"""Tests for grep_email — the literal/regex escape hatch over the raw corpus.

Builds a tiny on-disk ``.eml`` corpus (quoted-printable, base64, HTML, and an
attachment) in a temp dir, so no live services and no loader coupling. Asserts:
literal match, regex match, no-match, decoding of each encoding, subject match,
metadata + attachment-name surfacing, and the bounds (max_matches / hard cap /
invalid inputs / missing corpus).
"""

import base64
import os
import tempfile
import unittest

from src.mcp_server import grep


def _write(path, text):
    with open(path, "wb") as fh:
        fh.write(text.encode("utf-8") if isinstance(text, str) else text)


def _plain_eml(subject, sender, to, date, mid, body):
    return (
        f"From: {sender}\r\n"
        f"To: {to}\r\n"
        f"Subject: {subject}\r\n"
        f"Date: {date}\r\n"
        f"Message-ID: {mid}\r\n"
        "Content-Type: text/plain; charset=utf-8\r\n"
        "\r\n"
        f"{body}\r\n"
    )


def _qp_eml(body_qp):
    return (
        "From: qp@example.com\r\n"
        "Subject: QP message\r\n"
        "Message-ID: <qp1>\r\n"
        "Content-Type: text/plain; charset=utf-8\r\n"
        "Content-Transfer-Encoding: quoted-printable\r\n"
        "\r\n"
        f"{body_qp}\r\n"
    )


def _b64_eml(body_text):
    b64 = base64.b64encode(body_text.encode("utf-8")).decode("ascii")
    return (
        "From: b64@example.com\r\n"
        "Subject: B64 message\r\n"
        "Message-ID: <b64_1>\r\n"
        "Content-Type: text/plain; charset=utf-8\r\n"
        "Content-Transfer-Encoding: base64\r\n"
        "\r\n"
        f"{b64}\r\n"
    )


def _html_eml(html_body):
    return (
        "From: html@example.com\r\n"
        "Subject: HTML message\r\n"
        "Message-ID: <html1>\r\n"
        "Content-Type: text/html; charset=utf-8\r\n"
        "\r\n"
        f"{html_body}\r\n"
    )


def _attach_eml():
    b64 = base64.b64encode(b"Team Target FY2025 210,000,000").decode("ascii")
    return (
        "From: dana@northwind.example\r\n"
        "Subject: Q3 MBO targets partner team.xlsx\r\n"
        "Message-ID: <mbo1>\r\n"
        'Content-Type: multipart/mixed; boundary="BND"\r\n'
        "\r\n"
        "--BND\r\n"
        "Content-Type: text/plain; charset=utf-8\r\n"
        "\r\n"
        "Team / Here are MBO targets for Q3 / LMK if anything is wrong\r\n"
        "--BND\r\n"
        "Content-Type: application/vnd.ms-excel\r\n"
        "Content-Transfer-Encoding: base64\r\n"
        'Content-Disposition: attachment; filename="targets.xlsx"\r\n'
        "\r\n"
        f"{b64}\r\n"
        "--BND--\r\n"
    )


class _Corpus:
    """Context manager building a temp .eml corpus and cleaning it up."""

    def __enter__(self):
        self.dir = tempfile.mkdtemp(prefix="grep_eml_")
        _write(
            os.path.join(self.dir, "plain.eml"),
            _plain_eml(
                "Invoice",
                "alice@x.com",
                "bob@y.com",
                "Mon, 1 Jan 2025 10:00:00 +0000",
                "<inv1>",
                "The March Acme invoice #4021 was $12,480.",
            ),
        )
        # QP encodes '=' as '=3D' and soft-wraps with '='.
        _write(os.path.join(self.dir, "qp.eml"), _qp_eml("Cost was =2450 =3D fifty."))
        _write(os.path.join(self.dir, "b64.eml"), _b64_eml("secret token ABC-999-XYZ here"))
        _write(
            os.path.join(self.dir, "html.eml"),
            _html_eml("<html><body><p>Revenue <b>210,000,000</b> plan</p></body></html>"),
        )
        _write(os.path.join(self.dir, "mbo.eml"), _attach_eml())
        return self.dir

    def __exit__(self, *a):
        import shutil

        shutil.rmtree(self.dir, ignore_errors=True)


class TestGrepEmail(unittest.TestCase):
    def test_literal_match_in_plain_body(self):
        with _Corpus() as root:
            rows = grep.grep_email("4021", root=root)
        subjects = [r["subject"] for r in rows]
        self.assertIn("Invoice", subjects)
        hit = next(r for r in rows if r["subject"] == "Invoice")
        self.assertTrue(any("4021" in m for m in hit["matches"]))

    def test_metadata_surfaced(self):
        with _Corpus() as root:
            rows = grep.grep_email("4021", root=root)
        hit = next(r for r in rows if r["subject"] == "Invoice")
        self.assertEqual(hit["from"], "alice@x.com")
        self.assertEqual(hit["to"], "bob@y.com")
        self.assertEqual(hit["message_id"], "<inv1>")
        self.assertIn("2025", hit["date"])
        self.assertEqual(hit["attachment_names"], [])

    def test_decodes_quoted_printable(self):
        with _Corpus() as root:
            rows = grep.grep_email("fifty", root=root)
        self.assertTrue(any(r["subject"] == "QP message" for r in rows))

    def test_decodes_base64_body(self):
        with _Corpus() as root:
            rows = grep.grep_email("ABC-999-XYZ", root=root)
        self.assertTrue(any(r["subject"] == "B64 message" for r in rows))

    def test_strips_html_and_matches_text(self):
        with _Corpus() as root:
            rows = grep.grep_email("210,000,000", root=root)
        subjects = [r["subject"] for r in rows]
        self.assertIn("HTML message", subjects)

    def test_regex_match(self):
        with _Corpus() as root:
            rows = grep.grep_email(r"\$[0-9,]+", regex=True, root=root)
        self.assertTrue(any(r["subject"] == "Invoice" for r in rows))

    def test_invalid_regex_raises(self):
        with _Corpus() as root:
            with self.assertRaises(ValueError):
                grep.grep_email("(unclosed", regex=True, root=root)

    def test_no_match_returns_empty(self):
        with _Corpus() as root:
            rows = grep.grep_email("thisstringappearsnowhere", root=root)
        self.assertEqual(rows, [])

    def test_subject_only_match_surfaces(self):
        with _Corpus() as root:
            rows = grep.grep_email("MBO targets", root=root)
        self.assertTrue(any(r["subject"].startswith("Q3 MBO") for r in rows))

    def test_attachment_names_surfaced_but_bytes_not_searched(self):
        with _Corpus() as root:
            # Match the body so the message surfaces; assert the attachment name
            # is reported even though its cell contents are NOT searched (#80).
            rows = grep.grep_email("MBO targets", root=root)
            mbo = next(r for r in rows if r["subject"].startswith("Q3 MBO"))
            self.assertIn("targets.xlsx", mbo["attachment_names"])
            # The number lives ONLY in the attachment bytes -> grep must NOT find it.
            att_hits = grep.grep_email("210,000,000", root=root)
            self.assertFalse(any(r["subject"].startswith("Q3 MBO") for r in att_hits))

    def test_blank_pattern_rejected(self):
        with _Corpus() as root:
            with self.assertRaises(ValueError):
                grep.grep_email("   ", root=root)

    def test_max_matches_bounds_results(self):
        with _Corpus() as root:
            rows = grep.grep_email("e", regex=False, max_matches=1, root=root)
        self.assertEqual(len(rows), 1)

    def test_max_matches_hard_capped(self):
        with _Corpus() as root:
            rows = grep.grep_email("e", max_matches=10**9, root=root)
        self.assertLessEqual(len(rows), grep._HARD_MAX_MATCHES)

    def test_missing_corpus_raises_clear_error(self):
        with self.assertRaises(ValueError) as ctx:
            grep.grep_email("x", root="/nonexistent/path/here")
        self.assertIn("MAILRAG_EML_ROOT", str(ctx.exception))

    def test_env_override_resolves_root(self):
        with _Corpus() as root:
            from unittest import mock

            with mock.patch.dict(os.environ, {"MAILRAG_EML_ROOT": root}, clear=True):
                self.assertEqual(grep.resolve_eml_root(), root)


if __name__ == "__main__":
    unittest.main()
