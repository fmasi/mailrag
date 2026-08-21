"""Tests for grep_email — the literal/regex escape hatch over the raw corpus.

Builds a tiny on-disk ``.eml`` corpus (quoted-printable, base64, HTML, and an
attachment) in a temp dir, so no live services and no loader coupling. Asserts:
literal match, regex match, no-match, decoding of each encoding, subject match,
metadata + attachment-name surfacing, and the bounds (max_matches / hard cap /
invalid inputs / missing corpus).

``TestGrepScanBounds`` covers the work bounds and the scan report — the part
that lets a caller tell "this needle is not in the corpus" apart from "the scan
ran out of budget", which an unqualified empty list could not express.
"""

import base64
import os
import tempfile
import unittest

from src.mcp_server import grep


# Real Mail Archive X exports prepend an mbox "From " separator AND a stray
# byte-count line before the RFC 2822 headers. Every file in the live corpus has
# one. The original fixtures here were hand-written without it, so the suite
# passed while grep reported "(no subject)" and empty sender/date for 100% of
# real mail. Fixtures now carry the preamble by default: a test corpus that is
# cleaner than the real one tests the wrong program.
def _mbox_preamble(body_len=1234):
    return f"From <sender@example.com> Tue Aug 11 16:33:18 2025\r\n{body_len}     \r\n"


def _write(path, text, *, preamble=True):
    raw = text.encode("utf-8") if isinstance(text, str) else text
    if preamble:
        raw = _mbox_preamble(len(raw)).encode("ascii") + raw
    with open(path, "wb") as fh:
        fh.write(raw)


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
        # One plain RFC 2822 file with no envelope, so stripping cannot regress
        # the non-mbox case.
        _write(
            os.path.join(self.dir, "nopreamble.eml"),
            _plain_eml(
                "Bare RFC822",
                "carol@x.com",
                "dan@y.com",
                "Tue, 2 Jan 2025 10:00:00 +0000",
                "<bare1>",
                "no mbox envelope here, token BARE-7788",
            ),
            preamble=False,
        )
        return self.dir

    def __exit__(self, *a):
        import shutil

        shutil.rmtree(self.dir, ignore_errors=True)


def _matches(*args, **kwargs):
    """Call grep_email and return just the rows (the scan report is asserted below)."""
    return grep.grep_email(*args, **kwargs)["matches"]


class TestGrepEmail(unittest.TestCase):
    def test_literal_match_in_plain_body(self):
        with _Corpus() as root:
            rows = _matches("4021", root=root)
        subjects = [r["subject"] for r in rows]
        self.assertIn("Invoice", subjects)
        hit = next(r for r in rows if r["subject"] == "Invoice")
        self.assertTrue(any("4021" in m for m in hit["matches"]))

    def test_metadata_surfaced(self):
        with _Corpus() as root:
            rows = _matches("4021", root=root)
        hit = next(r for r in rows if r["subject"] == "Invoice")
        self.assertEqual(hit["from"], "alice@x.com")
        self.assertEqual(hit["to"], "bob@y.com")
        self.assertEqual(hit["message_id"], "<inv1>")
        self.assertIn("2025", hit["date"])
        self.assertEqual(hit["attachment_names"], [])

    def test_decodes_quoted_printable(self):
        with _Corpus() as root:
            rows = _matches("fifty", root=root)
        self.assertTrue(any(r["subject"] == "QP message" for r in rows))

    def test_decodes_base64_body(self):
        with _Corpus() as root:
            rows = _matches("ABC-999-XYZ", root=root)
        self.assertTrue(any(r["subject"] == "B64 message" for r in rows))

    def test_strips_html_and_matches_text(self):
        with _Corpus() as root:
            rows = _matches("210,000,000", root=root)
        subjects = [r["subject"] for r in rows]
        self.assertIn("HTML message", subjects)

    def test_regex_match(self):
        with _Corpus() as root:
            rows = _matches(r"\$[0-9,]+", regex=True, root=root)
        self.assertTrue(any(r["subject"] == "Invoice" for r in rows))

    def test_invalid_regex_raises(self):
        with _Corpus() as root:
            with self.assertRaises(ValueError):
                grep.grep_email("(unclosed", regex=True, root=root)

    def test_no_match_returns_empty(self):
        with _Corpus() as root:
            rows = _matches("thisstringappearsnowhere", root=root)
        self.assertEqual(rows, [])

    def test_subject_only_match_surfaces(self):
        with _Corpus() as root:
            rows = _matches("MBO targets", root=root)
        self.assertTrue(any(r["subject"].startswith("Q3 MBO") for r in rows))

    def test_attachment_names_surfaced_but_bytes_not_searched(self):
        with _Corpus() as root:
            # Match the body so the message surfaces; assert the attachment name
            # is reported even though its cell contents are NOT searched (#80).
            rows = _matches("MBO targets", root=root)
            mbo = next(r for r in rows if r["subject"].startswith("Q3 MBO"))
            self.assertIn("targets.xlsx", mbo["attachment_names"])
            # The number lives ONLY in the attachment bytes -> grep must NOT find it.
            att_hits = _matches("210,000,000", root=root)
            self.assertFalse(any(r["subject"].startswith("Q3 MBO") for r in att_hits))

    def test_blank_pattern_rejected(self):
        with _Corpus() as root:
            with self.assertRaises(ValueError):
                grep.grep_email("   ", root=root)

    def test_max_matches_bounds_results(self):
        with _Corpus() as root:
            rows = _matches("e", regex=False, max_matches=1, root=root)
        self.assertEqual(len(rows), 1)

    def test_max_matches_hard_capped(self):
        with _Corpus() as root:
            rows = _matches("e", max_matches=10**9, root=root)
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


class TestMboxPreamble(unittest.TestCase):
    """Headers must survive the mbox envelope every real export carries.

    Without stripping it, Python's parser stops at the first non-header line and
    silently loses every header — subject, sender, date, message-id — while the
    entire raw file, base64 attachment payloads included, becomes "body". That
    was the live behaviour for 100% of a real corpus.
    """

    def test_headers_survive_the_envelope(self):
        with _Corpus() as root:
            rows = _matches("4021", root=root)
        hit = next(r for r in rows if "Invoice" in r["subject"])
        self.assertEqual(hit["from"], "alice@x.com")
        self.assertEqual(hit["message_id"], "<inv1>")
        self.assertIn("2025", hit["date"])

    def test_subject_is_not_lost_to_the_envelope(self):
        with _Corpus() as root:
            rows = _matches("MBO targets", root=root)
        self.assertTrue(any(r["subject"].startswith("Q3 MBO") for r in rows))

    def test_body_is_decoded_not_raw_bytes(self):
        # The giveaway of a failed parse: MIME headers leaking into the body,
        # which is how grep appeared to be "finding attachments" in the field.
        with _Corpus() as root:
            rows = _matches("4021", root=root)
        hit = next(r for r in rows if "Invoice" in r["subject"])
        joined = " ".join(hit["matches"])
        self.assertNotIn("Content-Type:", joined)
        self.assertNotIn("Received:", joined)

    def test_files_without_an_envelope_still_parse(self):
        with _Corpus() as root:
            rows = _matches("BARE-7788", root=root)
        self.assertEqual([r["subject"] for r in rows], ["Bare RFC822"])
        self.assertEqual(rows[0]["from"], "carol@x.com")


class TestGrepScanBounds(unittest.TestCase):
    """The scan report, and the three ways a scan can stop early."""

    CORPUS_FILES = 6  # .eml files written by _Corpus

    def test_full_scan_reports_complete(self):
        with _Corpus() as root:
            res = grep.grep_email("thisstringappearsnowhere", root=root)
        self.assertTrue(res["complete"])
        self.assertEqual(res["stop_reason"], "complete")
        self.assertEqual(res["scanned"], self.CORPUS_FILES)
        self.assertEqual(res["corpus_files"], self.CORPUS_FILES)
        self.assertEqual(res["matches"], [])

    def test_max_files_truncates_scan(self):
        with _Corpus() as root:
            res = grep.grep_email("e", max_files=2, root=root)
        self.assertEqual(res["scanned"], 2)
        self.assertEqual(res["corpus_files"], self.CORPUS_FILES)
        self.assertFalse(res["complete"])
        self.assertEqual(res["stop_reason"], "max_files")

    def test_absent_needle_under_a_file_budget_is_not_reported_as_complete(self):
        # The whole point of the report: an empty result from a truncated scan
        # must NOT look like an empty result from a full scan, or a caller will
        # turn "I ran out of budget" into a confident "it is not there".
        with _Corpus() as root:
            truncated = grep.grep_email("thisstringappearsnowhere", max_files=1, root=root)
            full = grep.grep_email("thisstringappearsnowhere", root=root)
        self.assertEqual(truncated["matches"], full["matches"])
        self.assertFalse(truncated["complete"])
        self.assertTrue(full["complete"])

    def test_match_cap_marks_the_scan_incomplete(self):
        # Stopping on max_matches means the corpus was NOT exhausted either.
        with _Corpus() as root:
            res = grep.grep_email("e", max_matches=1, root=root)
        self.assertEqual(len(res["matches"]), 1)
        self.assertEqual(res["stop_reason"], "max_matches")
        self.assertFalse(res["complete"])

    def test_deadline_stops_the_scan(self):
        # An already-expired budget stops before the first file is parsed, which
        # keeps the assertion deterministic rather than timing-dependent.
        with _Corpus() as root:
            res = grep.grep_email("e", max_seconds=1e-9, root=root)
        self.assertEqual(res["stop_reason"], "deadline")
        self.assertEqual(res["scanned"], 0)
        self.assertFalse(res["complete"])
        self.assertEqual(res["matches"], [])

    def test_max_seconds_none_disables_the_deadline(self):
        with _Corpus() as root:
            res = grep.grep_email("thisstringappearsnowhere", max_seconds=None, root=root)
        self.assertTrue(res["complete"])
        self.assertEqual(res["scanned"], self.CORPUS_FILES)

    def test_non_positive_max_files_rejected(self):
        # Rejected rather than clamped to 1, matching max_seconds: a caller
        # passing 0 means "none", and scanning one file would answer a
        # different question without saying so.
        with _Corpus() as root:
            for bad in (0, -1):
                with self.subTest(max_files=bad):
                    with self.assertRaises(ValueError):
                        grep.grep_email("e", max_files=bad, root=root)

    def test_max_files_above_corpus_size_completes(self):
        with _Corpus() as root:
            res = grep.grep_email("thisstringappearsnowhere", max_files=10_000, root=root)
        self.assertTrue(res["complete"])
        self.assertEqual(res["stop_reason"], "complete")
        self.assertEqual(res["scanned"], self.CORPUS_FILES)

    def test_elapsed_includes_file_discovery(self):
        """elapsed_s is the wall time the CALLER waited, walk included.

        Discovery is ~0.25s over 73k files, so timing it from after the walk
        under-reports the cost of exactly the calls that hurt. Proven by making
        discovery slow rather than by asserting a number is positive, which a
        fast temp corpus rounds to 0.0 anyway.
        """
        import time
        from unittest import mock

        real = grep._discover_eml

        def slow(root):
            time.sleep(0.05)
            return real(root)

        with _Corpus() as root:
            with mock.patch.object(grep, "_discover_eml", slow):
                res = grep.grep_email("e", root=root)
        self.assertGreaterEqual(res["elapsed_s"], 0.05)

    def test_non_positive_max_seconds_rejected(self):
        with _Corpus() as root:
            with self.assertRaises(ValueError):
                grep.grep_email("e", max_seconds=0, root=root)

    def test_max_seconds_clamped_to_hard_cap(self):
        with _Corpus() as root:
            res = grep.grep_email("e", max_seconds=10**9, root=root)
        # Clamping must not itself stop the scan; the cap only bounds the wait.
        self.assertLessEqual(res["elapsed_s"], grep._HARD_MAX_SECONDS)
        self.assertEqual(res["corpus_files"], self.CORPUS_FILES)

    def test_root_is_reported(self):
        with _Corpus() as root:
            res = grep.grep_email("e", root=root)
        self.assertEqual(res["root"], root)


if __name__ == "__main__":
    unittest.main()
