"""Body cleanup: base64 blobs, tracking params, signatures, whitespace.

Rules adapted from msgvault (MIT, (c) 2025-2026 Wes McKinney) — see
src/data/body_cleanup.py and NOTICE. The tests that matter most are the
*negative* ones: each rule is aggressive enough to destroy real content if it
overreaches, and a cleaner that eats a signed URL or a code block is worse than
no cleaner at all.
"""

from __future__ import annotations

import unittest

from src.data.body_cleanup import (
    clean_body,
    normalize_whitespace,
    strip_base64_blobs,
    strip_signature_block,
    strip_tracking_params,
)


class TestStripBase64(unittest.TestCase):
    def test_removes_a_data_uri_image(self):
        blob = "A" * 400
        text = f"Here is the chart: <data:image/png;base64,{blob}> regards"
        out = strip_base64_blobs(text)
        self.assertNotIn(blob, out)
        self.assertIn("Here is the chart", out)
        self.assertIn("regards", out)

    def test_removes_a_bare_base64_run(self):
        text = "before " + ("QWxpY2VCb2JDaGFybGll" * 20) + " after"
        out = strip_base64_blobs(text)
        self.assertIn("before", out)
        self.assertIn("after", out)
        self.assertLess(len(out), 100)

    def test_removes_base64_containing_slashes(self):
        blob = ("aB3/xY9+" * 50) + "=="  # 400 chars, slash-bearing
        out = strip_base64_blobs(f"x {blob} y")
        self.assertNotIn(blob, out)

    def test_leaves_ordinary_prose_alone(self):
        text = "The quarterly figure is 210000000 and the plan was approved on Tuesday."
        self.assertEqual(strip_base64_blobs(text), text)

    def test_leaves_a_long_signed_url_alone(self):
        """The whole point of the two-threshold split: URL paths must survive.

        Every '/', '.', '?', '&', '-' and '_' resets the run, so a realistic
        signed URL never reaches either threshold.
        """
        url = (
            "https://my-bucket.s3.eu-west-1.amazonaws.com/reports/2026/q4/"
            "final-summary-v3.pdf?X-Amz-Algorithm=AWS4-HMAC-SHA256&"
            "X-Amz-Credential=AKIAIOSFODNN7EXAMPLE%2F20260115%2Feu-west-1%2Fs3%2F"
            "aws4_request&X-Amz-Signature=abcdef0123456789abcdef0123456789"
        )
        self.assertEqual(strip_base64_blobs(url), url)

    def test_leaves_a_long_hex_digest_alone(self):
        """A sha256 is 64 chars — well under the threshold, and often meaningful."""
        text = "commit a" + "0123456789abcdef" * 4
        self.assertEqual(strip_base64_blobs(text), text)

    def test_handles_empty_input(self):
        self.assertEqual(strip_base64_blobs(""), "")


class TestStripTrackingParams(unittest.TestCase):
    def test_removes_utm_parameters(self):
        out = strip_tracking_params("see https://example.com/post?utm_source=news&utm_medium=email")
        self.assertEqual(out, "see https://example.com/post")

    def test_removes_click_ids(self):
        out = strip_tracking_params("https://example.com/x?fbclid=abc&gclid=def")
        self.assertEqual(out, "https://example.com/x")

    def test_keeps_meaningful_parameters(self):
        """An unknown parameter may be an order id or a document reference —
        dropping it would make the mail less searchable, not more."""
        url = "https://shop.example.com/order?order_id=ABC123&utm_campaign=spring"
        self.assertEqual(
            strip_tracking_params(url), "https://shop.example.com/order?order_id=ABC123"
        )

    def test_is_case_insensitive_on_parameter_names(self):
        out = strip_tracking_params("https://example.com/x?UTM_Source=news&keep=1")
        self.assertEqual(out, "https://example.com/x?keep=1")

    def test_makes_campaign_urls_identical_so_dedup_can_fire(self):
        """The actual payoff: two newsletter copies differing only by tracking
        params become byte-identical, so exact-content chunk dedup works."""
        a = strip_tracking_params("Read more: https://news.example.com/a?utm_campaign=jan&utm_id=1")
        b = strip_tracking_params("Read more: https://news.example.com/a?utm_campaign=feb&utm_id=2")
        self.assertEqual(a, b)

    def test_preserves_trailing_sentence_punctuation(self):
        out = strip_tracking_params("Go to https://example.com/x?utm_id=9.")
        self.assertEqual(out, "Go to https://example.com/x.")

    def test_leaves_urls_without_a_query_untouched(self):
        url = "https://example.com/a/b/c"
        self.assertEqual(strip_tracking_params(url), url)

    def test_text_without_urls_is_returned_unchanged(self):
        text = "no links here at all"
        self.assertEqual(strip_tracking_params(text), text)


class TestStripSignature(unittest.TestCase):
    def test_removes_a_standard_signature_block(self):
        body = "Please review the attached plan before Friday.\n-- \nJane Doe\nCTO\n+44 7000 000000"
        out = strip_signature_block(body)
        self.assertIn("Please review", out)
        self.assertNotIn("Jane Doe", out)

    def test_tolerates_the_trailing_space_being_stripped(self):
        body = "Please review the attached plan before Friday.\n--\nJane Doe\nCTO"
        self.assertNotIn("Jane Doe", strip_signature_block(body))

    def test_keeps_the_message_when_it_is_mostly_signature(self):
        """A terse reply is mostly signature; an empty body retrieves worse than
        a signature does."""
        body = "Thanks!\n-- \nJane Doe\nCTO\nExample Ltd"
        self.assertEqual(strip_signature_block(body), body)

    def test_does_not_fire_on_an_em_dash_line_in_prose(self):
        body = (
            "We considered three options.\n---\nOption one is the cheapest and we should take it."
        )
        self.assertEqual(strip_signature_block(body), body)

    def test_handles_a_body_with_no_signature(self):
        body = "A perfectly ordinary message with enough words to clear the guard."
        self.assertEqual(strip_signature_block(body), body)


class TestNormalizeWhitespace(unittest.TestCase):
    def test_collapses_blank_line_runs_to_a_paragraph_break(self):
        self.assertEqual(normalize_whitespace("a\n\n\n\n\nb"), "a\n\nb")

    def test_preserves_a_single_paragraph_break(self):
        self.assertEqual(normalize_whitespace("a\n\nb"), "a\n\nb")

    def test_collapses_horizontal_runs(self):
        self.assertEqual(normalize_whitespace("a     b"), "a b")

    def test_blank_lines_containing_spaces_still_collapse(self):
        """Trailing horizontal whitespace must go first, or '\\n  \\n' survives."""
        self.assertEqual(normalize_whitespace("a\n  \n  \n  \nb"), "a\n\nb")

    def test_strips_the_ends(self):
        self.assertEqual(normalize_whitespace("\n\n  hello  \n\n"), "hello")


class TestCleanBody(unittest.TestCase):
    def test_applies_every_stage(self):
        body = (
            "Please review https://example.com/plan?utm_source=news before Friday, "
            "the chart is inline: data:image/png;base64," + ("A" * 300) + "\n\n\n\n"
            "-- \nJane Doe\nCTO\nExample Ltd"
        )
        out = clean_body(body)
        self.assertIn("Please review", out)
        self.assertIn("https://example.com/plan", out)
        self.assertNotIn("utm_source", out)
        self.assertNotIn("A" * 300, out)
        self.assertNotIn("Jane Doe", out)
        self.assertNotIn("\n\n\n", out)

    def test_is_idempotent(self):
        """Cleaning twice must not differ from cleaning once — otherwise a
        re-index would change content hashes and churn the whole collection."""
        body = "See https://example.com/x?utm_id=1 for details on the quarterly plan."
        once = clean_body(body)
        self.assertEqual(clean_body(once), once)

    def test_leaves_a_clean_body_untouched(self):
        body = "The Q4 figure is 210000000 and the plan is approved."
        self.assertEqual(clean_body(body), body)

    def test_handles_empty_input(self):
        self.assertEqual(clean_body(""), "")


class TestLoaderIntegration(unittest.TestCase):
    def test_the_loader_applies_cleanup_to_parsed_bodies(self):
        import os
        import shutil
        import tempfile
        from email.message import EmailMessage

        from src.data.loaders.mail_archive_x import MailArchiveXLoader

        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d, ignore_errors=True)
        m = EmailMessage()
        m["From"] = "a@x.com"
        m["To"] = "b@x.com"
        m["Subject"] = "Plan"
        m["Message-ID"] = "<p@x>"
        m["Date"] = "Tue, 15 Jan 2026 09:30:00 +0000"
        m.set_content(
            "Please review https://example.com/plan?utm_source=news&utm_medium=email "
            "before Friday and confirm the budget figure.\n-- \nJane Doe\nCTO\nExample Ltd"
        )
        path = os.path.join(d, "a.eml")
        with open(path, "wb") as fh:
            fh.write(bytes(m))

        email = MailArchiveXLoader(eml_files=[path], verbose=False).load()[0]
        self.assertIn("Please review", email.body)
        self.assertNotIn("utm_source", email.body)
        self.assertNotIn("Jane Doe", email.body)


if __name__ == "__main__":
    unittest.main()
