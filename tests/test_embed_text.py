# tests/test_embed_text.py
"""Tests for contextual-retrieval embed-text assembly (stdlib-only)."""
import unittest

from src.ingest.embed_text import prepend_summary


class TestPrependSummary(unittest.TestCase):
    def test_prepends_summary_with_blank_line(self):
        self.assertEqual(
            prepend_summary("Approved, go ahead.", "Kevin approves the 24.09 release."),
            "Kevin approves the 24.09 release.\n\nApproved, go ahead.",
        )

    def test_none_summary_returns_text_unchanged(self):
        self.assertEqual(prepend_summary("body", None), "body")

    def test_empty_or_whitespace_summary_returns_text_unchanged(self):
        self.assertEqual(prepend_summary("body", ""), "body")
        self.assertEqual(prepend_summary("body", "   "), "body")

    def test_strips_summary_whitespace(self):
        self.assertEqual(prepend_summary("body", "  sum  "), "sum\n\nbody")


if __name__ == "__main__":
    unittest.main()
