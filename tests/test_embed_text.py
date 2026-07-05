# tests/test_embed_text.py
"""Tests for contextual-retrieval embed-text assembly (stdlib-only)."""

import unittest

from src.ingest.embed_text import embed_max_length, prepend_summary


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


class TestEmbedMaxLength(unittest.TestCase):
    def test_body_only_default_unchanged(self):
        """Body-only builds: max_length == chunk_size, no headroom added."""
        self.assertEqual(embed_max_length(512, False), 512)

    def test_embed_summary_adds_headroom(self):
        """Summary prepend: default 256-token headroom added."""
        self.assertEqual(embed_max_length(512, True), 512 + 256)

    def test_custom_headroom(self):
        """Custom headroom kwarg overrides the 256 default."""
        self.assertEqual(embed_max_length(512, True, headroom=128), 512 + 128)

    def test_override_wins_with_embed_summary(self):
        """Explicit override always wins, even when embed_summary=True."""
        self.assertEqual(embed_max_length(512, True, override=700), 700)

    def test_override_wins_body_only(self):
        """Explicit override wins for body-only builds too."""
        self.assertEqual(embed_max_length(512, False, override=4096), 4096)

    def test_clamped_to_bge_m3_ceiling(self):
        """Result is clamped to bge-m3's 8192-token ceiling."""
        self.assertEqual(embed_max_length(8000, True), 8192)

    def test_override_also_clamped(self):
        """An override exceeding 8192 is clamped to 8192."""
        self.assertEqual(embed_max_length(512, True, override=9000), 8192)

    def test_body_only_clamped_to_max(self):
        # chunk_size above bge-m3's 8192 ceiling is clamped even with no summary
        self.assertEqual(embed_max_length(9000, False), 8192)


if __name__ == "__main__":
    unittest.main()
