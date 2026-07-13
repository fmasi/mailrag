"""Tests for src/ingest/numeric.py — exact-figure normalisation (issue #82)."""

import unittest

from src.ingest.numeric import augment_numeric, normalized_numeric_tokens


class TestNormalizedNumericTokens(unittest.TestCase):
    def test_currency_with_commas(self):
        self.assertIn("210000000", normalized_numeric_tokens("Team Target FY2025 $210,000,000"))

    def test_magnitude_suffix_glued(self):
        self.assertIn("210000000", normalized_numeric_tokens("we booked 210M this quarter"))

    def test_magnitude_word_spaced(self):
        self.assertIn("210000000", normalized_numeric_tokens("about 210 million in bookings"))

    def test_finance_mm_shorthand(self):
        # "MM" is finance shorthand for million.
        self.assertIn("5000000", normalized_numeric_tokens("EBITDA of $5MM"))

    def test_billion_and_thousand(self):
        toks = normalized_numeric_tokens("1.5B pipeline, 12k seats")
        self.assertIn("1500000000", toks)
        self.assertIn("12000", toks)

    def test_bare_small_integer_is_ignored(self):
        # No comma, no decimal, no suffix -> not normalised (avoid vocab flooding).
        self.assertEqual(normalized_numeric_tokens("we have 42 people"), [])

    def test_plain_grouped_number_normalised(self):
        # A comma-grouped number normalises to its comma-free canonical form.
        self.assertEqual(normalized_numeric_tokens("cost 1,234,567"), ["1234567"])

    def test_already_canonical_is_skipped(self):
        # A bare, marker-free integer produces nothing (no-op).
        self.assertEqual(normalized_numeric_tokens("210000000"), [])

    def test_non_magnitude_word_after_number(self):
        # "210 people" -> "people" is not a magnitude; the number has no marker.
        self.assertEqual(normalized_numeric_tokens("210 people"), [])

    def test_decimal_currency(self):
        self.assertIn("1250", normalized_numeric_tokens("$1,250.00 due"))

    def test_dedup_preserves_order(self):
        toks = normalized_numeric_tokens("$210,000,000 and again 210M and 5k")
        self.assertEqual(toks, ["210000000", "5000"])

    def test_version_string_not_treated_as_number(self):
        # "v1.2" style: preceded by a dot guard avoids mangling; 1.2 has a decimal
        # but no magnitude -> fractional non-integer -> dropped.
        self.assertEqual(normalized_numeric_tokens("version 1.2 release"), [])


class TestAugmentNumeric(unittest.TestCase):
    def test_appends_canonical_token(self):
        out = augment_numeric("Team Target FY2025 $210,000,000")
        self.assertTrue(out.startswith("Team Target FY2025 $210,000,000"))
        self.assertIn("210000000", out)

    def test_no_number_unchanged(self):
        self.assertEqual(augment_numeric("hello world"), "hello world")

    def test_empty_string(self):
        self.assertEqual(augment_numeric(""), "")

    def test_surface_form_preserved(self):
        # The original surface form must survive so ordinary lexical matches work.
        out = augment_numeric("we booked 210M")
        self.assertIn("210M", out)
        self.assertIn("210000000", out)


if __name__ == "__main__":
    unittest.main()
