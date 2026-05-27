"""Tests for the time-limit helpers in scripts/batch_index_to_vector_store.py.

Covers:
- _parse_time_limit  — duration string parsing
- _time_budget_exhausted — stop-before-next-batch logic
- _fmt_duration      — human-readable duration formatting
"""

import sys
import os
import unittest

# The script lives in scripts/ which is not a package; add it to sys.path.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from batch_index_to_vector_store import (  # noqa: E402
    _fmt_duration,
    _parse_time_limit,
    _time_budget_exhausted,
)


class TestParsTimeLimit(unittest.TestCase):
    """_parse_time_limit converts human-readable strings to seconds."""

    def test_hours(self):
        self.assertAlmostEqual(_parse_time_limit("3h"), 10800.0)

    def test_minutes(self):
        self.assertAlmostEqual(_parse_time_limit("90m"), 5400.0)

    def test_seconds_suffix(self):
        self.assertAlmostEqual(_parse_time_limit("5400s"), 5400.0)

    def test_plain_integer(self):
        self.assertAlmostEqual(_parse_time_limit("5400"), 5400.0)

    def test_plain_float(self):
        self.assertAlmostEqual(_parse_time_limit("90.5"), 90.5)

    def test_fractional_hours(self):
        # 1.5h == 5400s
        self.assertAlmostEqual(_parse_time_limit("1.5h"), 5400.0)

    def test_strips_whitespace(self):
        self.assertAlmostEqual(_parse_time_limit("  2h  "), 7200.0)

    def test_one_minute(self):
        self.assertAlmostEqual(_parse_time_limit("1m"), 60.0)

    def test_invalid_raises(self):
        with self.assertRaises(ValueError):
            _parse_time_limit("abc")

    def test_invalid_unit_raises(self):
        with self.assertRaises(ValueError):
            _parse_time_limit("3d")


class TestTimeBudgetExhausted(unittest.TestCase):
    """_time_budget_exhausted decides whether to skip the next batch."""

    def test_no_batch_times_always_false(self):
        """First batch always runs — no data to estimate with."""
        self.assertFalse(_time_budget_exhausted(9999.0, [], 10.0))

    def test_fits_within_budget(self):
        """elapsed + mean_batch < limit → should NOT stop."""
        # elapsed=100s, mean_batch=50s, limit=200s → 150 < 200 → False
        self.assertFalse(_time_budget_exhausted(100.0, [50.0], 200.0))

    def test_exactly_at_limit_stops(self):
        """elapsed + mean_batch == limit → should stop (not strictly less than)."""
        # elapsed=150s, mean_batch=50s, limit=200s → 200 == 200 → True
        self.assertTrue(_time_budget_exhausted(150.0, [50.0], 200.0))

    def test_exceeds_budget(self):
        """elapsed + mean_batch > limit → should stop."""
        # elapsed=180s, mean_batch=50s, limit=200s → 230 > 200 → True
        self.assertTrue(_time_budget_exhausted(180.0, [50.0], 200.0))

    def test_uses_mean_of_multiple_batches(self):
        """ETA is the mean of all completed batches, not just the last."""
        # Batch times: 40, 60, 50 → mean=50
        # elapsed=155, limit=200 → 155+50=205 > 200 → True
        self.assertTrue(_time_budget_exhausted(155.0, [40.0, 60.0, 50.0], 200.0))

    def test_mean_fits_when_last_batch_was_slow(self):
        """A single slow outlier doesn't dominate when other batches were fast."""
        # Batch times: 10, 10, 10, 100 → mean=32.5
        # elapsed=50, limit=100 → 50+32.5=82.5 < 100 → False
        self.assertFalse(_time_budget_exhausted(50.0, [10.0, 10.0, 10.0, 100.0], 100.0))

    def test_already_over_limit_stops(self):
        """Even if elapsed already exceeds limit, we still stop."""
        self.assertTrue(_time_budget_exhausted(300.0, [50.0], 200.0))

    def test_very_tight_budget_stops_after_first_batch(self):
        """If the first batch took longer than the full limit, stop before second."""
        # elapsed=10s, mean=60s (first batch), limit=30s → 10+60=70 > 30 → True
        self.assertTrue(_time_budget_exhausted(10.0, [60.0], 30.0))


class TestFmtDuration(unittest.TestCase):
    """_fmt_duration formats seconds as human-readable strings."""

    def test_zero(self):
        self.assertEqual(_fmt_duration(0), "0s")

    def test_seconds_only(self):
        self.assertEqual(_fmt_duration(45), "45s")

    def test_one_minute_exactly(self):
        self.assertEqual(_fmt_duration(60), "1m 00s")

    def test_minutes_and_seconds(self):
        self.assertEqual(_fmt_duration(90), "1m 30s")

    def test_one_hour_exactly(self):
        self.assertEqual(_fmt_duration(3600), "1h 00m 00s")

    def test_hours_minutes_seconds(self):
        self.assertEqual(_fmt_duration(3661), "1h 01m 01s")

    def test_three_hours(self):
        self.assertEqual(_fmt_duration(10800), "3h 00m 00s")

    def test_float_truncated(self):
        # fractional seconds are truncated, not rounded
        self.assertEqual(_fmt_duration(61.9), "1m 01s")


if __name__ == "__main__":
    unittest.main()
