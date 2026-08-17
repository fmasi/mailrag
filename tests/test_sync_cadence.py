"""Per-account cadence: one scheduler tick, accounts due at different rates.

The two accounts want different freshness. iCloud is the live mailbox and needs
4 h at worst; Gmail is archive-only and nearly dormant (34 messages in a month),
so a day is fine. Before this, `cadence` in accounts.yaml was read only when a
SINGLE account was targeted — a two-account install silently fell back to a
hardcoded 43200, and the plist actually running had been hand-edited to 14400,
so the config and reality had drifted apart with nothing to catch it.

The shape chosen: ONE launchd unit ticking at the shortest cadence, with each
account gated on its own cadence inside the tick. One unit per account would run
two processes that each load bge-m3 (~2 GB of GPU) and write the same SQLite
ledger concurrently; one process keeps the memoised single load and single writer.
"""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from src.sync.schedule import DUE_GRACE_FRACTION, is_due, unit_interval_seconds

NOW = datetime(2026, 8, 17, 12, 0, 0, tzinfo=timezone.utc)
FOUR_HOURS = 4 * 3600
ONE_DAY = 24 * 3600


def _ago(hours):
    return (NOW - timedelta(hours=hours)).isoformat()


class TestUnitInterval(unittest.TestCase):
    def test_is_the_shortest_cadence_so_the_keenest_account_is_satisfied(self):
        """Ticking at the longest would starve iCloud; the gate handles the rest."""
        self.assertEqual(unit_interval_seconds([ONE_DAY, FOUR_HOURS]), FOUR_HOURS)

    def test_a_single_account_uses_its_own_cadence(self):
        self.assertEqual(unit_interval_seconds([ONE_DAY]), ONE_DAY)

    def test_order_does_not_matter(self):
        self.assertEqual(unit_interval_seconds([FOUR_HOURS, ONE_DAY]), FOUR_HOURS)

    def test_no_accounts_is_an_error_not_a_silent_default(self):
        """The old code's silent 43200 fallback is exactly what caused the drift."""
        with self.assertRaises(ValueError):
            unit_interval_seconds([])


class TestIsDue(unittest.TestCase):
    def test_an_account_that_has_never_succeeded_is_due(self):
        self.assertTrue(is_due(cadence_seconds=ONE_DAY, last_success_completed_at=None, now=NOW))

    def test_a_long_overdue_account_is_due(self):
        self.assertTrue(
            is_due(cadence_seconds=FOUR_HOURS, last_success_completed_at=_ago(30), now=NOW)
        )

    def test_a_freshly_synced_account_is_skipped(self):
        """Gmail on a 24 h cadence must sit out five of every six 4 h ticks."""
        self.assertFalse(
            is_due(cadence_seconds=ONE_DAY, last_success_completed_at=_ago(2), now=NOW)
        )

    def test_a_tick_arriving_a_touch_early_still_counts_as_due(self):
        """The bug this grace exists to prevent.

        launchd fires a 4 h StartInterval at 4.0–4.3 h, but jitter the other way
        means an account on a 4 h cadence can be measured at 3.98 h elapsed. With
        a strict >= comparison it is skipped, and the NEXT chance is a whole tick
        later — silently turning a 4 h cadence into 8 h.
        """
        self.assertTrue(
            is_due(cadence_seconds=FOUR_HOURS, last_success_completed_at=_ago(3.95), now=NOW)
        )

    def test_but_genuinely_early_is_still_skipped(self):
        self.assertFalse(
            is_due(cadence_seconds=FOUR_HOURS, last_success_completed_at=_ago(3.0), now=NOW)
        )

    def test_the_grace_is_proportional_to_the_cadence(self):
        """A fixed grace would be either useless at 24 h or sloppy at 4 h."""
        just_inside = ONE_DAY * (1 - DUE_GRACE_FRACTION / 2) / 3600
        just_outside = ONE_DAY * (1 - DUE_GRACE_FRACTION * 2) / 3600
        self.assertTrue(
            is_due(cadence_seconds=ONE_DAY, last_success_completed_at=_ago(just_inside), now=NOW)
        )
        self.assertFalse(
            is_due(cadence_seconds=ONE_DAY, last_success_completed_at=_ago(just_outside), now=NOW)
        )

    def test_an_unparseable_timestamp_fails_OPEN(self):
        """Skipping on bad data would mean an account that never syncs again.

        Syncing needlessly costs one idle tick; not syncing costs the index.
        """
        self.assertTrue(
            is_due(cadence_seconds=FOUR_HOURS, last_success_completed_at="not-a-date", now=NOW)
        )

    def test_the_real_pair_behaves_as_configured(self):
        """iCloud 4 h + Gmail 24 h, on a 4 h tick, 6 h after both last succeeded."""
        self.assertTrue(
            is_due(cadence_seconds=FOUR_HOURS, last_success_completed_at=_ago(6), now=NOW)
        )
        self.assertFalse(
            is_due(cadence_seconds=ONE_DAY, last_success_completed_at=_ago(6), now=NOW)
        )


if __name__ == "__main__":
    unittest.main()
