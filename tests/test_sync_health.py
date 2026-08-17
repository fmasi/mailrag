"""The warnings `sync --status` prints, as a pure decision over ledger rows.

Why these live away from the CLI: the health verdict is the only part of
`--status` that DECIDES anything, and while it was inline in a print loop it had
no test at all — the 48 h threshold shipped for months against a 4 h schedule
without anything noticing it could not detect a missed day.

The governing fact about this laptop: **a closed, unplugged lid legitimately
produces no runs.** So an age-based warning cannot mean "broken" — it can only
mean "stale". Anything that IS unambiguously broken (a run that failed, a run
that never finished, a schedule that has never once succeeded) must warn on its
own evidence rather than waiting for a clock.
"""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from src.sync.health import STALE_AFTER_HOURS, health_warnings

NOW = datetime(2026, 8, 17, 12, 0, 0, tzinfo=timezone.utc)


def _run(status="ok", *, completed_hours_ago=1.0, message=""):
    """A sync_runs row as state.status() hands it over."""
    completed = None
    if completed_hours_ago is not None:
        completed = (NOW - timedelta(hours=completed_hours_ago)).isoformat()
    return {
        "status": status,
        "started_at": (NOW - timedelta(hours=(completed_hours_ago or 0) + 0.1)).isoformat(),
        "completed_at": completed,
        "fetched": 0,
        "judged": 0,
        "indexed": 0,
        "errors": 0,
        "message": message,
    }


class TestNothingToWarnAbout(unittest.TestCase):
    def test_an_account_that_has_never_run_is_not_a_fault(self):
        """Freshly installed is not broken; the caller already prints 'never'."""
        self.assertEqual(health_warnings(last_run=None, last_success=None, now=NOW), [])

    def test_a_recent_successful_run_is_silent(self):
        ok = _run("ok", completed_hours_ago=1.0)
        self.assertEqual(health_warnings(last_run=ok, last_success=ok, now=NOW), [])

    def test_a_closed_lid_overnight_is_silent(self):
        """The accepted normal: lid shut and unplugged, so no tick could fire.

        Warning here would train the warning to be ignored, which is worse than
        not having it.
        """
        ok = _run("ok", completed_hours_ago=STALE_AFTER_HOURS - 1)
        self.assertEqual(health_warnings(last_run=ok, last_success=ok, now=NOW), [])

    def test_the_threshold_itself_does_not_warn(self):
        ok = _run("ok", completed_hours_ago=STALE_AFTER_HOURS)
        self.assertEqual(health_warnings(last_run=ok, last_success=ok, now=NOW), [])


class TestUnambiguousFailuresWarnRegardlessOfAge(unittest.TestCase):
    def test_a_failed_newest_run_warns_even_though_it_is_minutes_old(self):
        """The case the old 48 h clock could not see.

        The schedule is firing perfectly and failing every time; measured by age
        alone that looks healthy, because the newest attempt is always recent.
        """
        bad = _run("partial", completed_hours_ago=0.2)
        warnings = health_warnings(last_run=bad, last_success=None, now=NOW)
        self.assertTrue(any("partial" in w for w in warnings), warnings)

    def test_an_errored_newest_run_warns(self):
        bad = _run("error", completed_hours_ago=0.2)
        warnings = health_warnings(last_run=bad, last_success=None, now=NOW)
        self.assertTrue(any("error" in w for w in warnings), warnings)

    def test_a_run_that_never_finished_warns(self):
        """A killed tick (sleep, shutdown, SIGKILL) leaves completed_at NULL."""
        killed = _run("running", completed_hours_ago=None)
        warnings = health_warnings(last_run=killed, last_success=None, now=NOW)
        self.assertTrue(any("never finished" in w for w in warnings), warnings)

    def test_having_never_succeeded_warns_even_if_attempts_are_recent(self):
        bad = _run("partial", completed_hours_ago=0.2)
        warnings = health_warnings(last_run=bad, last_success=None, now=NOW)
        self.assertTrue(any("EVER completed successfully" in w for w in warnings), warnings)


class TestAcceptsWhatTheLedgerActuallyHandsOver(unittest.TestCase):
    """The ledger yields ``sqlite3.Row``, not ``dict``.

    Row supports subscripting but NOT ``.get()``, so a health check written
    against dicts passes every unit test and then raises AttributeError on the
    first real failed run — i.e. it breaks exactly when it is needed. Caught by
    running `--status` against a copy of the real database with a run flipped to
    'partial'; these tests are the regression guard.
    """

    @staticmethod
    def _row(mapping):
        import sqlite3

        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        cols = ", ".join(f'? AS "{k}"' for k in mapping)
        row = conn.execute(f"SELECT {cols}", tuple(mapping.values())).fetchone()
        conn.close()
        return row

    def test_a_healthy_row_is_silent(self):
        ok = self._row(_run("ok", completed_hours_ago=1.0))
        self.assertEqual(health_warnings(last_run=ok, last_success=ok, now=NOW), [])

    def test_a_failed_row_warns_instead_of_raising(self):
        bad = self._row(_run("partial", completed_hours_ago=0.2))
        warnings = health_warnings(last_run=bad, last_success=None, now=NOW)
        self.assertTrue(any("partial" in w for w in warnings), warnings)

    def test_a_stale_row_warns_instead_of_raising(self):
        old = self._row(_run("ok", completed_hours_ago=STALE_AFTER_HOURS + 6))
        warnings = health_warnings(last_run=old, last_success=old, now=NOW)
        self.assertTrue(any("stale" in w.lower() for w in warnings), warnings)


class TestStaleness(unittest.TestCase):
    def test_warns_once_past_the_bound(self):
        old = _run("ok", completed_hours_ago=STALE_AFTER_HOURS + 6)
        warnings = health_warnings(last_run=old, last_success=old, now=NOW)
        self.assertTrue(any("stale" in w.lower() for w in warnings), warnings)

    def test_age_is_measured_from_the_last_SUCCESS_not_the_last_attempt(self):
        """Consecutive failures must not read as fresh.

        The newest attempt being two minutes old says nothing about the index;
        only a successful run advances freshness.
        """
        success = _run("ok", completed_hours_ago=40)
        newest = _run("partial", completed_hours_ago=0.05)
        warnings = health_warnings(last_run=newest, last_success=success, now=NOW)
        self.assertTrue(any("40h" in w for w in warnings), warnings)
        self.assertTrue(any("partial" in w for w in warnings), warnings)

    def test_an_unparseable_timestamp_does_not_crash_or_invent_staleness(self):
        broken = _run("ok")
        broken["completed_at"] = "not-a-date"
        self.assertEqual(health_warnings(last_run=broken, last_success=broken, now=NOW), [])

    def test_a_missing_timestamp_does_not_crash(self):
        broken = _run("ok", completed_hours_ago=None)
        self.assertEqual(health_warnings(last_run=broken, last_success=broken, now=NOW), [])

    def test_the_bound_is_proportionate_to_a_four_hour_schedule(self):
        """Regression guard on the actual number.

        48 h against a 4 h interval meant twelve missed ticks could pass without
        a word; 24 h is longer than any ordinary closed-lid night and shorter
        than a lost day.
        """
        self.assertEqual(STALE_AFTER_HOURS, 24)


if __name__ == "__main__":
    unittest.main()
