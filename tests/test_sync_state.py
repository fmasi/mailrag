"""Sync state store: cursors, the message ledger, and run records (issue #101)."""

from __future__ import annotations

import os
import shutil
import sqlite3
import tempfile
import unittest

from src.sync.sources import CURSOR_UID, Cursor
from src.sync.state import STATUS_FAILED, STATUS_OK, STATUS_RUNNING, SyncState


class _StateTest(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.d, ignore_errors=True)
        self.state = SyncState(os.path.join(self.d, "sync.db"))
        self.addCleanup(self.state.close)


class TestCursors(_StateTest):
    def test_an_unseen_folder_has_no_cursor(self):
        self.assertEqual(self.state.get_cursor("acct", "INBOX"), (None, ""))

    def test_a_cursor_round_trips(self):
        c = Cursor(CURSOR_UID, {"last_uid": 42})
        self.state.set_cursor("acct", "INBOX", c, generation="1234")
        got, gen = self.state.get_cursor("acct", "INBOX")
        self.assertEqual(got, c)
        self.assertEqual(gen, "1234")

    def test_the_cursor_value_is_opaque_to_the_store(self):
        """A Gmail historyId or a JMAP state string must persist untouched — the
        store must never assume IMAP's shape."""
        c = Cursor("jmap_state", {"state": "abc-123", "nested": {"a": [1, 2]}})
        self.state.set_cursor("acct", "INBOX", c)
        self.assertEqual(self.state.get_cursor("acct", "INBOX")[0], c)

    def test_setting_a_cursor_twice_updates_rather_than_duplicating(self):
        self.state.set_cursor("acct", "INBOX", Cursor(CURSOR_UID, {"last_uid": 1}))
        self.state.set_cursor("acct", "INBOX", Cursor(CURSOR_UID, {"last_uid": 9}))
        self.assertEqual(self.state.get_cursor("acct", "INBOX")[0].value["last_uid"], 9)
        self.assertEqual(len(self.state.folders("acct")), 1)

    def test_cursors_are_per_account_and_per_folder(self):
        self.state.set_cursor("a1", "INBOX", Cursor(CURSOR_UID, {"last_uid": 1}))
        self.state.set_cursor("a2", "INBOX", Cursor(CURSOR_UID, {"last_uid": 2}))
        self.state.set_cursor("a1", "Sent", Cursor(CURSOR_UID, {"last_uid": 3}))
        self.assertEqual(self.state.get_cursor("a1", "INBOX")[0].value["last_uid"], 1)
        self.assertEqual(self.state.get_cursor("a2", "INBOX")[0].value["last_uid"], 2)
        self.assertEqual(self.state.get_cursor("a1", "Sent")[0].value["last_uid"], 3)

    def test_last_sync_at_is_stamped(self):
        self.state.set_cursor("acct", "INBOX", Cursor(CURSOR_UID, {"last_uid": 1}))
        self.assertTrue(self.state.folders("acct")[0]["last_sync_at"])


class TestGenerationReset(_StateTest):
    def test_reset_voids_the_cursor_but_records_the_new_generation(self):
        self.state.set_cursor("acct", "INBOX", Cursor(CURSOR_UID, {"last_uid": 42}), generation="1")
        self.state.reset_folder("acct", "INBOX", "2")
        cursor, gen = self.state.get_cursor("acct", "INBOX")
        self.assertIsNone(cursor)
        self.assertEqual(gen, "2")

    def test_reset_keeps_the_message_ledger(self):
        """The point of the content-keyed ledger: a UIDVALIDITY reset costs
        bandwidth, not LLM calls — re-enumerated mail is recognised, not re-judged."""
        self.state.record_fetched("acct", message_key="h1", folder="INBOX")
        self.state.reset_folder("acct", "INBOX", "2")
        self.assertTrue(self.state.have_message("acct", "h1"))

    def test_reset_works_on_a_folder_never_seen_before(self):
        self.state.reset_folder("acct", "NewFolder", "7")
        self.assertEqual(self.state.get_cursor("acct", "NewFolder"), (None, "7"))


class TestMessageLedger(_StateTest):
    def test_records_and_recognises_a_message(self):
        self.assertFalse(self.state.have_message("acct", "h1"))
        self.state.record_fetched("acct", message_key="h1", folder="INBOX", source_uid="10")
        self.assertTrue(self.state.have_message("acct", "h1"))

    def test_the_same_content_in_two_folders_is_one_row(self):
        """A message filed in Archive as well as Inbox must not be spooled twice."""
        self.state.record_fetched("acct", message_key="h1", folder="INBOX", source_uid="10")
        self.state.record_fetched("acct", message_key="h1", folder="Archive", source_uid="77")
        self.assertEqual(self.state.counts("acct")["total"], 1)

    def test_re_sighting_retargets_the_location(self):
        self.state.record_fetched("acct", message_key="h1", folder="INBOX", source_uid="10")
        self.state.record_fetched("acct", message_key="h1", folder="Archive", source_uid="77")
        row = self.state.pending("acct", "judged")[0]
        self.assertEqual((row["folder"], row["source_uid"]), ("Archive", "77"))

    def test_re_sighting_does_not_undo_completed_stages(self):
        """Otherwise a moved message would be re-judged and re-embedded for free."""
        self.state.record_fetched("acct", message_key="h1", folder="INBOX")
        self.state.mark_judged("acct", ["h1"])
        self.state.mark_indexed("acct", ["h1"])
        self.state.record_fetched("acct", message_key="h1", folder="Archive")
        self.assertEqual(self.state.counts("acct")["pending_judge"], 0)
        self.assertEqual(self.state.counts("acct")["pending_index"], 0)

    def test_ledgers_are_isolated_per_account(self):
        self.state.record_fetched("a1", message_key="h1")
        self.assertFalse(self.state.have_message("a2", "h1"))


class TestStageTracking(_StateTest):
    def setUp(self):
        super().setUp()
        for h in ("h1", "h2", "h3"):
            self.state.record_fetched("acct", message_key=h, folder="INBOX")

    def test_everything_starts_pending_judge(self):
        """With a judge stage configured, NEW mail owes judging first — indexing
        it now would embed it without its summary, permanently."""
        self.assertEqual(len(self.state.pending("acct", "judged")), 3)
        self.assertEqual(len(self.state.pending("acct", "indexed")), 0)
        # With no judge stage there is nothing to wait for.
        self.assertEqual(len(self.state.pending("acct", "indexed", judge_configured=False)), 3)

    def test_marking_judged_moves_a_message_to_the_index_queue(self):
        """Stage-skipping: mail judged while Qdrant was down must still get indexed."""
        self.state.mark_judged("acct", ["h1", "h2"])
        self.assertEqual(len(self.state.pending("acct", "judged")), 1)
        self.assertEqual(len(self.state.pending("acct", "indexed")), 2)

    def test_marking_is_idempotent_and_returns_a_count(self):
        self.assertEqual(self.state.mark_judged("acct", ["h1", "h1"]), 1)
        self.assertEqual(self.state.mark_judged("acct", ["h1"]), 1)
        self.assertEqual(len(self.state.pending("acct", "judged")), 2)

    def test_marking_nothing_is_a_no_op(self):
        self.assertEqual(self.state.mark_indexed("acct", []), 0)

    def test_counts_summarise_the_backlog(self):
        self.state.mark_judged("acct", ["h1"])
        self.state.mark_indexed("acct", ["h1"])
        self.state.record_error("acct", "h2", "boom")
        counts = self.state.counts("acct")
        self.assertEqual(counts["total"], 3)
        self.assertEqual(counts["pending_judge"], 2)
        self.assertEqual(counts["pending_index"], 0)
        self.assertEqual(counts["done"], 1)
        self.assertEqual(counts["errors"], 1)

    def test_counts_on_an_unknown_account_are_zero_not_none(self):
        self.assertEqual(self.state.counts("nobody")["total"], 0)
        self.assertEqual(self.state.counts("nobody")["pending_judge"], 0)

    def test_a_parked_error_still_leaves_the_message_in_the_ledger(self):
        """Parking a poison message must not block the folder — the cursor moves on."""
        self.state.record_error("acct", "h1", "unparseable MIME")
        self.assertTrue(self.state.have_message("acct", "h1"))


class TestRuns(_StateTest):
    def test_a_run_starts_running_and_finishes_ok(self):
        run_id = self.state.start_run("acct")
        self.assertEqual(self.state.last_run("acct")["status"], STATUS_RUNNING)
        self.state.finish_run(run_id, status=STATUS_OK, fetched=5, judged=5, indexed=5)
        last = self.state.last_run("acct")
        self.assertEqual(last["status"], STATUS_OK)
        self.assertEqual((last["fetched"], last["judged"], last["indexed"]), (5, 5, 5))
        self.assertTrue(last["completed_at"])

    def test_starting_a_run_supersedes_a_crashed_one(self):
        """A run killed by sleep or SIGKILL leaves a 'running' row forever; the next
        tick must not be blocked by that ghost."""
        ghost = self.state.start_run("acct")
        self.state.start_run("acct")
        rows = {r["id"]: r for r in self.state.recent_runs("acct")}
        self.assertEqual(rows[ghost]["status"], STATUS_FAILED)
        self.assertIn("superseded", rows[ghost]["message"])

    def test_superseding_does_not_touch_other_accounts(self):
        other = self.state.start_run("other")
        self.state.start_run("acct")
        self.assertEqual(self.state.last_run("other")["status"], STATUS_RUNNING)
        self.assertEqual(self.state.last_run("other")["id"], other)

    def test_a_finished_run_is_not_superseded(self):
        done = self.state.start_run("acct")
        self.state.finish_run(done, status=STATUS_OK)
        self.state.start_run("acct")
        rows = {r["id"]: r for r in self.state.recent_runs("acct")}
        self.assertEqual(rows[done]["status"], STATUS_OK)

    def test_last_run_is_none_before_any_run(self):
        self.assertIsNone(self.state.last_run("acct"))

    def test_recent_runs_are_newest_first_and_limited(self):
        ids = [self.state.start_run("acct") for _ in range(5)]
        recent = self.state.recent_runs("acct", limit=3)
        self.assertEqual([r["id"] for r in recent], list(reversed(ids))[:3])

    def test_a_long_error_message_is_truncated_rather_than_rejected(self):
        run_id = self.state.start_run("acct")
        self.state.finish_run(run_id, status=STATUS_FAILED, message="x" * 5000)
        self.assertLessEqual(len(self.state.last_run("acct")["message"]), 2000)


class TestStatusReadout(_StateTest):
    def test_status_gathers_counts_folders_and_the_last_run(self):
        self.state.set_cursor("acct", "INBOX", Cursor(CURSOR_UID, {"last_uid": 3}), role="inbox")
        self.state.record_fetched("acct", message_key="h1")
        run_id = self.state.start_run("acct")
        self.state.finish_run(run_id, status=STATUS_OK, fetched=1)
        st = self.state.status("acct")
        self.assertEqual(st["counts"]["total"], 1)
        self.assertEqual(st["folders"][0]["name"], "INBOX")
        self.assertEqual(st["folders"][0]["role"], "inbox")
        self.assertEqual(st["last_run"]["status"], STATUS_OK)

    def test_status_of_a_fresh_account_is_empty_but_well_formed(self):
        st = self.state.status("nobody")
        self.assertEqual(st["folders"], [])
        self.assertIsNone(st["last_run"])


class TestPersistence(_StateTest):
    def test_state_survives_reopening(self):
        path = os.path.join(self.d, "reopen.db")
        with SyncState(path) as s:
            s.set_cursor("acct", "INBOX", Cursor(CURSOR_UID, {"last_uid": 7}), generation="9")
            s.record_fetched("acct", message_key="h1")
        with SyncState(path) as s:
            self.assertEqual(s.get_cursor("acct", "INBOX")[0].value["last_uid"], 7)
            self.assertTrue(s.have_message("acct", "h1"))

    def test_opening_an_existing_db_twice_does_not_wipe_it(self):
        path = os.path.join(self.d, "twice.db")
        SyncState(path).close()
        s = SyncState(path)
        self.addCleanup(s.close)
        s.record_fetched("acct", message_key="h1")
        SyncState(path).close()
        self.assertTrue(s.have_message("acct", "h1"))

    def test_rows_are_addressable_by_name(self):
        self.state.record_fetched("acct", message_key="k1", content_sha256="deadbeef")
        row = self.state.pending("acct", "judged")[0]
        self.assertIsInstance(row, sqlite3.Row)
        self.assertEqual(row["message_key"], "k1")
        self.assertEqual(row["content_sha256"], "deadbeef")


if __name__ == "__main__":
    unittest.main()


class TestLedgerKeying(_StateTest):
    """The ledger keys on message_key, NOT content_sha256 — see the module docstring."""

    def test_two_distinct_emails_with_identical_content_are_two_rows(self):
        """content_sha256 deliberately excludes the Message-ID so a re-export still
        hits the Pass-2 cache. Keying the ledger there would collapse a newsletter
        sent twice into one row, and the second copy would never be judged."""
        self.state.record_fetched("acct", message_key="a@x", content_sha256="same")
        self.state.record_fetched("acct", message_key="b@x", content_sha256="same")
        self.assertEqual(self.state.counts("acct")["total"], 2)
        self.assertEqual(self.state.counts("acct")["pending_judge"], 2)

    def test_the_content_hash_is_still_recorded(self):
        self.state.record_fetched("acct", message_key="a@x", content_sha256="abc")
        self.assertEqual(self.state.pending("acct", "judged")[0]["content_sha256"], "abc")
