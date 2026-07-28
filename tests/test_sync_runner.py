"""Sync runner: fetch -> judge -> index with stage skipping and resumption (#101).

Driven end to end through :class:`MaildirSource` against real files, so these are
genuine integration tests of the orchestration — no fake source, no network.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from email.message import EmailMessage
from unittest import mock

from src.sync.accounts import AccountConfig
from src.sync.factory import build_source
from src.sync.maildir_source import MaildirSource
from src.sync.runner import index_pending, judge_pending, sync_account
from src.sync.sources import FolderRole
from src.sync.state import STATUS_OK, STATUS_PARTIAL, SyncState


def _eml(message_id: str, subject="Hi", body="A body with some words in it.") -> bytes:
    m = EmailMessage()
    m["From"] = "alice@example.com"
    m["To"] = "bob@example.com"
    m["Subject"] = subject
    m["Message-ID"] = message_id
    m["Date"] = "Tue, 15 Jan 2026 09:30:00 +0000"
    m.set_content(body)
    return bytes(m)


class _RunnerTest(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.d, ignore_errors=True)
        self.maildir = os.path.join(self.d, "Maildir")
        for sub in ("cur", "new"):
            os.makedirs(os.path.join(self.maildir, sub))
        self.state = SyncState(os.path.join(self.d, "sync.db"))
        self.addCleanup(self.state.close)
        self.account = AccountConfig(
            id="acct",
            source="maildir",
            path=self.maildir,
            collection="test-collection",
            spool_root=os.path.join(self.d, "incoming"),
        )

    def _deliver(self, name: str, message_id: str, folder=None, mtime=None, **kw):
        base = os.path.join(self.maildir, folder) if folder else self.maildir
        os.makedirs(os.path.join(base, "cur"), exist_ok=True)
        os.makedirs(os.path.join(base, "new"), exist_ok=True)
        path = os.path.join(base, "cur", name)
        with open(path, "wb") as fh:
            fh.write(_eml(message_id, **kw))
        if mtime is not None:
            os.utime(path, (mtime, mtime))
        return path

    def _sync(self, **kw):
        kw.setdefault("fetch_only", True)
        return sync_account(self.account, state=self.state, source_factory=build_source, **kw)


class TestFetchStage(_RunnerTest):
    def test_fetches_new_mail_into_the_spool(self):
        self._deliver("m1", "<1@x>", mtime=1000)
        self._deliver("m2", "<2@x>", mtime=2000)
        report = self._sync()
        self.assertEqual(report.fetched, 2)
        self.assertEqual(self.state.counts("acct")["total"], 2)

    def test_spooled_files_are_real_eml_on_disk(self):
        self._deliver("m1", "<1@x>", mtime=1000)
        self._sync()
        row = self.state.pending("acct", "judged")[0]
        self.assertTrue(os.path.exists(row["eml_path"]))
        self.assertTrue(row["eml_path"].endswith(".eml"))

    def test_a_second_run_fetches_nothing_new(self):
        self._deliver("m1", "<1@x>", mtime=1000)
        self._sync()
        second = self._sync()
        self.assertEqual(second.fetched, 0)
        self.assertEqual(self.state.counts("acct")["total"], 1)

    def test_only_the_delta_is_fetched_on_a_later_run(self):
        self._deliver("m1", "<1@x>", mtime=1000)
        self._sync()
        self._deliver("m2", "<2@x>", mtime=2000)
        report = self._sync()
        self.assertEqual(report.fetched, 1)
        self.assertEqual(self.state.counts("acct")["total"], 2)

    def test_out_of_scope_folders_are_not_fetched(self):
        """The scope decision is expressed in roles, so 'Junk' means junk on any
        provider — including a Maildir++ '.Junk'."""
        self._deliver("m1", "<1@x>", mtime=1000)
        self._deliver("j1", "<j@x>", folder=".Junk", mtime=1000)
        report = self._sync()
        self.assertEqual(report.fetched, 1)

    def test_sent_mail_is_in_scope_by_default(self):
        """A thread without your own replies reads one-sided."""
        self._deliver("s1", "<s@x>", folder=".Sent", mtime=1000)
        self.assertEqual(self._sync().fetched, 1)

    def test_the_same_message_in_two_folders_is_stored_once(self):
        self._deliver("m1", "<same@x>", mtime=1000)
        self._deliver("a1", "<same@x>", folder=".Archive", mtime=1000)
        report = self._sync()
        self.assertEqual(report.fetched, 1)
        self.assertEqual(report.already_had, 1)
        self.assertEqual(self.state.counts("acct")["total"], 1)

    def test_the_cursor_is_persisted_per_folder(self):
        self._deliver("m1", "<1@x>", mtime=1000)
        self._sync()
        cursor, _gen = self.state.get_cursor("acct", "INBOX")
        self.assertIsNotNone(cursor)
        self.assertEqual(cursor.value["name"], "m1")

    def test_a_limit_stops_the_run_early(self):
        for i in range(5):
            self._deliver(f"m{i}", f"<{i}@x>", mtime=1000 + i)
        self.assertEqual(self._sync(limit=2).fetched, 2)

    def test_an_unparseable_message_is_parked_without_blocking_the_folder(self):
        """The msgvault lesson: a poison message must never wedge the watermark."""
        self._deliver("m1", "<1@x>", mtime=1000)
        with open(os.path.join(self.maildir, "cur", "bad"), "wb") as fh:
            fh.write(b"")
        os.utime(os.path.join(self.maildir, "cur", "bad"), (1500, 1500))
        self._deliver("m2", "<2@x>", mtime=2000)
        report = self._sync()
        # The good mail on both sides of the poison message still arrives.
        self.assertEqual(report.fetched, 2)

    def test_a_dead_source_defers_instead_of_failing_the_run(self):
        """No network is the normal state of a laptop, not an exception."""

        def broken(_account):
            raise OSError("network is unreachable")

        report = sync_account(
            self.account, state=self.state, source_factory=broken, fetch_only=True
        )
        self.assertIn("fetch", report.skipped_stages)
        self.assertEqual(report.status, STATUS_PARTIAL)

    def test_the_run_record_is_always_closed(self):
        self._deliver("m1", "<1@x>", mtime=1000)
        self._sync()
        last = self.state.last_run("acct")
        self.assertEqual(last["status"], STATUS_OK)
        self.assertTrue(last["completed_at"])

    def test_a_source_is_always_closed(self):
        self._deliver("m1", "<1@x>", mtime=1000)
        closed = {}
        real = MaildirSource(self.maildir)
        real.close = lambda: closed.setdefault("yes", True)
        sync_account(
            self.account, state=self.state, source_factory=lambda _a: real, fetch_only=True
        )
        self.assertTrue(closed.get("yes"))


class TestGenerationReset(_RunnerTest):
    def test_a_generation_change_re_enumerates_without_re_fetching(self):
        """A UIDVALIDITY bump must cost bandwidth, not LLM calls or duplicate rows."""
        self._deliver("m1", "<1@x>", mtime=1000)
        self._sync()

        source = MaildirSource(self.maildir)
        with mock.patch.object(
            MaildirSource,
            "open_folder",
            lambda self, folder: folder.__class__(folder.name, folder.role, generation="CHANGED"),
        ):
            report = sync_account(
                self.account, state=self.state, source_factory=lambda _a: source, fetch_only=True
            )
        self.assertEqual(report.folders_reset, 1)
        self.assertEqual(report.already_had, 1)  # recognised, not re-ingested
        self.assertEqual(report.fetched, 0)
        self.assertEqual(self.state.counts("acct")["total"], 1)


class TestJudgeStage(_RunnerTest):
    def _profile(self):
        return mock.Mock(pass2_cache=os.path.join(self.d, "p2.db"), rubric="personal")

    def test_judges_only_what_is_pending(self):
        self._deliver("m1", "<1@x>", mtime=1000)
        self._sync()
        seen = {}

        def fake_run_pass(*, profile, paths, model, workers):
            seen["paths"] = paths
            return {"cached": 0, "done": len(paths), "error": 0}

        report = judge_pending(
            self.account,
            self.state,
            profile=self._profile(),
            model="m",
            run_pass_fn=fake_run_pass,
        )
        self.assertEqual(report.judged, 1)
        self.assertEqual(len(seen["paths"]), 1)
        self.assertEqual(self.state.counts("acct")["pending_judge"], 0)

    def test_a_dead_llm_defers_the_stage_and_keeps_the_mail(self):
        """The expensive work is cache-protected and the mail is already on disk,
        so an unreachable endpoint costs nothing but a delay."""
        self._deliver("m1", "<1@x>", mtime=1000)
        self._sync()

        def broken(**_kw):
            raise ConnectionError("connection refused")

        report = judge_pending(
            self.account, self.state, profile=self._profile(), model="m", run_pass_fn=broken
        )
        self.assertIn("judge", report.skipped_stages)
        self.assertEqual(self.state.counts("acct")["pending_judge"], 1)

    def test_nothing_pending_is_a_no_op(self):
        report = judge_pending(
            self.account,
            self.state,
            profile=self._profile(),
            model="m",
            run_pass_fn=lambda **kw: self.fail("should not have been called"),
        )
        self.assertEqual(report.judged, 0)

    def test_errors_stay_pending_for_the_next_run(self):
        for i in range(3):
            self._deliver(f"m{i}", f"<{i}@x>", mtime=1000 + i)
        self._sync()
        report = judge_pending(
            self.account,
            self.state,
            profile=self._profile(),
            model="m",
            run_pass_fn=lambda **kw: {"cached": 0, "done": 2, "error": 1},
        )
        self.assertEqual(report.judged, 2)
        self.assertEqual(self.state.counts("acct")["pending_judge"], 1)


class TestIndexStage(_RunnerTest):
    def _profile(self):
        return mock.Mock(pass2_cache=None, chunk_size=512, chunk_overlap=64, qdrant_url="http://x")

    def test_indexes_pending_mail_into_the_accounts_collection(self):
        self._deliver("m1", "<1@x>", mtime=1000)
        self._sync()
        seen = {}

        def fake_index(*, profile, embedder, paths, collection):
            seen.update(paths=paths, collection=collection)
            return 3

        report = index_pending(
            self.account,
            self.state,
            profile=self._profile(),
            embedder=mock.Mock(),
            index_fn=fake_index,
        )
        self.assertEqual(report.indexed, 1)
        self.assertEqual(seen["collection"], "test-collection")
        self.assertEqual(self.state.counts("acct")["pending_index"], 0)

    def test_a_dead_qdrant_defers_indexing_but_keeps_the_judgments(self):
        """Docker being stopped is routine; the LLM work already done must not be
        thrown away."""
        self._deliver("m1", "<1@x>", mtime=1000)
        self._sync()
        self.state.mark_judged(
            "acct", [r["message_key"] for r in self.state.pending("acct", "judged")]
        )

        def broken(**_kw):
            raise ConnectionError("cannot connect to Qdrant")

        report = index_pending(
            self.account,
            self.state,
            profile=self._profile(),
            embedder=mock.Mock(),
            index_fn=broken,
        )
        self.assertIn("index", report.skipped_stages)
        self.assertEqual(self.state.counts("acct")["pending_index"], 1)
        self.assertEqual(self.state.counts("acct")["pending_judge"], 0)

    def test_a_later_run_picks_up_what_was_deferred(self):
        self._deliver("m1", "<1@x>", mtime=1000)
        self._sync()
        index_pending(
            self.account,
            self.state,
            profile=self._profile(),
            embedder=mock.Mock(),
            index_fn=lambda **kw: (_ for _ in ()).throw(ConnectionError("down")),
        )
        report = index_pending(
            self.account,
            self.state,
            profile=self._profile(),
            embedder=mock.Mock(),
            index_fn=lambda **kw: 3,
        )
        self.assertEqual(report.indexed, 1)
        self.assertEqual(self.state.counts("acct")["pending_index"], 0)


class TestMultiAccount(_RunnerTest):
    def test_two_accounts_keep_separate_ledgers_and_cursors(self):
        other_dir = os.path.join(self.d, "Other")
        for sub in ("cur", "new"):
            os.makedirs(os.path.join(other_dir, sub))
        with open(os.path.join(other_dir, "cur", "o1"), "wb") as fh:
            fh.write(_eml("<o@x>"))
        other = AccountConfig(
            id="other",
            source="maildir",
            path=other_dir,
            collection="other-collection",
            spool_root=os.path.join(self.d, "incoming-other"),
        )
        self._deliver("m1", "<1@x>", mtime=1000)

        self._sync()
        sync_account(other, state=self.state, source_factory=build_source, fetch_only=True)

        self.assertEqual(self.state.counts("acct")["total"], 1)
        self.assertEqual(self.state.counts("other")["total"], 1)
        self.assertIsNotNone(self.state.get_cursor("other", "INBOX")[0])

    def test_the_same_message_in_two_accounts_is_kept_per_account(self):
        """Cross-account dedup is deliberately NOT automatic — the same message
        arriving at two of your addresses is meaningful provenance."""
        other_dir = os.path.join(self.d, "Other")
        for sub in ("cur", "new"):
            os.makedirs(os.path.join(other_dir, sub))
        with open(os.path.join(other_dir, "cur", "o1"), "wb") as fh:
            fh.write(_eml("<same@x>"))
        self._deliver("m1", "<same@x>", mtime=1000)
        other = AccountConfig(
            id="other",
            source="maildir",
            path=other_dir,
            collection="other-collection",
            spool_root=os.path.join(self.d, "incoming-other"),
        )
        self._sync()
        sync_account(other, state=self.state, source_factory=build_source, fetch_only=True)
        self.assertEqual(self.state.counts("acct")["total"], 1)
        self.assertEqual(self.state.counts("other")["total"], 1)


class TestSourceFactory(unittest.TestCase):
    def test_builds_a_maildir_source(self):
        src = build_source(AccountConfig(id="a", source="maildir", path="/tmp/x"))
        self.assertEqual(src.name, "maildir")

    def test_maildir_without_a_path_is_rejected(self):
        with self.assertRaises(ValueError):
            build_source(AccountConfig(id="a", source="maildir"))

    def test_imap_resolves_its_secret_at_build_time(self):
        with mock.patch("src.sync.factory.resolve_secret", return_value="pw") as resolve:
            src = build_source(
                AccountConfig(
                    id="a",
                    source="imap",
                    host="imap.example.com",
                    login="u",
                    secret="keychain:svc",
                )
            )
        self.assertEqual(src.name, "imap")
        resolve.assert_called_once_with("keychain:svc")

    def test_imap_missing_fields_are_named_in_the_error(self):
        with self.assertRaises(ValueError) as ctx:
            build_source(AccountConfig(id="a", source="imap"))
        self.assertIn("host", str(ctx.exception))

    def test_an_unknown_source_is_rejected(self):
        with self.assertRaises(ValueError):
            build_source(AccountConfig(id="a", source="carrier-pigeon"))


class TestScopeRoles(unittest.TestCase):
    def test_the_default_scope_matches_the_documented_decision(self):
        cfg = AccountConfig(id="a")
        included = [r for r in FolderRole if cfg.wants(r)]
        self.assertEqual(
            set(included),
            {FolderRole.INBOX, FolderRole.SENT, FolderRole.ARCHIVE, FolderRole.OTHER},
        )


if __name__ == "__main__":
    unittest.main()
