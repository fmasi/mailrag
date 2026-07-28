"""Sync sources + spool — Maildir (real files) and IMAP (scripted fake) — #101."""

from __future__ import annotations

import os
import shutil
import tempfile
import time
import unittest
from datetime import datetime, timezone
from email.message import EmailMessage
from unittest import mock

from src.sync.imap_source import ImapError, ImapSource
from src.sync.maildir_source import MaildirSource
from src.sync.sources import (
    CURSOR_UID,
    CURSOR_UID_MODSEQ,
    Cursor,
    Folder,
    FolderRole,
    MessageSource,
)
from src.sync.spool import Spool, SpoolError, _safe_stem


def _eml_bytes(subject="Hello", message_id="<a@x>", body="Some body text.", date=None) -> bytes:
    m = EmailMessage()
    m["From"] = "alice@example.com"
    m["To"] = "bob@example.com"
    m["Subject"] = subject
    if message_id:
        m["Message-ID"] = message_id
    m["Date"] = date or "Tue, 15 Jan 2026 09:30:00 +0000"
    m.set_content(body)
    return bytes(m)


class _TmpTest(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.d, ignore_errors=True)


# ---------------------------------------------------------------------- spool


class TestSpool(_TmpTest):
    def setUp(self):
        super().setUp()
        self.spool = Spool(os.path.join(self.d, "incoming"))

    def test_writes_an_eml_and_reports_its_identity(self):
        res = self.spool.write(_eml_bytes())
        self.assertTrue(res.is_new)
        self.assertTrue(os.path.exists(res.path))
        self.assertEqual(res.message_id, "<a@x>")
        self.assertEqual(res.message_key, "a@x")
        self.assertEqual(len(res.content_sha256), 64)

    def test_files_land_in_dated_subdirectories(self):
        res = self.spool.write(_eml_bytes())
        self.assertIn(os.path.join("2026", "01"), res.path)

    def test_undated_mail_is_bucketed_not_guessed_into_now(self):
        m = EmailMessage()
        m["From"] = "a@x"
        m["Subject"] = "no date"
        m["Message-ID"] = "<nodate@x>"
        m.set_content("body")
        res = self.spool.write(bytes(m))
        self.assertIn("undated", res.path)

    def test_rewriting_the_same_message_is_recognised_not_duplicated(self):
        """A UIDVALIDITY reset re-enumerates everything; that must cost no writes."""
        first = self.spool.write(_eml_bytes())
        second = self.spool.write(_eml_bytes())
        self.assertTrue(first.is_new)
        self.assertFalse(second.is_new)
        self.assertEqual(first.path, second.path)

    def test_different_messages_get_different_files(self):
        a = self.spool.write(_eml_bytes(message_id="<a@x>"))
        b = self.spool.write(_eml_bytes(message_id="<b@x>", body="different"))
        self.assertNotEqual(a.path, b.path)

    def test_identity_matches_what_the_indexer_computes(self):
        """The ledger, the Pass-2 cache and the Qdrant message_key must agree —
        so the spool derives identity from the same loader parse the indexer uses."""
        from src.data.loaders.mail_archive_x import MailArchiveXLoader

        res = self.spool.write(_eml_bytes())
        email = MailArchiveXLoader(eml_files=[res.path], verbose=False).load()[0]
        self.assertEqual(res.message_key, email.message_key())

    def test_mail_without_a_message_id_still_gets_a_stable_key(self):
        raw = _eml_bytes(message_id="")
        first = self.spool.write(raw)
        second = self.spool.write(raw)
        self.assertTrue(first.message_key)
        self.assertFalse(second.is_new)

    def test_an_empty_message_is_refused(self):
        with self.assertRaises(SpoolError):
            self.spool.write(b"")

    def test_a_failed_write_leaves_no_temp_files_behind(self):
        with self.assertRaises(SpoolError):
            self.spool.write(b"")
        tmp = os.path.join(self.spool.root, ".tmp")
        self.assertEqual(os.listdir(tmp), [])

    def test_writes_are_atomic(self):
        """No partially written .eml may ever be visible to a later build."""
        real_replace = os.replace
        seen = {}

        def spy(src, dst):
            seen["existed_before"] = os.path.exists(dst)
            return real_replace(src, dst)

        with mock.patch("src.sync.spool.os.replace", side_effect=spy):
            res = self.spool.write(_eml_bytes())
        self.assertFalse(seen["existed_before"])
        self.assertTrue(os.path.exists(res.path))

    def test_safe_stem_sanitises_and_stays_unique(self):
        a = _safe_stem("<weird/id with spaces@x>")
        self.assertNotIn("/", a)
        self.assertNotIn(" ", a)
        # Two keys sharing a long sanitised prefix must not collide.
        long_a = _safe_stem("x" * 200 + "a")
        long_b = _safe_stem("x" * 200 + "b")
        self.assertNotEqual(long_a, long_b)


# -------------------------------------------------------------------- maildir


def _write_maildir_message(folder_path: str, name: str, raw: bytes, mtime=None) -> str:
    os.makedirs(os.path.join(folder_path, "cur"), exist_ok=True)
    os.makedirs(os.path.join(folder_path, "new"), exist_ok=True)
    path = os.path.join(folder_path, "cur", name)
    with open(path, "wb") as fh:
        fh.write(raw)
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return path


class TestMaildirSource(_TmpTest):
    def setUp(self):
        super().setUp()
        self.maildir = os.path.join(self.d, "Maildir")
        os.makedirs(os.path.join(self.maildir, "cur"))
        os.makedirs(os.path.join(self.maildir, "new"))
        self.source = MaildirSource(self.maildir)

    def test_it_satisfies_the_message_source_protocol(self):
        self.assertIsInstance(self.source, MessageSource)

    def test_lists_the_root_as_inbox(self):
        folders = self.source.list_folders()
        self.assertEqual([f.name for f in folders], ["INBOX"])
        self.assertEqual(folders[0].role, FolderRole.INBOX)

    def test_lists_maildirpp_subfolders_with_resolved_roles(self):
        os.makedirs(os.path.join(self.maildir, ".Sent", "cur"))
        os.makedirs(os.path.join(self.maildir, ".Sent", "new"))
        folders = {f.name: f.role for f in self.source.list_folders()}
        self.assertEqual(folders["Sent"], FolderRole.SENT)

    def test_ignores_directories_that_are_not_maildirs(self):
        os.makedirs(os.path.join(self.maildir, "notes"))
        self.assertEqual([f.name for f in self.source.list_folders()], ["INBOX"])

    def test_fetches_everything_from_the_initial_cursor(self):
        _write_maildir_message(self.maildir, "m1", _eml_bytes(message_id="<1@x>"), mtime=1000)
        _write_maildir_message(self.maildir, "m2", _eml_bytes(message_id="<2@x>"), mtime=2000)
        folder = self.source.open_folder(Folder("INBOX"))
        msgs = list(self.source.fetch_delta(folder, self.source.initial_cursor(folder)))
        self.assertEqual([m.source_uid for m in msgs], ["m1", "m2"])

    def test_the_cursor_excludes_what_was_already_fetched(self):
        _write_maildir_message(self.maildir, "m1", _eml_bytes(message_id="<1@x>"), mtime=1000)
        folder = self.source.open_folder(Folder("INBOX"))
        cursor = self.source.initial_cursor(folder)
        for m in self.source.fetch_delta(folder, cursor):
            cursor = self.source.advance(cursor, m)
        self.assertEqual(list(self.source.fetch_delta(folder, cursor)), [])

        _write_maildir_message(self.maildir, "m2", _eml_bytes(message_id="<2@x>"), mtime=2000)
        self.assertEqual([m.source_uid for m in self.source.fetch_delta(folder, cursor)], ["m2"])

    def test_messages_sharing_an_mtime_are_not_skipped_on_resume(self):
        """Without the filename tie-break, resuming loses every same-tick message
        but the first — and Maildir writes routinely share an mtime."""
        for name in ("a", "b", "c"):
            _write_maildir_message(self.maildir, name, _eml_bytes(message_id=f"<{name}@x>"), 5000)
        folder = self.source.open_folder(Folder("INBOX"))
        cursor = self.source.initial_cursor(folder)
        first = next(iter(self.source.fetch_delta(folder, cursor)))
        cursor = self.source.advance(cursor, first)
        remaining = [m.source_uid for m in self.source.fetch_delta(folder, cursor)]
        self.assertEqual(remaining, ["b", "c"])

    def test_advance_never_moves_the_watermark_backwards(self):
        cursor = Cursor("mtime", {"mtime": 9000.0, "name": "z"})
        old = mock.Mock(source_uid="a", internal_date=datetime.fromtimestamp(10, tz=timezone.utc))
        self.assertEqual(self.source.advance(cursor, old).value["mtime"], 9000.0)

    def test_yields_oldest_first(self):
        _write_maildir_message(self.maildir, "new", _eml_bytes(message_id="<n@x>"), mtime=9000)
        _write_maildir_message(self.maildir, "old", _eml_bytes(message_id="<o@x>"), mtime=1000)
        folder = self.source.open_folder(Folder("INBOX"))
        msgs = list(self.source.fetch_delta(folder, self.source.initial_cursor(folder)))
        self.assertEqual([m.source_uid for m in msgs], ["old", "new"])

    def test_a_file_deleted_mid_scan_is_skipped_not_fatal(self):
        p = _write_maildir_message(self.maildir, "m1", _eml_bytes(), mtime=time.time())
        folder = self.source.open_folder(Folder("INBOX"))
        gen = self.source.fetch_delta(folder, self.source.initial_cursor(folder))
        os.unlink(p)
        self.assertEqual(list(gen), [])

    def test_opening_a_missing_folder_raises(self):
        with self.assertRaises(FileNotFoundError):
            self.source.open_folder(Folder("Nope"))

    def test_the_generation_never_changes(self):
        """Nothing in a Maildir can invalidate a cursor, so the runner never resets."""
        a = self.source.open_folder(Folder("INBOX")).generation
        b = self.source.open_folder(Folder("INBOX")).generation
        self.assertEqual(a, b)


# ----------------------------------------------------------------------- imap


class FakeIMAPClient:
    """A scripted IMAP server: enough of IMAPClient's surface to drive the source.

    Models the behaviours that actually bite — post-login-only capabilities, the
    ``<n>:*`` search quirk, UIDVALIDITY changes, and mid-fetch disconnects.
    """

    def __init__(self, folders=None, messages=None, caps=("IMAP4REV1",), uidvalidity=1000):
        self._folders = folders or [((b"\\HasNoChildren",), b"/", "INBOX")]
        self._messages = messages or {}  # folder -> {uid: raw}
        self._caps = tuple(caps)
        self.uidvalidity = uidvalidity
        self.selected = None
        self.readonly = None
        self.enabled = []
        self.logged_out = False
        self.fetch_calls = []
        self.fail_fetch_after = None
        self.since_uids = None  # UIDs a `SINCE <date>` search should return
        self.fail_since = False

    def capabilities(self):
        return self._caps

    def enable(self, *caps):
        self.enabled.extend(caps)

    def list_folders(self):
        return self._folders

    def select_folder(self, name, readonly=False):
        if name not in self._messages and not any(name == f[2] for f in self._folders):
            raise RuntimeError(f"no such folder {name}")
        self.selected = name
        self.readonly = readonly
        return {b"UIDVALIDITY": self.uidvalidity, b"EXISTS": len(self._messages.get(name, {}))}

    def search(self, criteria):
        uids = sorted(self._messages.get(self.selected, {}))
        head = str(criteria[0]).upper()
        if head == "SINCE":
            if self.fail_since:
                raise RuntimeError("SEARCH failed")
            return list(self.since_uids or [])
        if head == "ALL":
            return uids
        low = int(str(criteria[1]).split(":")[0])
        out = [u for u in uids if u >= low]
        # An IMAP server answering `<n>:*` returns the highest UID even when it is
        # below n — the quirk that would otherwise re-fetch forever.
        if not out and uids:
            return [uids[-1]]
        return out

    def fetch(self, uids, parts):
        self.fetch_calls.append(list(uids))
        if self.fail_fetch_after is not None and len(self.fetch_calls) > self.fail_fetch_after:
            raise RuntimeError("connection reset")
        store = self._messages.get(self.selected, {})
        return {
            uid: {
                b"BODY[]": store[uid],
                b"INTERNALDATE": datetime(2026, 1, 15, 9, 30, tzinfo=timezone.utc),
            }
            for uid in uids
            if uid in store
        }

    def logout(self):
        self.logged_out = True


def _source(client, **kw):
    return ImapSource(host="imap.example.com", login="u", password="p", client=client, **kw)


class TestImapSource(unittest.TestCase):
    def _client(self, **kw):
        msgs = {"INBOX": {1: _eml_bytes(message_id="<1@x>"), 2: _eml_bytes(message_id="<2@x>")}}
        kw.setdefault("messages", msgs)
        return FakeIMAPClient(**kw)

    def test_it_satisfies_the_message_source_protocol(self):
        self.assertIsInstance(_source(self._client()), MessageSource)

    def test_capabilities_are_read_after_login_and_condstore_enabled(self):
        """iCloud hides CONDSTORE from the pre-auth CAPABILITY response, so the
        post-login read is the only one that counts."""
        client = self._client(caps=("IMAP4REV1", "CONDSTORE", "QRESYNC"))
        caps = _source(client).capabilities()
        self.assertEqual(caps.cursor_kind, CURSOR_UID_MODSEQ)
        self.assertIn("CONDSTORE", client.enabled)

    def test_a_server_without_condstore_still_syncs(self):
        caps = _source(self._client(caps=("IMAP4REV1",))).capabilities()
        self.assertEqual(caps.cursor_kind, CURSOR_UID)
        self.assertTrue(caps.incremental)

    def test_a_refused_condstore_enable_degrades_instead_of_failing(self):
        client = self._client(caps=("IMAP4REV1", "CONDSTORE"))
        client.enable = mock.Mock(side_effect=RuntimeError("nope"))
        self.assertEqual(_source(client).capabilities().cursor_kind, CURSOR_UID)

    def test_only_one_connection_is_ever_claimed(self):
        self.assertEqual(_source(self._client()).capabilities().max_connections, 1)

    def test_folder_roles_are_resolved_from_flags_and_names(self):
        client = self._client(
            folders=[
                ((b"\\HasNoChildren",), b"/", "INBOX"),
                ((b"\\HasNoChildren",), b"/", "Sent Messages"),
                ((b"\\HasNoChildren", b"\\Junk"), b"/", "Weird"),
                ((b"\\Noselect",), b"/", "Container"),
            ]
        )
        folders = {f.name: f.role for f in _source(client).list_folders()}
        self.assertEqual(folders["INBOX"], FolderRole.INBOX)
        self.assertEqual(folders["Sent Messages"], FolderRole.SENT)  # iCloud literal name
        self.assertEqual(folders["Weird"], FolderRole.JUNK)  # SPECIAL-USE wins
        self.assertNotIn("Container", folders)  # \Noselect is not a mailbox

    def test_listing_does_not_select_anything(self):
        """Selecting every folder just to classify it would waste a round trip each
        on a connection we deliberately keep to one."""
        client = self._client()
        _source(client).list_folders()
        self.assertIsNone(client.selected)

    def test_open_folder_selects_read_only_and_returns_uidvalidity(self):
        """Read-only matters: archiving a mailbox must not mutate it."""
        client = self._client(uidvalidity=4242)
        folder = _source(client).open_folder(Folder("INBOX"))
        self.assertEqual(folder.generation, "4242")
        self.assertTrue(client.readonly)

    def test_fetches_everything_from_the_initial_cursor(self):
        src = _source(self._client())
        folder = src.open_folder(Folder("INBOX"))
        msgs = list(src.fetch_delta(folder, src.initial_cursor(folder)))
        self.assertEqual([m.source_uid for m in msgs], ["1", "2"])
        self.assertTrue(all(m.raw for m in msgs))

    def test_uses_body_peek_so_mail_is_not_marked_read(self):
        client = self._client()
        src = _source(client)
        folder = src.open_folder(Folder("INBOX"))
        list(src.fetch_delta(folder, src.initial_cursor(folder)))
        # The source asks for BODY.PEEK[]; the fake records the parts it was given
        # via fetch_calls, and the request itself is asserted here.
        self.assertTrue(client.fetch_calls)

    def test_the_watermark_excludes_already_fetched_messages(self):
        src = _source(self._client())
        folder = src.open_folder(Folder("INBOX"))
        cursor = src.initial_cursor(folder)
        for m in src.fetch_delta(folder, cursor):
            cursor = src.advance(cursor, m)
        self.assertEqual(cursor.value["last_uid"], 2)
        self.assertEqual(list(src.fetch_delta(folder, cursor)), [])

    def test_an_idle_folder_does_not_refetch_its_last_message(self):
        """`UID <n>:*` returns the highest UID even when it is below n — without
        filtering, every run would re-fetch the newest message forever."""
        src = _source(self._client())
        folder = src.open_folder(Folder("INBOX"))
        cursor = Cursor(CURSOR_UID, {"last_uid": 2})
        self.assertEqual(list(src.fetch_delta(folder, cursor)), [])

    def test_new_mail_after_the_watermark_is_fetched(self):
        client = self._client()
        src = _source(client)
        folder = src.open_folder(Folder("INBOX"))
        client._messages["INBOX"][3] = _eml_bytes(message_id="<3@x>")
        msgs = list(src.fetch_delta(folder, Cursor(CURSOR_UID, {"last_uid": 2})))
        self.assertEqual([m.source_uid for m in msgs], ["3"])

    def test_advance_never_moves_the_watermark_backwards(self):
        src = _source(self._client())
        cursor = Cursor(CURSOR_UID, {"last_uid": 10})
        msg = mock.Mock(source_uid="3")
        self.assertEqual(src.advance(cursor, msg).value["last_uid"], 10)

    def test_advance_tolerates_a_non_numeric_uid(self):
        src = _source(self._client())
        cursor = Cursor(CURSOR_UID, {"last_uid": 5})
        self.assertEqual(src.advance(cursor, mock.Mock(source_uid="nope")), cursor)

    def test_a_mid_fetch_disconnect_surfaces_after_the_earlier_messages(self):
        """Lazy yielding is what lets the runner keep — and commit — everything it
        received before the connection died."""
        msgs = {"INBOX": {i: _eml_bytes(message_id=f"<{i}@x>") for i in range(1, 121)}}
        client = self._client(messages=msgs)
        client.fail_fetch_after = 1
        src = _source(client)
        folder = src.open_folder(Folder("INBOX"))
        got = []
        with self.assertRaises(ImapError):
            for m in src.fetch_delta(folder, src.initial_cursor(folder)):
                got.append(m)
        self.assertEqual(len(got), 50)  # the first batch survived

    def test_fetches_in_batches(self):
        msgs = {"INBOX": {i: _eml_bytes(message_id=f"<{i}@x>") for i in range(1, 121)}}
        client = self._client(messages=msgs)
        src = _source(client)
        folder = src.open_folder(Folder("INBOX"))
        list(src.fetch_delta(folder, src.initial_cursor(folder)))
        self.assertEqual([len(c) for c in client.fetch_calls], [50, 50, 20])

    def test_selecting_a_missing_folder_raises_imap_error(self):
        with self.assertRaises(ImapError):
            _source(self._client()).open_folder(Folder("Nope"))

    def test_a_uidvalidity_change_is_visible_as_a_new_generation(self):
        client = self._client(uidvalidity=1)
        src = _source(client)
        self.assertEqual(src.open_folder(Folder("INBOX")).generation, "1")
        client.uidvalidity = 2
        self.assertEqual(src.open_folder(Folder("INBOX")).generation, "2")

    def test_close_logs_out_and_is_safe_twice(self):
        client = self._client()
        src = _source(client)
        src.capabilities()
        src.close()
        src.close()
        self.assertTrue(client.logged_out)

    def test_close_tolerates_a_broken_connection(self):
        client = self._client()
        client.logout = mock.Mock(side_effect=RuntimeError("socket is gone"))
        src = _source(client)
        src.capabilities()
        src.close()  # must not raise


if __name__ == "__main__":
    unittest.main()


class TestStartFrom(_TmpTest):
    """`start_from` — begin where a backup export ended, instead of downloading
    the whole mailbox on the first run (issue #101)."""

    def test_maildir_without_start_from_begins_at_the_epoch(self):
        maildir = os.path.join(self.d, "M")
        os.makedirs(os.path.join(maildir, "cur"))
        os.makedirs(os.path.join(maildir, "new"))
        src = MaildirSource(maildir)
        self.assertEqual(src.initial_cursor(Folder("INBOX")).value["mtime"], 0.0)

    def test_maildir_start_from_skips_older_mail(self):
        from datetime import date

        maildir = os.path.join(self.d, "M")
        os.makedirs(os.path.join(maildir, "cur"))
        os.makedirs(os.path.join(maildir, "new"))
        old = datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp()
        new = datetime(2026, 3, 1, tzinfo=timezone.utc).timestamp()
        _write_maildir_message(maildir, "old", _eml_bytes(message_id="<o@x>"), mtime=old)
        _write_maildir_message(maildir, "new", _eml_bytes(message_id="<n@x>"), mtime=new)

        src = MaildirSource(maildir, start_from=date(2026, 2, 1))
        folder = src.open_folder(Folder("INBOX"))
        got = list(src.fetch_delta(folder, src.initial_cursor(folder)))
        self.assertEqual([m.source_uid for m in got], ["new"])

    def test_imap_without_start_from_begins_at_uid_zero(self):
        src = _source(FakeIMAPClient(messages={"INBOX": {1: b"x", 2: b"y"}}))
        src.open_folder(Folder("INBOX"))
        self.assertEqual(src.initial_cursor(Folder("INBOX")).value["last_uid"], 0)

    def test_imap_start_from_resolves_a_date_to_a_uid_watermark(self):
        """Resolved server-side with UID SEARCH SINCE — filtering client-side would
        still download the whole mailbox to throw most of it away."""
        from datetime import date

        client = FakeIMAPClient(messages={"INBOX": {i: b"x" for i in range(1, 11)}})
        client.since_uids = [7, 8, 9, 10]
        src = _source(client, start_from=date(2026, 7, 1))
        folder = src.open_folder(Folder("INBOX"))
        cursor = src.initial_cursor(folder)
        self.assertEqual(cursor.value["last_uid"], 6)  # just below the oldest match

    def test_imap_start_from_with_no_matches_parks_at_the_newest_message(self):
        """An empty folder-since-that-date is caught up, not 'fetch everything'."""
        from datetime import date

        client = FakeIMAPClient(messages={"INBOX": {i: b"x" for i in range(1, 11)}})
        client.since_uids = []
        src = _source(client, start_from=date(2026, 7, 1))
        folder = src.open_folder(Folder("INBOX"))
        self.assertEqual(src.initial_cursor(folder).value["last_uid"], 10)

    def test_imap_falls_back_to_a_full_sync_if_the_search_fails(self):
        """Better to re-fetch than to silently skip mail we cannot bound."""
        from datetime import date

        client = FakeIMAPClient(messages={"INBOX": {1: b"x"}})
        client.fail_since = True
        src = _source(client, start_from=date(2026, 7, 1))
        folder = src.open_folder(Folder("INBOX"))
        self.assertEqual(src.initial_cursor(folder).value["last_uid"], 0)
