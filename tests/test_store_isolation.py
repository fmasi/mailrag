"""Attachment stores must be separate per collection, with no shared state.

Measured on a real machine before this change: with one shared store, four
thread ids existed in BOTH a work corpus and a personal one, so listing either
returned the other's attachments — and `get_attachment` accepted any sha256 from
any corpus because a content hash carried no corpus with it.

These tests pin physical separation rather than a filter. A filter is one
forgotten `WHERE` clause away from leaking; separate directories are not.
"""

import os
import shutil
import tempfile
import unittest
from unittest import mock

from src.attachments.store import AttachmentStore
from src.mcp_server import server


class TestStorePathsAreSeparate(unittest.TestCase):
    def test_each_collection_gets_its_own_directory(self):
        with mock.patch.dict(os.environ, {"RAG_ATTACH_STORE": "/base"}):
            work = server.resolve_attach_store(collection="work-rag")
            personal = server.resolve_attach_store(collection="personal-rag")
        self.assertNotEqual(work, personal)
        self.assertTrue(work.startswith("/base"))
        self.assertTrue(personal.startswith("/base"))

    def test_an_explicit_store_still_wins(self):
        self.assertEqual(
            server.resolve_attach_store("/explicit", collection="anything"), "/explicit"
        )

    def test_collection_names_cannot_escape_the_root(self):
        # Collection names come from config; a traversal attempt must not walk out.
        for hostile in ("../../etc/passwd", "/absolute/path", "..", "a/b/c"):
            with self.subTest(name=hostile):
                seg = server._safe_dirname(hostile)
                self.assertNotIn("/", seg)
                self.assertNotIn("..", seg)

    def test_an_empty_name_still_yields_a_usable_segment(self):
        self.assertTrue(server._safe_dirname("!!!"))


class TestNoCrossCorpusReads(unittest.TestCase):
    """The property that matters: one corpus cannot see the other's rows."""

    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="iso_")
        self.work = AttachmentStore(os.path.join(self.root, "work"))
        self.personal = AttachmentStore(os.path.join(self.root, "personal"))
        # Deliberately COLLIDING identifiers: the same thread id in both corpora
        # is what actually happened on the real machine.
        self.work_sha = self.work.put(
            b"work-doc",
            message_id="<m1>",
            thread_id="shared-thread",
            filename="work.pdf",
            mime="application/pdf",
            size=8,
            source_type="eml",
            source_ref="/w.eml",
        )
        self.personal_sha = self.personal.put(
            b"personal-doc",
            message_id="<m2>",
            thread_id="shared-thread",
            filename="private.pdf",
            mime="application/pdf",
            size=12,
            source_type="eml",
            source_ref="/p.eml",
        )

    def tearDown(self):
        self.work.close()
        self.personal.close()
        shutil.rmtree(self.root, ignore_errors=True)

    def test_a_colliding_thread_id_returns_only_its_own_corpus(self):
        work_rows = [m.filename for m in self.work.list_for(thread_id="shared-thread")]
        personal_rows = [m.filename for m in self.personal.list_for(thread_id="shared-thread")]
        self.assertEqual(work_rows, ["work.pdf"])
        self.assertEqual(personal_rows, ["private.pdf"])

    def test_a_hash_from_the_other_corpus_does_not_resolve(self):
        # A content hash is not a capability. Before the split, any sha256
        # fetched from any corpus.
        with self.assertRaises(Exception):
            self.work.fetch(self.personal_sha)

    def test_blobs_are_not_shared_on_disk(self):
        self.assertFalse(os.path.exists(self.work.path_for(self.personal_sha)))
        self.assertFalse(os.path.exists(self.personal.path_for(self.work_sha)))

    def test_names_for_does_not_cross_corpora(self):
        self.assertEqual(self.work.names_for(thread_id="shared-thread"), ["work.pdf"])
        self.assertEqual(self.personal.names_for(message_ids=["<m1>"]), [])


class TestLegacySharedStoreIsRefused(unittest.TestCase):
    """A pre-split store holds every corpus at once and cannot be split later.

    Rows record the file they came from, not the collection they were indexed
    into, so there is no safe automatic migration — only a clear refusal and a
    cheap rebuild.
    """

    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="legacy_")

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_flat_index_at_the_root_is_refused(self):
        open(os.path.join(self.root, "index.db"), "wb").close()
        with self.assertRaises(ValueError) as ctx:
            server.assert_no_legacy_shared_store(self.root)
        msg = str(ctx.exception)
        self.assertIn("attachments build", msg)
        self.assertIn("per-collection", msg)

    def test_a_split_layout_is_accepted(self):
        os.makedirs(os.path.join(self.root, "work-rag"), exist_ok=True)
        open(os.path.join(self.root, "work-rag", "index.db"), "wb").close()
        server.assert_no_legacy_shared_store(self.root)  # must not raise

    def test_an_absent_store_is_accepted(self):
        server.assert_no_legacy_shared_store(os.path.join(self.root, "nothing-here"))


class TestAttachmentToolsRequireACollection(unittest.TestCase):
    def test_listing_without_a_collection_refuses_rather_than_guessing(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MAILRAG_COLLECTION", None)
            with mock.patch("src.mcp_server.server.latest_manifest_collection", return_value=None):
                with self.assertRaises(ValueError) as ctx:
                    server.list_attachments(thread_id="t1")
        self.assertIn("collection", str(ctx.exception))


class _Ctx:
    def __init__(self, thread_id="shared-thread", message_ids=("<m1>",)):
        self.thread_id = thread_id
        self.emails = [_Email(mid) for mid in message_ids]


class _Email:
    def __init__(self, message_id):
        self.message_id = message_id


class TestAttachmentNamesNeverFallsBackToTheUnscopedStore(unittest.TestCase):
    """A collection whose scoped store fails to open must not leak another
    corpus's attachment names.

    `_attachment_store()` never raises: when ``AttachmentStore(...)`` blows up
    opening a collection's subdirectory (locked/corrupted ``index.db``,
    disk-full, permissions), it swallows the exception and yields
    ``store=None``. `_attachment_names(ctx, store=None)` must not then reach
    for the plain, unscoped default store as a silent fallback — that store
    can be a completely different corpus (or a stale pre-split legacy store),
    and returning its names under a result labeled with the *correct*
    collection is a cross-corpus data leak.
    """

    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="unscoped_leak_")
        # The plain, UNSCOPED default store living at the store root (no
        # collection subdirectory) — exactly what `resolve_attach_store()`
        # with no arguments resolves to. It holds an attachment filed under
        # the SAME thread/message ids the scoped lookup will use, so a leak
        # is directly observable rather than incidentally invisible.
        self.unscoped = AttachmentStore(self.root)
        self.unscoped.put(
            b"unrelated-doc",
            message_id="<m1>",
            thread_id="shared-thread",
            filename="other-corpus-secret.pdf",
            mime="application/pdf",
            size=13,
            source_type="eml",
            source_ref="/other.eml",
        )
        self.env = mock.patch.dict(os.environ, {"RAG_ATTACH_STORE": self.root})
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.unscoped.close()
        shutil.rmtree(self.root, ignore_errors=True)

    def test_a_failed_scoped_store_does_not_leak_the_default_store_names(self):
        # This is the state `_attachment_store(collection)` leaves behind when
        # `AttachmentStore(...)` raises for that collection's subdirectory: it
        # swallows the exception and yields `store=None`.
        names = server._attachment_names(_Ctx(), store=None)
        self.assertIsNone(names)

    def test_thread_meta_omits_the_field_rather_than_leaking_it(self):
        meta = server._thread_meta(_Ctx(), store=None)
        self.assertNotIn("attachment_names", meta)


if __name__ == "__main__":
    unittest.main()
