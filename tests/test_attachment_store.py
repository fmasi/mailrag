import os, shutil, tempfile, unittest
from unittest import mock
from src.attachments.store import AttachmentStore, AttachmentMeta


class TestAttachmentStore(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.d, True)
        self.store = AttachmentStore(self.d)

    def tearDown(self):
        self.store.close()

    def _put(self, data=b"hello", mid="<m1>", tid="t1", name="a.txt", mime="text/plain"):
        return self.store.put(data, message_id=mid, thread_id=tid, filename=name,
                              mime=mime, size=len(data), source_type="eml",
                              source_ref="/x/a.eml")

    def test_put_writes_blob_and_row_and_dedups(self):
        sha = self._put()
        self.assertTrue(os.path.exists(self.store.path_for(sha)))
        self.assertEqual(self.store.get_bytes(sha), b"hello")
        # same bytes again (different message) -> same sha, blob written once
        blob = self.store.path_for(sha)
        mtime = os.path.getmtime(blob)
        sha2 = self._put(mid="<m2>")
        self.assertEqual(sha, sha2)
        self.assertEqual(os.path.getmtime(blob), mtime)   # not rewritten

    def test_list_for_message_and_thread(self):
        self._put(name="a.txt")
        self._put(data=b"world", name="b.txt", mime="text/plain")
        metas = self.store.list_for(message_id="<m1>")
        self.assertEqual({m.filename for m in metas}, {"a.txt", "b.txt"})
        self.assertTrue(all(isinstance(m, AttachmentMeta) for m in metas))
        self.assertEqual(len(self.store.list_for(thread_id="t1")), 2)
        self.assertEqual(self.store.list_for(message_id="<none>"), [])

    def test_path_for_unknown_raises(self):
        with self.assertRaises(KeyError):
            self.store.get_bytes("deadbeef")

    def test_get_text_extracts_and_caches(self):
        from src.attachments.extract import default_extractor_name
        sha = self._put(data=b"hello cache", name="a.txt", mime="text/plain")
        r1 = self.store.get_text(sha)
        self.assertEqual(r1.status, "extracted")
        self.assertIn("hello cache", r1.text)
        # Poison the cache with a sentinel to prove the second call reads from DB:
        self.store._conn.execute(
            "UPDATE text_cache SET text=? WHERE sha256=? AND extractor=?",
            ("SENTINEL_CACHE", sha, default_extractor_name()))
        self.store._conn.commit()
        r2 = self.store.get_text(sha)
        self.assertEqual(r2.text, "SENTINEL_CACHE",
                         "second get_text must return the cached value, not re-extract")
        # Confirm extractor_used is recorded on the cache row:
        row = self.store._conn.execute(
            "SELECT extractor_used FROM text_cache WHERE sha256=?", (sha,)).fetchone()
        self.assertIsNotNone(row["extractor_used"])

    def test_fetch_always_returns_path_even_when_unsupported(self):
        sha = self._put(data=b"\x00\x01", name="x.bin", mime="application/x-thing")
        f = self.store.fetch(sha)
        self.assertEqual(f["text_status"], "unsupported")
        self.assertEqual(f["text"], "")
        self.assertTrue(os.path.exists(f["path"]))
        self.assertEqual(f["filename"], "x.bin")

    def test_get_text_caches_per_extractor_and_records_used(self):
        sha = self._put(data=b"plain body", name="a.txt", mime="text/plain")
        r = self.store.get_text(sha)                 # default extractor
        self.assertEqual(r.status, "extracted")
        row = self.store._conn.execute(
            "SELECT extractor, extractor_used, status FROM text_cache WHERE sha256=?",
            (sha,)).fetchone()
        self.assertIsNotNone(row["extractor_used"])

    def test_cache_hit_preserves_extractor_used(self):
        """A cache hit must report the extractor that actually produced the text
        (extractor_used), not the registry name the cache row is keyed by."""
        sha = self._put(data=b"plain body", name="a.txt", mime="text/plain")
        first = self.store.get_text(sha)            # miss -> real extraction
        self.assertEqual(first.extractor, "plaintext")
        hit = self.store.get_text(sha)              # hit -> same answer
        self.assertEqual(hit.extractor, first.extractor)

    def test_fetch_reads_attachment_row_once(self):
        sha = self._put(data=b"plain body", name="a.txt", mime="text/plain")
        queries = []
        self.store._conn.set_trace_callback(queries.append)
        self.store.fetch(sha)                       # cache miss -> extracts
        self.store._conn.set_trace_callback(None)
        meta_reads = [q for q in queries if "FROM attachments" in q]
        self.assertEqual(len(meta_reads), 1,
                         "fetch must not query the attachments row twice")

    def test_force_bypasses_cache_and_reextracts(self):
        from src.attachments.extract import default_extractor_name
        sha = self._put(data=b"plain body", name="a.txt", mime="text/plain")
        first = self.store.get_text(sha)
        self.assertEqual(first.text.strip(), "plain body")
        # Poison the cache with a sentinel:
        self.store._conn.execute(
            "UPDATE text_cache SET text=? WHERE sha256=? AND extractor=?",
            ("SENTINEL", sha, default_extractor_name()))
        self.store._conn.commit()
        # Plain get_text reads the (poisoned) cache -> proves cache-hit:
        self.assertEqual(self.store.get_text(sha).text, "SENTINEL")
        # force=True bypasses + overwrites with a real extraction:
        forced = self.store.get_text(sha, force=True)
        self.assertEqual(forced.text.strip(), "plain body")
        # and the cache row was overwritten (no longer the sentinel):
        row = self.store._conn.execute(
            "SELECT text FROM text_cache WHERE sha256=? AND extractor=?",
            (sha, default_extractor_name())).fetchone()
        self.assertEqual(row["text"].strip(), "plain body")


if __name__ == "__main__":
    unittest.main()
