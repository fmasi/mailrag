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
        sha = self._put(data=b"hello cache", name="a.txt", mime="text/plain")
        from src.attachments import store as store_mod
        with mock.patch.object(store_mod, "extract_text",
                               wraps=store_mod.extract_text) as spy:
            r1 = self.store.get_text(sha)
            r2 = self.store.get_text(sha)
        self.assertEqual(r1.status, "extracted")
        self.assertIn("hello cache", r1.text)
        self.assertEqual(r2.text, r1.text)
        spy.assert_called_once()                 # second call hit the cache

    def test_fetch_always_returns_path_even_when_binary(self):
        sha = self._put(data=b"\x00\x01", name="x.bin", mime="application/x-thing")
        f = self.store.fetch(sha)
        self.assertEqual(f["text_status"], "binary")
        self.assertEqual(f["text"], "")
        self.assertTrue(os.path.exists(f["path"]))
        self.assertEqual(f["filename"], "x.bin")


if __name__ == "__main__":
    unittest.main()
