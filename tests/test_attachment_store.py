import os, shutil, tempfile, unittest
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


if __name__ == "__main__":
    unittest.main()
