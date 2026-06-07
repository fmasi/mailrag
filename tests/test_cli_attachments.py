import unittest
from unittest import mock

from src import cli


class TestAttachmentsCli(unittest.TestCase):
    def test_build_routes(self):
        prof = mock.Mock(selection_rules=[], blacklist=None)
        prof.resolved_root.return_value = "/root"
        with mock.patch("src.cli.CorpusProfile.load", return_value=prof), \
             mock.patch("src.cli.resolve_index_files", return_value=(["/root/a.eml"], [])), \
             mock.patch("src.cli.AttachmentStore") as store_cls, \
             mock.patch("src.cli.ingest_eml",
                        return_value={"emails": 1, "attachments": 2, "skipped": 0}) as ing:
            rc = cli.main(["attachments", "build", "--profile", "p.json",
                           "--store", "/tmp/st"])
        self.assertEqual(rc, 0)
        ing.assert_called_once()
        store_cls.assert_called_once_with("/tmp/st")

    def test_get_text_routes(self):
        store = mock.Mock()
        store.fetch.return_value = {"sha256": "ab", "filename": "x.pdf",
                                    "mime": "application/pdf", "size": 3,
                                    "text": "hello", "text_status": "extracted",
                                    "path": "/tmp/st/blobs/ab/abc"}
        with mock.patch("src.cli.AttachmentStore", return_value=store):
            rc = cli.main(["attachments", "get", "abc", "--text", "--store", "/tmp/st"])
        self.assertEqual(rc, 0)
        store.fetch.assert_called_once_with("abc")

    def test_list_routes(self):
        store = mock.Mock()
        store.list_for.return_value = []
        with mock.patch("src.cli.AttachmentStore", return_value=store):
            rc = cli.main(["attachments", "list", "--thread-id", "t1", "--store", "/tmp/st"])
        self.assertEqual(rc, 0)
        store.list_for.assert_called_once_with(thread_id="t1", message_id=None)


if __name__ == "__main__":
    unittest.main()
