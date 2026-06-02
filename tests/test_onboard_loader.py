import os
import tempfile
import unittest
from unittest import mock

from src import onboard


class TestSlug(unittest.TestCase):
    def test_collection_slug(self):
        self.assertEqual(onboard.collection_slug("/data/My Mail/"), "mailrag-my-mail")
        self.assertEqual(onboard.collection_slug("/x/Inbox_2024"), "mailrag-inbox-2024")


class TestLoadEmlDir(unittest.TestCase):
    def test_missing_dir_raises(self):
        with self.assertRaises(ValueError):
            onboard.load_eml_dir("/no/such/dir")

    def test_empty_dir_raises(self):
        d = tempfile.mkdtemp()
        with self.assertRaises(ValueError):
            onboard.load_eml_dir(d)

    def test_loads_eml_files_with_limit(self):
        d = tempfile.mkdtemp()
        for i in range(3):
            with open(os.path.join(d, f"{i}.eml"), "w") as fh:
                fh.write("From: s@x.com\nSubject: hi\n\nbody\n")
        with mock.patch("src.data.loaders.mail_archive_x.MailArchiveXLoader") as L:
            L.return_value.load.return_value = ["E1", "E2"]
            out = onboard.load_eml_dir(d, limit=2)
        # constructed with exactly 2 of the 3 paths
        self.assertEqual(len(L.call_args.kwargs["eml_files"]), 2)
        self.assertEqual(out, ["E1", "E2"])


if __name__ == "__main__":
    unittest.main()
