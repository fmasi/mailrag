"""Loader honors an explicit eml_files list (parses only those). Devcontainer."""

import os
import tempfile
import unittest

from src.data.loaders.mail_archive_x import MailArchiveXLoader


def _eml(subject):
    return (
        b"From: x@example.com\r\n"
        b"Subject: " + subject.encode() + b"\r\n"
        b"Content-Type: text/plain; charset=utf-8\r\n\r\n"
        b"body\r\n"
    )


class TestLoaderFileList(unittest.TestCase):
    def test_loads_only_listed_files(self):
        with tempfile.TemporaryDirectory() as root:
            paths = {}
            for name in ("a", "b", "c"):
                p = os.path.join(root, f"{name}.eml")
                with open(p, "wb") as fh:
                    fh.write(_eml(name))
                paths[name] = p

            emails = MailArchiveXLoader(eml_files=[paths["a"], paths["c"]]).load()
            self.assertEqual(sorted(e.subject for e in emails), ["a", "c"])

    def test_requires_a_source(self):
        with self.assertRaises(ValueError):
            MailArchiveXLoader()


if __name__ == "__main__":
    unittest.main()
