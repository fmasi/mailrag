"""Tests for the sha256(raw .eml) blacklist (stdlib-only, host-runnable)."""

import hashlib
import os
import tempfile
import unittest

from src.data import blacklist


def _write(path, data=b"data"):
    with open(path, "wb") as fh:
        fh.write(data)


class TestFileSha256(unittest.TestCase):
    def test_matches_hashlib_and_is_content_addressed(self):
        with tempfile.TemporaryDirectory() as d:
            a, b, c = (os.path.join(d, n) for n in ("a", "b", "c"))
            _write(a, b"hello")
            _write(b, b"hello")  # identical content
            _write(c, b"world")
            self.assertEqual(blacklist.file_sha256(a), hashlib.sha256(b"hello").hexdigest())
            self.assertEqual(blacklist.file_sha256(a), blacklist.file_sha256(b))
            self.assertNotEqual(blacklist.file_sha256(a), blacklist.file_sha256(c))


class TestLoadBlacklist(unittest.TestCase):
    def test_missing_file_is_empty(self):
        self.assertEqual(blacklist.load_blacklist("/no/such/file"), set())

    def test_reads_hashes_ignoring_blanks_and_comments(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "bl.txt")
            with open(p, "w") as fh:
                fh.write("# header\nabc123\n\n  def456  \n# note\n")
            self.assertEqual(blacklist.load_blacklist(p), {"abc123", "def456"})


class TestAppendToBlacklist(unittest.TestCase):
    def test_appends_only_new_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "bl.txt")
            self.assertEqual(blacklist.append_to_blacklist(p, ["h1", "h2"]), 2)
            self.assertEqual(blacklist.append_to_blacklist(p, ["h2", "h3"]), 1)
            self.assertEqual(blacklist.load_blacklist(p), {"h1", "h2", "h3"})


class TestFilterBlacklisted(unittest.TestCase):
    def test_splits_paths_into_kept_and_skipped(self):
        with tempfile.TemporaryDirectory() as d:
            keep, drop = os.path.join(d, "keep.eml"), os.path.join(d, "drop.eml")
            _write(keep, b"good email")
            _write(drop, b"junk email")
            bl = os.path.join(d, "bl.txt")
            blacklist.append_to_blacklist(bl, [blacklist.file_sha256(drop)])

            kept, skipped = blacklist.filter_blacklisted([keep, drop], bl)
            self.assertEqual(kept, [keep])
            self.assertEqual(skipped, [drop])

    def test_empty_blacklist_keeps_everything(self):
        with tempfile.TemporaryDirectory() as d:
            f = os.path.join(d, "x.eml")
            _write(f)
            kept, skipped = blacklist.filter_blacklisted([f], os.path.join(d, "none.txt"))
            self.assertEqual((kept, skipped), ([f], []))


if __name__ == "__main__":
    unittest.main()
