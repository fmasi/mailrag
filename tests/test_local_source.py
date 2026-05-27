"""Tests for resolving the index file-list: folder selection minus blacklist.

Stdlib-only (selection + blacklist), host-runnable.
"""
import os
import tempfile
import unittest

from src.data import blacklist
from src.ingest import local_source


def _make(root, rels):
    """Create each rel file with its own path as content (distinct hashes)."""
    for rel in rels:
        p = os.path.join(root, *rel.split("/"))
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as fh:
            fh.write(rel)


class TestResolveIndexFiles(unittest.TestCase):
    def test_selects_files_matching_rules(self):
        with tempfile.TemporaryDirectory() as root:
            _make(root, [
                "Inbox/Acme Corp/a.eml",
                "Inbox/Google/b.eml",
                "Acme Corp/Archive/c.eml",
            ])
            rules = [
                {"type": "prefix", "value": "Inbox/Acme Corp/"},
                {"type": "prefix", "value": "Acme Corp/"},
            ]
            kept, skipped = local_source.resolve_index_files(root, rules)
            self.assertEqual(
                [os.path.relpath(p, root).replace(os.sep, "/") for p in kept],
                ["Acme Corp/Archive/c.eml", "Inbox/Acme Corp/a.eml"],
            )
            self.assertEqual(skipped, [])

    def test_drops_blacklisted_files(self):
        with tempfile.TemporaryDirectory() as root:
            _make(root, ["Inbox/Acme Corp/a.eml", "Inbox/Acme Corp/b.eml"])
            bl = os.path.join(root, "bl.txt")
            blacklist.append_to_blacklist(
                bl, [blacklist.file_sha256(os.path.join(root, "Inbox", "Acme Corp", "b.eml"))]
            )
            rules = [{"type": "prefix", "value": "Inbox/Acme Corp/"}]
            kept, skipped = local_source.resolve_index_files(root, rules, blacklist_path=bl)
            self.assertEqual([os.path.basename(p) for p in kept], ["a.eml"])
            self.assertEqual([os.path.basename(p) for p in skipped], ["b.eml"])


if __name__ == "__main__":
    unittest.main()
