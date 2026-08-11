"""Tests for `scripts/noise.py purge` — what it deletes, and what it must not.

`purge` used to remove matched emails from Qdrant *and*, optionally, delete the
underlying `.eml` blobs from Azure. The Azure path was retired in #49, which
narrowed the command's contract: it now removes points from the index only and
leaves the source corpus on disk untouched.

That is a capability removal, so it is pinned here rather than left to the
commit message. The assertions are deliberately positive — the source file is
asserted to *still exist and still hold its original bytes*, not merely that
some deletion helper went uncalled — because an "assert nothing happened" test
passes just as happily when the code under test never ran at all.
"""

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import noise  # noqa: E402


class _Rules:
    """Stand-in NoiseFilter that condemns exactly one sender domain."""

    def is_empty(self):
        return False

    def category_names(self):
        return ["spam"]

    def match_payload(self, payload):
        if "@spam.example" in payload.get("sender", ""):
            return True, "spam"
        return False, None


class TestPurgeLeavesTheSourceCorpusAlone(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.eml = Path(self._tmp.name) / "noisy.eml"
        self.original_bytes = b"From: bulk@spam.example\nSubject: Buy now\n\nbody\n"
        self.eml.write_bytes(self.original_bytes)

        self.points = [
            (
                "point-1",
                {"sender": "bulk@spam.example", "subject": "Buy now", "source_id": str(self.eml)},
            ),
            (
                "point-2",
                {
                    "sender": "colleague@work.example",
                    "subject": "Q3 plan",
                    "source_id": "/somewhere/keep.eml",
                },
            ),
        ]

    def _run_purge(self, dry_run=False, confirm=True):
        deleted = []

        def fake_delete(qdrant, collection, point_ids):
            deleted.extend(point_ids)
            return len(point_ids)

        with (
            mock.patch.object(noise, "_scroll_all", return_value=iter(self.points)),
            mock.patch.object(noise, "_delete_qdrant_points", side_effect=fake_delete),
            mock.patch.object(noise, "_confirm", return_value=confirm),
            mock.patch(
                "src.data.noise_filter.NoiseFilter.from_project_rules", return_value=_Rules()
            ),
        ):
            noise.cmd_purge(SimpleNamespace(dry_run=dry_run), mock.Mock(), "a-collection")
        return deleted

    def test_matched_points_are_deleted_from_the_index(self):
        """The half of the contract that survived: noise leaves the index."""
        deleted = self._run_purge()
        self.assertEqual(deleted, ["point-1"])

    def test_the_matched_emails_source_file_survives_on_disk(self):
        """The half that changed: purge is an index operation, not a corpus one."""
        self._run_purge()
        self.assertTrue(self.eml.exists(), "purge must not delete the source .eml")
        self.assertEqual(self.eml.read_bytes(), self.original_bytes)

    def test_non_matching_points_are_left_in_the_index(self):
        deleted = self._run_purge()
        self.assertNotIn("point-2", deleted)

    def test_declining_the_prompt_deletes_nothing(self):
        deleted = self._run_purge(confirm=False)
        self.assertEqual(deleted, [])
        self.assertTrue(self.eml.exists())

    def test_dry_run_deletes_nothing(self):
        deleted = self._run_purge(dry_run=True)
        self.assertEqual(deleted, [])
        self.assertTrue(self.eml.exists())


class TestNoBlobDeletionPathRemains(unittest.TestCase):
    """Guard the removal itself, so it cannot be quietly reintroduced."""

    def test_the_azure_blob_helpers_are_gone(self):
        self.assertFalse(hasattr(noise, "_delete_blobs"))
        self.assertFalse(hasattr(noise, "_blob_path"))


if __name__ == "__main__":
    unittest.main()
