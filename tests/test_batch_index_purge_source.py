"""Tests for the --purge-source noise blob deletion helper."""

import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from batch_index_to_vector_store import _purge_noise_source_blobs  # noqa: E402


def _make_email(source_id: str):
    """Return a minimal email-like object with a source_id."""
    return SimpleNamespace(source_id=source_id)


class TestPurgeNoiseSourceBlobs(unittest.TestCase):
    """_purge_noise_source_blobs should delete the correct blob paths."""

    def _container(self):
        return MagicMock()

    def test_strips_tmp_prefix_and_deletes(self):
        """Blob name is derived by stripping the /tmp/<dir>/ prefix."""
        cc = self._container()
        emails = [_make_email("/tmp/tmpABC123/Inbox/email.eml")]
        purged, errors = _purge_noise_source_blobs(cc, emails)
        cc.delete_blob.assert_called_once_with("Inbox/email.eml")
        self.assertEqual(purged, 1)
        self.assertEqual(errors, 0)

    def test_deletes_multiple_blobs(self):
        cc = self._container()
        emails = [
            _make_email("/tmp/tmpXXX/a/one.eml"),
            _make_email("/tmp/tmpXXX/b/two.eml"),
            _make_email("/tmp/tmpXXX/three.eml"),
        ]
        purged, errors = _purge_noise_source_blobs(cc, emails)
        cc.delete_blob.assert_has_calls(
            [call("a/one.eml"), call("b/two.eml"), call("three.eml")],
            any_order=False,
        )
        self.assertEqual(purged, 3)
        self.assertEqual(errors, 0)

    def test_skips_emails_with_no_source_id(self):
        """Emails with an empty source_id are silently skipped."""
        cc = self._container()
        emails = [_make_email(""), _make_email("/tmp/tmpXXX/valid.eml")]
        purged, errors = _purge_noise_source_blobs(cc, emails)
        cc.delete_blob.assert_called_once_with("valid.eml")
        self.assertEqual(purged, 1)
        self.assertEqual(errors, 0)

    def test_empty_list_makes_no_calls(self):
        cc = self._container()
        purged, errors = _purge_noise_source_blobs(cc, [])
        cc.delete_blob.assert_not_called()
        self.assertEqual(purged, 0)
        self.assertEqual(errors, 0)

    def test_blob_deletion_error_is_counted_not_raised(self):
        """A failing delete_blob call increments errors but does not abort."""
        cc = self._container()
        cc.delete_blob.side_effect = Exception("blob not found")
        emails = [_make_email("/tmp/tmpXXX/a.eml"), _make_email("/tmp/tmpXXX/b.eml")]
        purged, errors = _purge_noise_source_blobs(cc, emails)
        self.assertEqual(errors, 2)
        self.assertEqual(purged, 0)

    def test_partial_failure_counts_correctly(self):
        """Only the failing blob increments errors; others still succeed."""
        cc = self._container()
        cc.delete_blob.side_effect = [None, Exception("gone"), None]
        emails = [
            _make_email("/tmp/tmpXXX/a.eml"),
            _make_email("/tmp/tmpXXX/b.eml"),
            _make_email("/tmp/tmpXXX/c.eml"),
        ]
        purged, errors = _purge_noise_source_blobs(cc, emails)
        self.assertEqual(purged, 2)
        self.assertEqual(errors, 1)

    def test_source_id_without_tmp_prefix_used_as_is(self):
        """source_ids that don't start with /tmp/ are passed through unchanged."""
        cc = self._container()
        emails = [_make_email("Inbox/nontmp.eml")]
        _purge_noise_source_blobs(cc, emails)
        cc.delete_blob.assert_called_once_with("Inbox/nontmp.eml")

    def test_nested_tmp_prefix_stripped_correctly(self):
        """Only the leading /tmp/<single-dir>/ is stripped, not deeper paths."""
        cc = self._container()
        emails = [_make_email("/tmp/tmpABCDEF/folder/sub/email.eml")]
        _purge_noise_source_blobs(cc, emails)
        cc.delete_blob.assert_called_once_with("folder/sub/email.eml")


if __name__ == "__main__":
    unittest.main()
