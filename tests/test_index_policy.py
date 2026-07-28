"""Index policy fingerprint — stop an append mixing incompatible vectors.

Idea adapted from msgvault (MIT, (c) 2025-2026 Wes McKinney); see
src/indexing/policy.py and NOTICE.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

from src.indexing.policy import (
    CHUNK_POLICY_VERSION,
    PREPROCESS_VERSION,
    describe_mismatch,
    policy_fingerprint,
)
from src.ingest import hybrid_qdrant as hq


def _fp(**kw):
    base = dict(chunk_size=512, chunk_overlap=64, embed_summary=True, embedder_name="Bge", dim=1024)
    base.update(kw)
    return policy_fingerprint(**base)


class TestPolicyFingerprint(unittest.TestCase):
    def test_is_stable_for_identical_settings(self):
        self.assertEqual(_fp(), _fp())

    def test_is_short_enough_for_a_payload_keyword(self):
        self.assertEqual(len(_fp()), 16)

    def test_chunk_size_changes_it(self):
        """The easiest mistake to make — re-indexing at 1024 into a 512 collection."""
        self.assertNotEqual(_fp(chunk_size=512), _fp(chunk_size=1024))

    def test_chunk_overlap_changes_it(self):
        self.assertNotEqual(_fp(chunk_overlap=64), _fp(chunk_overlap=128))

    def test_embed_summary_changes_it(self):
        """Contextual retrieval changes what is embedded, not just the metadata."""
        self.assertNotEqual(_fp(embed_summary=True), _fp(embed_summary=False))

    def test_embedder_and_dim_change_it(self):
        self.assertNotEqual(_fp(embedder_name="Bge"), _fp(embedder_name="Nvidia"))
        self.assertNotEqual(_fp(dim=1024), _fp(dim=2048))

    def test_the_version_constants_are_folded_in(self):
        with mock.patch("src.indexing.policy.PREPROCESS_VERSION", PREPROCESS_VERSION + 1):
            bumped = _fp()
        self.assertNotEqual(bumped, _fp())
        with mock.patch("src.indexing.policy.CHUNK_POLICY_VERSION", CHUNK_POLICY_VERSION + 1):
            bumped = _fp()
        self.assertNotEqual(bumped, _fp())

    def test_the_mismatch_message_names_both_policies_and_the_fix(self):
        msg = describe_mismatch("personal", "aaaa", "bbbb")
        self.assertIn("personal", msg)
        self.assertIn("aaaa", msg)
        self.assertIn("bbbb", msg)
        self.assertIn("--recreate", msg)


class TestCollectionPolicy(unittest.TestCase):
    @staticmethod
    def _client(payloads):
        client = mock.Mock()
        client.scroll.return_value = ([SimpleNamespace(payload=p) for p in payloads], None)
        return client

    def test_reads_the_fingerprint_from_a_sampled_point(self):
        self.assertEqual(
            hq.collection_policy(self._client([{"policy_fingerprint": "abc"}]), "c"), "abc"
        )

    def test_an_empty_collection_reports_no_policy(self):
        self.assertEqual(hq.collection_policy(self._client([]), "c"), "")

    def test_a_point_without_a_fingerprint_reports_no_policy(self):
        """Pre-fingerprint collections must read as 'unknown', not as a mismatch —
        the legacy guard is what handles those."""
        self.assertEqual(hq.collection_policy(self._client([{"text": "x"}]), "c"), "")

    def test_an_unreachable_collection_reports_no_policy(self):
        client = mock.Mock()
        client.scroll.side_effect = RuntimeError("not found")
        self.assertEqual(hq.collection_policy(client, "c"), "")

    def test_tolerates_a_null_payload(self):
        self.assertEqual(hq.collection_policy(self._client([None]), "c"), "")


class TestFingerprintRobustness(unittest.TestCase):
    def test_a_non_numeric_dim_does_not_raise(self):
        """A duck-typed embedder can expose anything; the fingerprint helper must
        never be the thing that fails a build."""
        self.assertTrue(
            policy_fingerprint(
                chunk_size=512,
                chunk_overlap=64,
                embed_summary=True,
                embedder_name="x",
                dim=object(),
            )
        )

    def test_a_non_numeric_chunk_size_does_not_raise(self):
        self.assertTrue(
            policy_fingerprint(
                chunk_size=None, chunk_overlap=64, embed_summary=False, embedder_name="x", dim=1024
            )
        )


if __name__ == "__main__":
    unittest.main()
