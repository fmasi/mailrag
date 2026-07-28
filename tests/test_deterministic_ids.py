"""Deterministic Qdrant point ids + incremental (append) indexing — issue #101.

Point ids used to be random per-run UUIDs, so re-indexing duplicated every chunk
and every build path had to drop the whole collection. These tests pin the
properties continuous sync depends on:

* the same email always yields the same point ids,
* an email's ids do not depend on what else is in the corpus,
* an email's body and attachment chunks share one ``message_key``,
* append mode deletes an email's existing points before upserting the new ones.
"""

from __future__ import annotations

import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest import mock

from src.data.identity import message_key
from src.data.models import NormalizedEmail
from src.indexing.point_ids import (
    MAILRAG_NAMESPACE,
    assign_deterministic_ids,
    content_hash,
    point_id,
)
from src.ingest import hybrid_qdrant as hq


def _email(**kw) -> NormalizedEmail:
    base = dict(
        sender="alice@example.com",
        subject="Quarterly numbers",
        date=datetime(2026, 1, 15, 9, 30, tzinfo=timezone.utc),
        body="The Q4 figure is 210000000.",
        source="local",
        source_id="/corpus/a.eml",
        message_id="<abc@example.com>",
    )
    base.update(kw)
    return NormalizedEmail(**base)


class TestPointId(unittest.TestCase):
    def test_is_stable_across_calls(self):
        self.assertEqual(point_id("body:k", 0), point_id("body:k", 0))

    def test_differs_per_chunk_index(self):
        self.assertNotEqual(point_id("body:k", 0), point_id("body:k", 1))

    def test_differs_per_doc_key(self):
        self.assertNotEqual(point_id("body:k1", 0), point_id("body:k2", 0))

    def test_is_a_uuid5_in_the_mailrag_namespace(self):
        import uuid

        self.assertEqual(point_id("body:k", 3), str(uuid.uuid5(MAILRAG_NAMESPACE, "body:k:3")))

    def test_returns_a_parseable_uuid_string(self):
        import uuid

        uuid.UUID(point_id("body:k", 0))  # raises if malformed

    def test_content_hash_is_stable_and_content_sensitive(self):
        self.assertEqual(content_hash("hello"), content_hash("hello"))
        self.assertNotEqual(content_hash("hello"), content_hash("hello "))


class TestAssignDeterministicIds(unittest.TestCase):
    @staticmethod
    def _node(ref_doc_id, node_id="random"):
        return SimpleNamespace(ref_doc_id=ref_doc_id, node_id=node_id, id_=node_id)

    def test_numbers_chunks_within_their_own_document(self):
        nodes = [self._node("d1"), self._node("d1"), self._node("d2")]
        assign_deterministic_ids(nodes)
        self.assertEqual(nodes[0].id_, point_id("d1", 0))
        self.assertEqual(nodes[1].id_, point_id("d1", 1))
        self.assertEqual(nodes[2].id_, point_id("d2", 0))

    def test_a_documents_ids_do_not_depend_on_the_rest_of_the_corpus(self):
        """The property incremental indexing rests on: indexing an email alone and
        indexing it alongside a thousand others must produce identical ids."""
        alone = [self._node("d1"), self._node("d1")]
        crowded = [self._node("other"), self._node("d1"), self._node("d1"), self._node("more")]
        assign_deterministic_ids(alone)
        assign_deterministic_ids(crowded)
        self.assertEqual([n.id_ for n in alone], [crowded[1].id_, crowded[2].id_])

    def test_is_idempotent(self):
        nodes = [self._node("d1"), self._node("d1")]
        first = [n.id_ for n in assign_deterministic_ids(nodes)]
        second = [n.id_ for n in assign_deterministic_ids(nodes)]
        self.assertEqual(first, second)

    def test_falls_back_to_node_id_when_there_is_no_source_document(self):
        n = SimpleNamespace(ref_doc_id=None, node_id="standalone", id_="standalone")
        assign_deterministic_ids([n])
        self.assertEqual(n.id_, point_id("standalone", 0))

    def test_handles_an_empty_batch(self):
        self.assertEqual(assign_deterministic_ids([]), [])


class TestMessageKey(unittest.TestCase):
    def test_prefers_the_normalized_message_id(self):
        self.assertEqual(_email().message_key(), "abc@example.com")

    def test_falls_back_to_the_content_hash_without_a_message_id(self):
        e = _email(message_id=None)
        self.assertEqual(
            e.message_key(),
            message_key(sender=e.sender, subject=e.subject, date=e.date, body=e.body),
        )
        self.assertEqual(len(e.message_key()), 64)  # sha256 hex

    def test_survives_a_reexport_that_only_reformats_the_body(self):
        """content_sha256 normalizes line endings and trailing whitespace, so a
        re-exported .eml must not read as a different email."""
        a = _email(message_id=None, body="line one\nline two")
        b = _email(message_id=None, body="line one  \r\nline two\r\n")
        self.assertEqual(a.message_key(), b.message_key())

    def test_differs_for_different_emails(self):
        self.assertNotEqual(_email().message_key(), _email(message_id="<zzz@x>").message_key())


class TestDocumentIdentity(unittest.TestCase):
    def test_doc_id_defaults_to_the_stable_message_key(self):
        doc = _email().to_document()
        self.assertEqual(doc.doc_id, "body:abc@example.com")

    def test_doc_id_does_not_depend_on_corpus_position(self):
        self.assertEqual(_email().to_document().doc_id, _email().to_document().doc_id)

    def test_message_key_is_exposed_as_payload_metadata(self):
        self.assertEqual(_email().to_document().metadata["message_key"], "abc@example.com")

    def test_message_key_is_excluded_from_the_embedded_text(self):
        doc = _email().to_document()
        self.assertIn("message_key", doc.excluded_embed_metadata_keys)
        self.assertIn("message_key", doc.excluded_llm_metadata_keys)

    def test_an_explicit_doc_id_is_still_honoured(self):
        self.assertEqual(_email().to_document(doc_id="legacy_0").doc_id, "legacy_0")


class TestDeleteByMessageKeys(unittest.TestCase):
    def test_deletes_with_a_match_any_filter_on_message_key(self):
        client = mock.Mock()
        n = hq.delete_by_message_keys(client, "c", ["k1", "k2"])
        self.assertEqual(n, 2)
        client.delete.assert_called_once()
        flt = client.delete.call_args.kwargs["points_selector"].filter
        cond = flt.must[0]
        self.assertEqual(cond.key, "message_key")
        self.assertEqual(list(cond.match.any), ["k1", "k2"])

    def test_no_ops_on_an_empty_key_set(self):
        """An unfiltered delete would wipe the collection — the guard is load-bearing."""
        client = mock.Mock()
        self.assertEqual(hq.delete_by_message_keys(client, "c", []), 0)
        client.delete.assert_not_called()

    def test_ignores_empty_and_duplicate_keys(self):
        client = mock.Mock()
        self.assertEqual(hq.delete_by_message_keys(client, "c", ["k", "", None, "k"]), 1)
        cond = client.delete.call_args.kwargs["points_selector"].filter.must[0]
        self.assertEqual(list(cond.match.any), ["k"])

    def test_batches_large_key_sets(self):
        client = mock.Mock()
        keys = [f"k{i}" for i in range(600)]
        self.assertEqual(hq.delete_by_message_keys(client, "c", keys), 600)
        self.assertEqual(client.delete.call_count, 3)  # 256 + 256 + 88

    def test_ensure_payload_indexes_covers_message_key_and_content_hash(self):
        client = mock.Mock()
        hq.ensure_payload_indexes(client, "c")
        indexed = {c.kwargs["field_name"] for c in client.create_payload_index.call_args_list}
        self.assertIn("message_key", indexed)
        self.assertIn("content_hash", indexed)

    def test_ensure_payload_indexes_survives_an_already_indexed_field(self):
        client = mock.Mock()
        client.create_payload_index.side_effect = RuntimeError("already exists")
        hq.ensure_payload_indexes(client, "c")  # must not raise


if __name__ == "__main__":
    unittest.main()


class TestLegacyCollectionGuard(unittest.TestCase):
    """Appending into a pre-deterministic-id collection duplicates it (#101 risk 5)."""

    @staticmethod
    def _client(payloads):
        client = mock.Mock()
        client.scroll.return_value = ([SimpleNamespace(payload=p) for p in payloads], None)
        return client

    def test_points_without_message_key_are_legacy(self):
        self.assertTrue(hq.has_legacy_points(self._client([{"text": "x"}]), "c"))

    def test_points_with_message_key_are_not_legacy(self):
        self.assertFalse(hq.has_legacy_points(self._client([{"message_key": "k"}]), "c"))

    def test_a_single_legacy_point_condemns_the_collection(self):
        client = self._client([{"message_key": "k"}, {"text": "old"}])
        self.assertTrue(hq.has_legacy_points(client, "c"))

    def test_an_empty_collection_is_not_legacy(self):
        """A freshly created collection must not trip the guard."""
        self.assertFalse(hq.has_legacy_points(self._client([]), "c"))

    def test_an_unreachable_collection_is_not_legacy(self):
        client = mock.Mock()
        client.scroll.side_effect = RuntimeError("not found")
        self.assertFalse(hq.has_legacy_points(client, "c"))

    def test_handles_a_null_payload(self):
        self.assertTrue(hq.has_legacy_points(self._client([None]), "c"))
