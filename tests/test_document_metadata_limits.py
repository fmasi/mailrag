"""Tests for document metadata size limits.

These tests exist because batch indexing 63k+ emails surfaced three
distinct failure modes:

1. LlamaIndex ValueError — metadata length exceeds chunk size (1024 by default).
   The sentence splitter refuses to chunk a node whose metadata string alone
   is longer than ``Settings.chunk_size``.

2. Pinecone 400 Bad Request — metadata exceeds the 40 960-byte per-vector
   limit.  Even after excluding keys from the embedding context, the full
   metadata dict is still shipped to Pinecone and must stay under 40 KB.

3. Extreme edge cases — empty bodies, missing headers, non-ASCII senders,
   very long source_id paths — that should not crash the pipeline.

Run with:
    poetry run pytest tests/test_document_metadata_limits.py -v
"""

import json
import unittest
from datetime import datetime

from llama_index.core import Document

from src.data.models import (
    NormalizedEmail,
    _addresses_preview,
    _truncate,
)

# ---------------------------------------------------------------------------
# Pinecone hard limit (bytes).  Metadata is JSON-serialised before upload.
# ---------------------------------------------------------------------------
PINECONE_MAX_METADATA_BYTES = 40_960

# Default and increased chunk sizes we support.
CHUNK_SIZE_DEFAULT = 1024
CHUNK_SIZE_INCREASED = 2048


def _metadata_byte_size(doc: Document) -> int:
    """Return the JSON-encoded byte size of a Document's metadata dict."""
    return len(json.dumps(doc.metadata).encode("utf-8"))


def _effective_metadata_str_len(doc: Document) -> int:
    """Return the metadata string length that LlamaIndex uses for chunking.

    LlamaIndex builds a metadata string from keys that are NOT in
    ``excluded_embed_metadata_keys``.  This helper replicates that logic
    so we can assert it fits inside the chunk size.
    """
    excluded = set(doc.excluded_embed_metadata_keys or [])
    parts = []
    for key, value in doc.metadata.items():
        if key not in excluded:
            parts.append(f"{key}: {value}")
    return len("\n".join(parts))


def _make_email(**overrides) -> NormalizedEmail:
    """Build a NormalizedEmail with sensible defaults, overridable per-field."""
    defaults = dict(
        sender="alice@example.com",
        subject="Hello",
        date=datetime(2024, 1, 15, 9, 30),
        body="Test body content.",
        source="mail_archive_x",
        source_id="/tmp/test/inbox/msg.eml",
    )
    defaults.update(overrides)
    return NormalizedEmail(**defaults)


# ===================================================================
# 1. Truncation helper tests
# ===================================================================
class TestTruncate(unittest.TestCase):
    """Verify _truncate correctly caps strings."""

    def test_short_string_unchanged(self):
        self.assertEqual(_truncate("hello", 100), "hello")

    def test_exact_limit_unchanged(self):
        s = "a" * 256
        self.assertEqual(_truncate(s, 256), s)

    def test_over_limit_truncated_with_ellipsis(self):
        s = "a" * 300
        result = _truncate(s, 256)
        self.assertEqual(len(result), 256)
        self.assertTrue(result.endswith("..."))

    def test_empty_string(self):
        self.assertEqual(_truncate("", 100), "")


# ===================================================================
# 2. Address preview helper tests
# ===================================================================
class TestAddressesPreview(unittest.TestCase):
    """Verify _addresses_preview extracts clean email addresses."""

    def test_single_address(self):
        result = _addresses_preview("Alice <alice@example.com>")
        self.assertEqual(result, "alice@example.com")

    def test_multiple_addresses(self):
        raw = "Alice <alice@example.com>, Bob <bob@example.com>"
        result = _addresses_preview(raw)
        self.assertIn("alice@example.com", result)
        self.assertIn("bob@example.com", result)

    def test_deduplication(self):
        raw = "alice@example.com, alice@example.com, bob@example.com"
        result = _addresses_preview(raw)
        self.assertEqual(result.count("alice@example.com"), 1)

    def test_plain_address(self):
        result = _addresses_preview("alice@example.com")
        self.assertEqual(result, "alice@example.com")

    def test_empty_string_passthrough(self):
        result = _addresses_preview("")
        self.assertEqual(result, "")


# ===================================================================
# 3. Metadata fits within LlamaIndex chunk size
# ===================================================================
class TestMetadataFitsChunkSize(unittest.TestCase):
    """Ensure the metadata string used during chunking never exceeds chunk size.

    Reproduces the original failure:
      ValueError: Metadata length (1509) is longer than chunk size (1024)
    """

    def test_normal_email_metadata_under_default_chunk(self):
        doc = _make_email().to_document(doc_id="test_0")
        self.assertLess(
            _effective_metadata_str_len(doc),
            CHUNK_SIZE_DEFAULT,
            "Normal email metadata should fit in default chunk size",
        )

    def test_long_subject_metadata_under_increased_chunk(self):
        email = _make_email(subject="RE: " * 500)
        doc = email.to_document(doc_id="test_1")
        self.assertLess(
            _effective_metadata_str_len(doc),
            CHUNK_SIZE_INCREASED,
            "Very long subject should be truncated to fit chunk size",
        )

    def test_long_sender_metadata_under_chunk(self):
        email = _make_email(sender="a]" * 500 + "@example.com")
        doc = email.to_document(doc_id="test_2")
        self.assertLess(
            _effective_metadata_str_len(doc),
            CHUNK_SIZE_DEFAULT,
        )

    def test_many_recipients_metadata_under_chunk(self):
        """Simulate an email sent to 200+ recipients (common in mailing lists)."""
        recipients = ", ".join(f"user{i}@company.example.com" for i in range(200))
        email = _make_email(recipients=recipients)
        doc = email.to_document(doc_id="test_3")
        self.assertLess(
            _effective_metadata_str_len(doc),
            CHUNK_SIZE_INCREASED,
            "Large recipient list should stay under chunk size via exclusion + truncation",
        )

    def test_many_cc_metadata_under_chunk(self):
        cc = ", ".join(f"cc{i}@example.com" for i in range(200))
        email = _make_email(cc=cc)
        doc = email.to_document(doc_id="test_4")
        self.assertLess(
            _effective_metadata_str_len(doc),
            CHUNK_SIZE_INCREASED,
        )

    def test_all_fields_maxed_metadata_under_chunk(self):
        """Worst case: every field is at or near its truncation limit."""
        email = _make_email(
            sender="x" * 1000,
            subject="y" * 2000,
            recipients=", ".join(f"r{i}@example.com" for i in range(300)),
            cc=", ".join(f"c{i}@example.com" for i in range(300)),
            source_id="z" * 2000,
        )
        doc = email.to_document(doc_id="test_5")
        self.assertLess(
            _effective_metadata_str_len(doc),
            CHUNK_SIZE_INCREASED,
            "Even with all fields maxed, effective metadata must fit chunk size",
        )


# ===================================================================
# 4. Total metadata fits within Pinecone's 40 KB limit
# ===================================================================
class TestMetadataFitsPineconeLimit(unittest.TestCase):
    """Ensure serialised metadata never exceeds Pinecone's 40 960-byte limit.

    Reproduces the Pinecone 400 error:
      Metadata size is 49276 bytes, which exceeds the limit of 40960 bytes
    """

    def test_normal_email_under_pinecone_limit(self):
        doc = _make_email().to_document(doc_id="test_0")
        self.assertLess(
            _metadata_byte_size(doc),
            PINECONE_MAX_METADATA_BYTES,
        )

    def test_huge_recipient_list_under_pinecone_limit(self):
        """Simulate a mass-mailing with 500+ recipients."""
        recipients = ", ".join(
            f'"User Number {i}" <user{i}@very-long-domain-name.example.com>' for i in range(500)
        )
        email = _make_email(recipients=recipients)
        doc = email.to_document(doc_id="test_1")
        self.assertLess(
            _metadata_byte_size(doc),
            PINECONE_MAX_METADATA_BYTES,
            "500-recipient email must stay under Pinecone 40KB limit",
        )

    def test_huge_cc_list_under_pinecone_limit(self):
        cc = ", ".join(f'"CC Person {i}" <cc{i}@long-domain.example.com>' for i in range(500))
        email = _make_email(cc=cc)
        doc = email.to_document(doc_id="test_2")
        self.assertLess(
            _metadata_byte_size(doc),
            PINECONE_MAX_METADATA_BYTES,
        )

    def test_all_fields_maxed_under_pinecone_limit(self):
        """Absolute worst case: every metadata field is maximally large."""
        email = _make_email(
            sender="s" * 1000,
            subject="j" * 2000,
            recipients=", ".join(
                f'"Recipient {i}" <r{i}@long-domain.example.com>' for i in range(500)
            ),
            cc=", ".join(f'"CC {i}" <c{i}@long-domain.example.com>' for i in range(500)),
            source_id="p" * 5000,
        )
        doc = email.to_document(doc_id="test_3")
        size = _metadata_byte_size(doc)
        self.assertLess(
            size,
            PINECONE_MAX_METADATA_BYTES,
            f"All-fields-maxed metadata is {size} bytes, exceeds Pinecone limit",
        )


# ===================================================================
# 5. Excluded keys are set correctly on Document
# ===================================================================
class TestExcludedMetadataKeys(unittest.TestCase):
    """Ensure bulky keys are excluded from embedding/LLM context."""

    def test_excluded_embed_keys(self):
        doc = _make_email(recipients="a@b.com", cc="c@d.com").to_document(doc_id="test_0")
        for key in ("source_id", "to_full", "cc_full"):
            self.assertIn(key, doc.excluded_embed_metadata_keys)
            self.assertIn(key, doc.excluded_llm_metadata_keys)

    def test_non_excluded_keys_present(self):
        # sender/subject/date/source give useful embedding context and stay in;
        # recipients (to/cc) are now excluded from embed (kept in payload).
        doc = _make_email(recipients="a@b.com").to_document(doc_id="test_0")
        for key in ("sender", "subject", "date", "source"):
            self.assertNotIn(key, doc.excluded_embed_metadata_keys)


# ===================================================================
# 6. Edge cases that should not crash the pipeline
# ===================================================================
class TestEdgeCases(unittest.TestCase):
    """Emails with unusual or missing data must not raise exceptions."""

    def test_empty_body(self):
        doc = _make_email(body="").to_document(doc_id="empty_body")
        self.assertEqual(doc.text, "")

    def test_none_date(self):
        doc = _make_email(date=None).to_document(doc_id="no_date")
        self.assertEqual(doc.metadata["date"], "unknown")

    def test_no_recipients(self):
        email = _make_email(recipients=None, cc=None)
        doc = email.to_document(doc_id="no_recip")
        self.assertNotIn("to", doc.metadata)
        self.assertNotIn("to_full", doc.metadata)
        self.assertNotIn("cc", doc.metadata)
        self.assertNotIn("cc_full", doc.metadata)

    def test_non_ascii_sender(self):
        email = _make_email(sender="Ünïcödé Üser <unicode@example.com>")
        doc = email.to_document(doc_id="unicode")
        self.assertIn("Ünïcödé", doc.metadata["sender"])

    def test_unicode_subject(self):
        email = _make_email(subject="日本語の件名 📧 Ré: wichtig")
        doc = email.to_document(doc_id="unicode_subj")
        self.assertIn("日本語", doc.metadata["subject"])

    def test_very_long_source_id_path(self):
        long_path = "/".join(["folder"] * 100) + "/message.eml"
        email = _make_email(source_id=long_path)
        doc = email.to_document(doc_id="long_path")
        self.assertEqual(doc.metadata["source_id"], long_path)

    def test_empty_recipients_string(self):
        """Empty string recipients should not create to/to_full keys."""
        email = _make_email(recipients="")
        doc = email.to_document(doc_id="empty_recip")
        self.assertNotIn("to", doc.metadata)
        self.assertNotIn("to_full", doc.metadata)


if __name__ == "__main__":
    unittest.main()
