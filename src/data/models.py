"""Shared data models for email ingestion."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from email.utils import getaddresses
from typing import Any, Dict, Optional

from llama_index.core import Document

from .threading import compute_thread_id

# Capped in CHARACTERS while the chunker budgets in TOKENS. For token-dense
# scripts (CJK, emoji) one character can exceed one token, so a 512-char subject
# could tokenise past a 512-token chunk budget and make the splitter reject the
# document outright. Halved to leave headroom; `_split_documents` quarantines
# anything that still slips through rather than failing the batch.
_MAX_SENDER_LEN = 128
_MAX_SUBJECT_LEN = 256
_MAX_RECIPIENT_PREVIEW_LEN = 512
_MAX_RECIPIENT_FULL_LEN = (
    8192  # Keep some full data but stay inside the 40KB per-vector payload budget
)
_MAX_SUMMARY_LEN = 1024
# Identity/threading headers are sender-supplied and were the only metadata that
# escaped truncation entirely — a 1 MB Message-ID produced ~4 MB of payload PER
# CHUNK, and one such email in a 256-point batch pushes a single Qdrant request
# past its 32 MB request cap. Generous enough that no legitimate header is
# touched (RFC 5322 recommends well under 1000 chars).
_MAX_ID_LEN = 998


@dataclass
class NormalizedEmail:
    """Unified email representation for any source."""

    sender: str
    subject: str
    date: Optional[datetime]
    body: str
    source: str
    source_id: str
    recipients: Optional[str] = None
    cc: Optional[str] = None
    message_id: Optional[str] = None
    in_reply_to: Optional[str] = None
    references: Optional[str] = None
    summary: Optional[str] = None  # LLM Pass-2 enrichment; payload-only
    is_bulk: bool = False  # bulk-mail header marker (List-Unsubscribe / Precedence:bulk)
    noise_candidate: bool = False  # bulk rule would drop it; kept for the no-LLM vector hunt

    # Example usage:
    # email = NormalizedEmail(
    #     sender="alice@example.com",
    #     subject="Hello",
    #     date=None,
    #     body="Hi there",
    #     source="enron",
    #     source_id="enron_0",
    # )
    # doc = email.to_document(doc_id="enron_0")

    def message_key(self) -> str:
        """Stable identity for this email: normalized Message-ID, else content hash.

        Delegates to :func:`src.data.identity.message_key` so the indexer, the
        Pass-2 cache and (later) the sync ledger all agree on what "the same
        email" means across re-exports and ingest paths.
        """
        from .identity import message_key as _message_key

        return _message_key(
            sender=self.sender,
            subject=self.subject,
            date=self.date,
            body=self.body,
            message_id=self.message_id or "",
        )

    def to_document(self, doc_id: Optional[str] = None) -> Document:
        """Convert to a LlamaIndex Document with consistent metadata.

        ``doc_id`` defaults to ``body:<message_key>`` — a *stable* key, unlike the
        positional ``<source>_<i>`` ids callers used to pass. Deterministic point
        ids are derived from it (see ``src/indexing/point_ids.py``), so passing a
        run-dependent id here reintroduces duplicate chunks on re-index.
        """
        key = self.message_key()
        if doc_id is None:
            doc_id = f"body:{key}"
        # Values are not all strings: the bulk markers below are booleans, so the
        # payload is heterogeneous by design.
        metadata: Dict[str, Any] = {
            # Stable per-email identity, indexed as a Qdrant payload keyword so an
            # incremental run can delete exactly this email's chunks (body AND
            # attachments, which carry the same key) before re-upserting them.
            "message_key": _truncate(key, _MAX_ID_LEN),
            "sender": _truncate(self.sender, _MAX_SENDER_LEN),
            "subject": _truncate(self.subject, _MAX_SUBJECT_LEN),
            "date": self.date.isoformat() if self.date else "unknown",
            "source": self.source,
            "source_id": self.source_id,
        }
        if self.recipients:
            metadata["to"] = _truncate(
                _addresses_preview(self.recipients), _MAX_RECIPIENT_PREVIEW_LEN
            )
            metadata["to_full"] = _truncate(self.recipients, _MAX_RECIPIENT_FULL_LEN)
        if self.cc:
            metadata["cc"] = _truncate(_addresses_preview(self.cc), _MAX_RECIPIENT_PREVIEW_LEN)
            metadata["cc_full"] = _truncate(self.cc, _MAX_RECIPIENT_FULL_LEN)

        # Thread linkage (RFC 5322). thread_id groups an entire conversation and
        # is always present so it can be used as a vector-store payload filter;
        # the raw ids are kept for debugging / parent-thread lookup.
        # subject= is passed so that header-less datasets (e.g. the public HF
        # Enron corpus, which carries no Message-ID/In-Reply-To/References)
        # fall back to a "subj:<slug>" key instead of persisting an empty string
        # that breaks thread-aware retrieval.
        metadata["thread_id"] = _truncate(
            compute_thread_id(
                self.message_id or "",
                self.in_reply_to or "",
                self.references or "",
                subject=self.subject or "",
            ),
            _MAX_ID_LEN,
        )
        if self.message_id:
            metadata["message_id"] = _truncate(self.message_id, _MAX_ID_LEN)
        if self.in_reply_to:
            metadata["in_reply_to"] = _truncate(self.in_reply_to, _MAX_ID_LEN)
        if self.references:
            metadata["references"] = _truncate(self.references, _MAX_RECIPIENT_FULL_LEN)

        if self.summary:
            metadata["summary"] = _truncate(self.summary, _MAX_SUMMARY_LEN)

        # Bulk markers — payload-only filters. is_bulk is the raw header signal;
        # noise_candidate flags mail the conservative Pass-1 bulk rule would have
        # dropped but we kept, so the no-LLM vector hunt can inspect it first.
        metadata["is_bulk"] = self.is_bulk
        metadata["noise_candidate"] = self.noise_candidate

        # Keep full recipient headers and thread ids in metadata for
        # filtering/debugging, but exclude them from chunking/embedding context
        # so they don't bloat node size or change the vectors.
        # Recipient lists (to/cc and their *_full variants) and the thread ids
        # are kept in the payload for filtering ("emails involving person X",
        # "this thread") but excluded from the embedded text: they add no
        # semantic value and their length can exceed the chunk size.
        excluded_keys = [
            "source_id",
            "message_key",
            "to",
            "to_full",
            "cc",
            "cc_full",
            "thread_id",
            "message_id",
            "in_reply_to",
            "references",
            "summary",
            "is_bulk",
            "noise_candidate",
        ]
        return Document(
            text=self.body,
            metadata=metadata,
            doc_id=doc_id,
            excluded_embed_metadata_keys=excluded_keys,
            excluded_llm_metadata_keys=excluded_keys,
        )


def _truncate(value: str, max_len: int) -> str:
    """Truncate metadata fields to a safe size for node parsing."""
    if len(value) <= max_len:
        return value
    return value[: max_len - 3] + "..."


def _addresses_preview(raw_header_value: str) -> str:
    """Return a concise, searchable preview of addresses from a header."""
    addresses = [addr for _, addr in getaddresses([raw_header_value]) if addr]
    if not addresses:
        return raw_header_value
    # Deduplicate while preserving order.
    deduped = list(dict.fromkeys(addresses))
    return ", ".join(deduped)


def normalize_enron_record(record: Dict[str, Any], index: int) -> NormalizedEmail:
    """Convert an Enron dataset record into a NormalizedEmail."""
    # Example usage:
    # normalized = normalize_enron_record({"email": "From: a\nSubject: hi\n\nBody"}, 0)
    email_text = record.get("email", "")

    sender = _extract_email_field(email_text, "From")
    subject = _extract_email_field(email_text, "Subject")
    date_str = _extract_email_field(email_text, "Date")
    body = _extract_email_body(email_text)

    date_obj = None
    if date_str:
        try:
            from email.utils import parsedate_to_datetime

            date_obj = parsedate_to_datetime(date_str)
        except Exception:
            date_obj = None

    return NormalizedEmail(
        sender=sender or "unknown",
        subject=subject or "(no subject)",
        date=date_obj,
        body=body or email_text,
        source="enron",
        source_id=f"enron_{index}",
    )


def _extract_email_field(email_text: str, field_name: str) -> str:
    """Extract a specific field from email headers."""
    # Example usage:
    # sender = _extract_email_field("From: alice@example.com\n\nBody", "From")
    lines = email_text.split("\n")
    for line in lines:
        if line.startswith(field_name + ":"):
            return line.replace(field_name + ":", "", 1).strip()
    return ""


def _extract_email_body(email_text: str) -> str:
    """Extract the body of the email (content after headers)."""
    # Example usage:
    # body = _extract_email_body("From: a\nSubject: hi\n\nHello there")
    if "\n\n" in email_text:
        body = email_text.split("\n\n", 1)[1]
        return body.strip()
    return email_text
