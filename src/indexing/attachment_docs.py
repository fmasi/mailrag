"""Turn email attachments into indexable LlamaIndex Documents (issue #80).

Attachment *bytes* are extracted on demand by the attachment subsystem (#40), but
nothing fed that text into the search index — so facts that live only inside an
``.xlsx``/``.pdf``/``.docx``/``.pptx`` (quotas, targets, contracts) were silently
unsearchable. This module closes that gap for the ``.eml`` ingest path.

For each ``.eml`` it walks the MIME parts, extracts each attachment's text with the
existing handler registry (``src/attachments/extract``), and emits a ``Document``
carrying:

* ``content_kind = "attachment"`` — so a chunk can be told apart from a body chunk;
* ``attachment_name`` — the (decoded) filename, so a hit names its source file;
* ``parent_message_id`` + ``thread_id`` — so a hit traces back to its email/thread.

The Documents are chunked **separately** from the (often 4-line) email body: the
caller appends them to the body Documents so the SentenceSplitter never mixes a
terse body with a 500-row sheet in one chunk.

Identity (``message_id`` / ``thread_id``) is taken from ``MailArchiveXLoader`` — the
same parse the indexer uses — so the ``thread_id`` on an attachment chunk matches
the value on the body chunks of the same email (see ``ingest_eml`` for the mbox
preamble gotcha this avoids).
"""

from __future__ import annotations

from email import message_from_bytes, policy
from typing import Iterable, List, Optional

from llama_index.core import Document

from src.attachments.extract import Status, build_default_extractor
from src.attachments.ingest_eml import _decode_filename
from src.data.loaders.mail_archive_x import MailArchiveXLoader
from src.data.models import _truncate
from src.data.threading import compute_thread_id
from src.indexing.attachment_chunking import DEFAULT_CHUNK_BUDGET, chunk_attachment

_MAX_ATTACH_NAME_LEN = 512

# Metadata keys that are payload-only filters and must not bloat the embedded text
# or the LLM context — mirrors NormalizedEmail.to_document()'s excluded set.
_EXCLUDED_KEYS = [
    "source_id",
    "thread_id",
    "message_id",
    "parent_message_id",
    "content_kind",
    "attachment_name",
    "chunk_index",
]


def _extract_texts_for_eml(path: str, extractor) -> List[tuple[str, str, str, str, bytes, str]]:
    """Return ``(filename, text, message_id, thread_id, data, mime)`` for every
    attachment in *path* that yielded non-empty text. The raw ``data`` bytes and
    ``mime`` are carried through so the structure-aware chunker can re-parse the
    attachment by its own format units (issue #89). Never raises — a malformed .eml
    yields []."""
    try:
        loaded = MailArchiveXLoader(eml_files=[path], verbose=False).load()
    except Exception:
        return []
    if not loaded:
        return []
    e = loaded[0]
    message_id = e.message_id or ""
    thread_id = compute_thread_id(
        message_id, e.in_reply_to or "", e.references or "", subject=e.subject or ""
    )

    try:
        with open(path, "rb") as fh:
            raw = MailArchiveXLoader._strip_mbox_preamble(fh.read())
        msg = message_from_bytes(raw, policy=policy.compat32)
    except Exception:
        return []

    out: List[tuple[str, str, str, str, bytes, str]] = []
    for part in msg.walk():
        if part.is_multipart():
            continue
        filename = _decode_filename(part.get_filename())
        disp = part.get_content_disposition() or ""
        if not filename and disp not in ("attachment", "inline"):
            continue
        try:
            data = part.get_payload(decode=True)
        except Exception:
            data = None
        if not data:
            continue
        mime = part.get_content_type()
        charset = part.get_content_charset()
        if charset and mime.startswith("text/"):
            mime = f"{mime}; charset={charset}"
        result = extractor.extract(data, mime, filename or "(unnamed)")
        if result.status == Status.EXTRACTED and result.text.strip():
            # ``data`` is bytes here (decode=True on a leaf part) — narrow it for the
            # typed tuple so the structure-aware chunker receives raw bytes.
            out.append(
                (filename or "(unnamed)", result.text, message_id, thread_id, bytes(data), mime)
            )
    return out


def build_attachment_documents(
    eml_paths: Iterable[str],
    *,
    extractor=None,
    extractor_name: Optional[str] = None,
    chunk_budget: int = DEFAULT_CHUNK_BUDGET,
    token_len=None,
) -> List[Document]:
    """Extract every attachment's text from *eml_paths* into LlamaIndex Documents,
    split structure-aware into embedder-sized chunks (issue #89).

    Each attachment is routed through :func:`chunk_attachment`, which splits it by
    its FORMAT's own units — spreadsheet row-groups (header repeated per chunk), PDF
    pages, PPTX slides, DOCX sections — with a prose-splitter fallback. Every emitted
    chunk becomes its own Document already sized under ``chunk_budget`` (a hard cap
    well below bge-m3's 8192-token limit), so the shared body ``SentenceSplitter`` in
    ``build_contextual_index`` leaves it intact instead of truncating a giant sheet.

    Parameters
    ----------
    eml_paths:
        The same ``.eml`` file list handed to the loader/indexer.
    extractor:
        An injectable ``Extractor`` (for tests). When None, the configured default
        extractor is built via ``build_default_extractor(extractor_name)``.
    extractor_name:
        Name of the default extractor/OCR backend to build when ``extractor`` is
        None. ``"tesseract"`` avoids constructing an LLM client for the OCR leg.
    chunk_budget:
        Hard per-chunk token cap (defaults to the profile ``chunk_size``).
    token_len:
        Injectable token counter passed through to :func:`chunk_attachment`.

    Returns
    -------
    list[Document]
        One Document per attachment *chunk*. Small attachments still yield a single
        Document. Body-diluting is avoided by keeping these documents separate from
        the email-body documents.
    """
    if extractor is None:
        extractor = build_default_extractor(extractor_name)

    docs: List[Document] = []
    for path in eml_paths:
        for filename, text, message_id, thread_id, data, mime in _extract_texts_for_eml(
            path, extractor
        ):
            base_meta: dict = {
                "content_kind": "attachment",
                "attachment_name": _truncate(filename, _MAX_ATTACH_NAME_LEN),
                "thread_id": thread_id,
                "source": "attachment",
                "source_id": path,
            }
            if message_id:
                base_meta["parent_message_id"] = message_id
            # A concise, human-readable subject line for the chunk so search results
            # are legible ("attachment: <name>") without embedding the long path.
            base_meta["subject"] = _truncate(f"attachment: {filename}", _MAX_ATTACH_NAME_LEN)

            chunks = chunk_attachment(
                data=data,
                text=text,
                mime=mime,
                filename=filename,
                budget=chunk_budget,
                token_len=token_len,
            )
            for idx, chunk in enumerate(chunks):
                metadata = dict(base_meta)
                if len(chunks) > 1:
                    # Tag the chunk's position so a hit names which slice of a large
                    # sheet/PDF it came from; single-chunk attachments stay untagged.
                    metadata["chunk_index"] = idx
                docs.append(
                    Document(
                        text=chunk,
                        metadata=metadata,
                        doc_id=f"attachment_{message_id or path}_{filename}_{idx}",
                        excluded_embed_metadata_keys=_EXCLUDED_KEYS,
                        excluded_llm_metadata_keys=_EXCLUDED_KEYS,
                    )
                )
    return docs
