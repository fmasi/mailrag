"""Ingest attachments from .eml files into an AttachmentStore.

The first ingester for the source-agnostic store; IMAP/AppleScript ingesters will
feed the same store later. Walks MIME parts that are attachments/inline or carry a
filename, and tags each with the email's ``message_id``/``thread_id`` so retrieved
threads can join their files.

Identity (message_id, thread_id) is taken from ``MailArchiveXLoader`` — the *same*
parse the indexer uses — so the ``thread_id`` matches the value stored in the Qdrant
payload. (These exports prepend an mbox ``From `` line + a numeric field that breaks
a naive ``email`` parse, dropping the Message-ID; the loader strips that preamble.
Parsing raw here instead silently broke the attachment->thread join. See issue #32.)
"""

from __future__ import annotations

from email import message_from_bytes, policy
from email.header import decode_header, make_header
from typing import Dict, Iterable

from src.data.loaders.mail_archive_x import MailArchiveXLoader
from src.data.threading import compute_thread_id


def _decode_filename(raw: str | None) -> str:
    """Decode an RFC2047 encoded-word filename (``=?charset?B?...?=``) to text.

    Python's ``email`` does not auto-decode these, so ``part.get_filename()`` returns
    the raw header. Returns "" for a missing name. Never raises (falls back to the raw
    string on a malformed header)."""
    if not raw:
        return ""
    try:
        return _storable(str(make_header(decode_header(raw))))
    except Exception:
        return _storable(raw)


def _storable(text: str) -> str:
    """Text that can always be written to the store.

    A header is whatever the sender typed. An encoded-word such as
    ``=?utf-7?q?+2AA-?=`` decodes to a lone surrogate, which is a valid Python
    string and not valid UTF-8, so sqlite refuses to bind it."""
    return text.encode("utf-8", "replace").decode("utf-8")


def ingest_eml(paths: Iterable[str], store, *, progress: bool = False) -> Dict[str, int]:
    counts = {"emails": 0, "attachments": 0, "skipped": 0, "bad_parts": 0}
    paths = list(paths)
    bar = None
    if progress:
        try:
            from tqdm import tqdm

            bar = tqdm(total=len(paths), unit="eml", desc="attachments")
        except ImportError:
            bar = None
    for path in paths:
        # Identity from the loader (preamble-stripped, decoded) so thread_id matches
        # the indexer's exactly.
        try:
            loaded = MailArchiveXLoader(eml_files=[path], verbose=False).load()
        except Exception:
            loaded = []
        if not loaded:
            counts["skipped"] += 1
            if bar:
                bar.update(1)
            continue
        e = loaded[0]
        message_id = e.message_id or ""
        thread_id = compute_thread_id(
            message_id, e.in_reply_to or "", e.references or "", subject=e.subject or ""
        )
        counts["emails"] += 1

        # Parts from the same preamble-stripped bytes the loader parsed.
        try:
            with open(path, "rb") as fh:
                raw = MailArchiveXLoader._strip_mbox_preamble(fh.read())
            msg = message_from_bytes(raw, policy=policy.compat32)
        except Exception:
            if bar:
                bar.update(1)
            continue
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
            # Preserve the declared charset for text parts (text/plain; charset=...)
            # so extraction can decode them correctly instead of guessing.
            mime = part.get_content_type()
            charset = part.get_content_charset()
            if charset and mime.startswith("text/"):
                mime = f"{mime}; charset={charset}"
            try:
                store.put(
                    data,
                    message_id=message_id,
                    thread_id=thread_id,
                    filename=filename or "(unnamed)",
                    mime=_storable(mime),
                    size=len(data),
                    source_type="eml",
                    source_ref=path,
                    inline=(disp == "inline"),
                )
            except ValueError:
                # Something else in THIS message the store cannot represent (an
                # id with an unencodable character, say). It costs this part.
                # Raised, one crafted email would fail the whole ingest, and
                # under sync that means the whole account, on every run. A
                # failure of the store itself is an OSError or a sqlite error
                # and still propagates.
                counts["bad_parts"] += 1
                continue
            counts["attachments"] += 1
        if bar:
            bar.update(1)
    if bar:
        bar.close()
    return counts
