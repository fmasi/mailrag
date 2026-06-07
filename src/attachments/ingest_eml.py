"""Ingest attachments from .eml files into an AttachmentStore.

The first ingester for the source-agnostic store; IMAP/AppleScript ingesters will
feed the same store later. Walks MIME parts that are attachments/inline or carry a
filename; computes message_id/thread_id so retrieved threads can join their files.
"""
from __future__ import annotations

from email import message_from_bytes, policy
from typing import Dict, Iterable

from src.data.threading import compute_thread_id


def _header(msg, name: str) -> str:
    return " ".join(str(msg.get(name) or "").split())


def ingest_eml(paths: Iterable[str], store, *, progress: bool = False) -> Dict[str, int]:
    counts = {"emails": 0, "attachments": 0, "skipped": 0}
    paths = list(paths)
    bar = None
    if progress:
        try:
            from tqdm import tqdm
            bar = tqdm(total=len(paths), unit="eml", desc="attachments")
        except ImportError:
            bar = None
    for path in paths:
        try:
            with open(path, "rb") as fh:
                msg = message_from_bytes(fh.read(), policy=policy.compat32)
        except Exception:
            counts["skipped"] += 1
            if bar:
                bar.update(1)
            continue
        counts["emails"] += 1
        message_id = _header(msg, "Message-ID")
        thread_id = compute_thread_id(message_id, _header(msg, "In-Reply-To"),
                                      _header(msg, "References"),
                                      subject=_header(msg, "Subject"))
        for part in msg.walk():
            if part.is_multipart():
                continue
            filename = part.get_filename()
            disp = (part.get_content_disposition() or "")
            if not filename and disp not in ("attachment", "inline"):
                continue
            try:
                data = part.get_payload(decode=True)
            except Exception:
                data = None
            if not data:
                continue
            store.put(data, message_id=message_id, thread_id=thread_id,
                      filename=filename or "(unnamed)",
                      mime=part.get_content_type(), size=len(data),
                      source_type="eml", source_ref=path,
                      inline=(disp == "inline"))
            counts["attachments"] += 1
        if bar:
            bar.update(1)
    if bar:
        bar.close()
    return counts
