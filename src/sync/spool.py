"""Write fetched messages to disk as ``.eml`` — the join between sync and the pipeline.

This is the whole integration trick. Every existing stage (folder selection, noise
filter, Pass-2 judging, attachment extraction, chunking, embedding) is driven by
``.eml`` files, so a syncer that materialises new mail as ``.eml`` files inside the
corpus tree needs no changes anywhere downstream: the delta simply becomes part of
the file universe ``resolve_index_files`` already walks.

Two properties matter:

**Identity must match the indexer's.** The spool derives a message's identity by
running the same :class:`MailArchiveXLoader` parse the indexer uses, rather than a
second, subtly different header parse. That is what keeps the sync ledger, the
Pass-2 cache and the Qdrant ``message_key`` all agreeing on what one email is —
see ``ingest_eml`` for the mbox-preamble variant of this bug.

**Writes must be atomic.** A run killed mid-write must never leave a truncated
``.eml`` that a later build would parse into a mangled email, so bytes land in a
temp file on the same filesystem and are then ``os.replace``d into place.
"""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
from dataclasses import dataclass
from typing import Optional

from src.data.identity import content_sha256
from src.data.loaders.mail_archive_x import MailArchiveXLoader

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]")
_MAX_STEM = 80


@dataclass(frozen=True)
class SpoolResult:
    """Where a message landed and how the rest of the system will identify it."""

    path: str
    content_sha256: str
    message_key: str
    message_id: Optional[str]
    is_new: bool


class SpoolError(RuntimeError):
    """A message could not be spooled (unparseable, or the disk refused it)."""


def _safe_stem(key: str) -> str:
    """A filesystem-safe, collision-free stem for a message key.

    The readable prefix makes the spool browsable; the hash suffix is what
    actually guarantees uniqueness once the prefix has been sanitised or
    truncated (two different Message-IDs can easily share their first 80
    sanitised characters).
    """
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]
    return f"{_UNSAFE.sub('_', key)[:_MAX_STEM]}-{digest}"


class Spool:
    """Append-only ``.eml`` store under ``<root>/<YYYY>/<MM>/``.

    Dated subdirectories keep any one directory small enough to list comfortably
    after years of mail, and make "what arrived in March" answerable with ``ls``.
    """

    def __init__(self, root: str):
        self.root = os.path.abspath(os.path.expanduser(root))
        self._tmp = os.path.join(self.root, ".tmp")
        os.makedirs(self._tmp, exist_ok=True)

    def write(self, raw: bytes) -> SpoolResult:
        """Spool *raw* RFC822 bytes; return where it went and how to identify it.

        Idempotent: a message already on disk is recognised by its destination
        path and reported with ``is_new=False`` rather than rewritten, so a
        re-enumeration after a UIDVALIDITY reset costs no writes.
        """
        if not raw:
            raise SpoolError("refusing to spool an empty message")

        fd, tmp_path = tempfile.mkstemp(dir=self._tmp, suffix=".eml")
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(raw)
            # Parse with the SAME loader the indexer uses, so identity agrees.
            try:
                loaded = MailArchiveXLoader(eml_files=[tmp_path], verbose=False).load()
            except Exception as exc:  # noqa: BLE001 — malformed mail is data, not a bug
                raise SpoolError(f"could not parse message: {exc}") from exc
            if not loaded:
                raise SpoolError("message parsed to nothing")
            email = loaded[0]

            key = email.message_key()
            digest = content_sha256(
                sender=email.sender, subject=email.subject, date=email.date, body=email.body
            )
            dest = self._destination(email, key)
            if os.path.exists(dest):
                return SpoolResult(dest, digest, key, email.message_id, is_new=False)

            os.makedirs(os.path.dirname(dest), exist_ok=True)
            os.replace(tmp_path, dest)  # atomic within the filesystem
            tmp_path = ""  # consumed by the rename
            return SpoolResult(dest, digest, key, email.message_id, is_new=True)
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def _destination(self, email, key: str) -> str:
        """``<root>/<YYYY>/<MM>/<stem>.eml``, or ``undated/`` when the date is unusable.

        Undated mail is bucketed rather than guessed into "now": a wrong date
        would scatter re-spooled copies of the same message across directories.
        """
        date = getattr(email, "date", None)
        if date is not None:
            try:
                sub = os.path.join(f"{date.year:04d}", f"{date.month:02d}")
            except (AttributeError, ValueError):
                sub = "undated"
        else:
            sub = "undated"
        return os.path.join(self.root, sub, f"{_safe_stem(key)}.eml")
