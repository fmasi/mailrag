"""Source-agnostic, content-addressed attachment store.

Bytes live at ``<root>/blobs/{first-2}/{sha256}`` (write-once, dedup); metadata in
a sqlite index; extracted text cached by sha. Any ingester (.eml now; IMAP/AppleScript
later) feeds the same store.
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, List, Optional

from src.attachments.extract import (
    ExtractResult,
    Status,
    build_default_extractor,
    default_extractor_name,
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS attachments (
    sha256 TEXT, message_id TEXT, thread_id TEXT, filename TEXT, mime TEXT,
    size INTEGER, source_type TEXT, source_ref TEXT, inline INTEGER, created_at TEXT,
    UNIQUE(message_id, sha256)
);
CREATE INDEX IF NOT EXISTS idx_att_msg ON attachments(message_id);
CREATE INDEX IF NOT EXISTS idx_att_thread ON attachments(thread_id);
CREATE INDEX IF NOT EXISTS idx_att_sha ON attachments(sha256);
CREATE TABLE IF NOT EXISTS text_cache (
    sha256 TEXT, extractor TEXT, text TEXT, status TEXT,
    extractor_used TEXT, created_at TEXT,
    PRIMARY KEY (sha256, extractor)
);
"""


# Boilerplate thresholds, calibrated on a real 45k-row corpus (2026-08-19).
# 73% of attachment rows are recurring inline images — signature logos, spacer
# pixels, footer badges — which drown the actual documents in any listing.
#
# Neither filename nor mime can separate them from real content: `image002.png`
# is a 259-byte spacer in one message and a 12 MB pasted screenshot in another,
# and both are `image/png` marked inline. What DOES separate them is recurrence:
# decoration is reused across many messages, content is not. Requiring BOTH a
# small size and reuse across several messages keeps every one-off screenshot
# (2,642 distinct images in that corpus) while removing 73% of the rows.
BOILERPLATE_MIN_MESSAGES = 5
BOILERPLATE_MAX_SIZE = 100_000

_NOT_BOILERPLATE = """
    NOT (a.inline = 1 AND a.mime LIKE 'image/%' AND a.size < ?
         AND (SELECT COUNT(DISTINCT b.message_id) FROM attachments b
              WHERE b.sha256 = a.sha256) >= ?)
"""


@dataclass
class AttachmentMeta:
    sha256: str
    message_id: str
    thread_id: str
    filename: str
    mime: str
    size: int
    source_type: str
    source_ref: str
    inline: bool


class AttachmentStore:
    def __init__(self, root: str):
        self.root = os.path.abspath(os.path.expanduser(root))
        self._blobs = os.path.join(self.root, "blobs")
        os.makedirs(self._blobs, exist_ok=True)
        self._conn = sqlite3.connect(os.path.join(self.root, "index.db"))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        # Migrate a legacy text_cache (pre composite-key) — it is a disposable cache.
        cols = {r["name"] for r in self._conn.execute("PRAGMA table_info(text_cache)")}
        if "extractor" not in cols or "extractor_used" not in cols:
            self._conn.execute("DROP TABLE IF EXISTS text_cache")
            self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def count(self) -> int:
        """Total attachment rows in the store (0 = never ingested).

        Exists so callers can tell "this thread has no attachments" apart from
        "no attachments have ever been ingested". The store is populated only by
        ``mailrag attachments build``; indexing and continuous sync extract
        attachment *text* for retrieval down a separate path
        (``src.indexing.attachment_docs``) and never write here. So a corpus can
        be fully indexed, with attachment content searchable, while this store
        is still empty — and every lookup then returns an empty list that looks
        exactly like a thread with no attachments.
        """
        return int(self._conn.execute("SELECT COUNT(*) FROM attachments").fetchone()[0])

    def path_for(self, sha256: str) -> str:
        return os.path.join(self._blobs, sha256[:2], sha256)

    def put(
        self,
        data: bytes,
        *,
        message_id: str,
        thread_id: str,
        filename: str,
        mime: str,
        size: int,
        source_type: str,
        source_ref: str,
        inline: bool = False,
    ) -> str:
        sha = hashlib.sha256(data).hexdigest()
        blob = self.path_for(sha)
        if not os.path.exists(blob):
            os.makedirs(os.path.dirname(blob), exist_ok=True)
            with open(blob, "wb") as fh:
                fh.write(data)
        self._conn.execute(
            """INSERT OR IGNORE INTO attachments
               (sha256, message_id, thread_id, filename, mime, size, source_type,
                source_ref, inline, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (
                sha,
                message_id or "",
                thread_id or "",
                filename or "",
                mime or "",
                int(size),
                source_type,
                source_ref,
                int(bool(inline)),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        self._conn.commit()
        return sha

    def _row_to_meta(self, r: sqlite3.Row) -> AttachmentMeta:
        return AttachmentMeta(
            sha256=r["sha256"],
            message_id=r["message_id"],
            thread_id=r["thread_id"],
            filename=r["filename"],
            mime=r["mime"],
            size=r["size"],
            source_type=r["source_type"],
            source_ref=r["source_ref"],
            inline=bool(r["inline"]),
        )

    def list_for(
        self,
        *,
        message_id: Optional[str] = None,
        thread_id: Optional[str] = None,
        include_boilerplate: bool = True,
    ) -> List[AttachmentMeta]:
        """Attachments for a message or thread (or the whole store).

        ``include_boilerplate=False`` drops recurring small inline images —
        signature logos, spacer pixels, footer badges — using the recurrence
        rule described at :data:`BOILERPLATE_MIN_MESSAGES`. One-off inline
        images (pasted screenshots) are kept: they are content, and only their
        reuse across messages distinguishes decoration from them.
        """
        where = "1=1"
        params: List[Any] = []
        if message_id is not None:
            where, params = "a.message_id=?", [message_id]
        elif thread_id is not None:
            where, params = "a.thread_id=?", [thread_id]
        if not include_boilerplate:
            where += " AND " + _NOT_BOILERPLATE
            params += [BOILERPLATE_MAX_SIZE, BOILERPLATE_MIN_MESSAGES]
        rows = self._conn.execute(f"SELECT a.* FROM attachments a WHERE {where}", params)
        return [self._row_to_meta(r) for r in rows]

    def get_bytes(self, sha256: str) -> bytes:
        blob = self.path_for(sha256)
        if not os.path.exists(blob):
            raise KeyError(f"no attachment blob for {sha256}")
        with open(blob, "rb") as fh:
            return fh.read()

    def _meta_row(self, sha256: str):
        return self._conn.execute(
            "SELECT * FROM attachments WHERE sha256=? LIMIT 1", (sha256,)
        ).fetchone()

    def _cached_text(self, sha256: str, name: str) -> Optional[ExtractResult]:
        """The cached result for (sha, extractor name), reporting the extractor that
        actually produced the text (extractor_used) — same answer as the original
        extraction. None on a cache miss."""
        r = self._conn.execute(
            "SELECT text, status, extractor_used FROM text_cache WHERE sha256=? AND extractor=?",
            (sha256, name),
        ).fetchone()
        if r is None:
            return None
        return ExtractResult(
            text=r["text"], status=r["status"], extractor=r["extractor_used"] or name
        )

    def _extract_and_cache(self, sha256: str, name: str, row) -> ExtractResult:
        result = build_default_extractor(name).extract(
            self.get_bytes(sha256), row["mime"], row["filename"]
        )
        if result.status in (Status.OCR_UNAVAILABLE, Status.BINARY):
            # ENVIRONMENT verdicts, not facts about the attachment: OCR_UNAVAILABLE
            # means tesseract was missing from PATH, BINARY means a parsing library
            # was not installed. Caching either freezes the failure — a later run
            # with a working environment reads the cache and never retries (GH #37,
            # re-triggered by scheduled runs, which inherit no PATH).
            return result
        self._conn.execute(
            """INSERT OR REPLACE INTO text_cache
               (sha256, extractor, text, status, extractor_used, created_at)
               VALUES (?,?,?,?,?,?)""",
            (
                sha256,
                name,
                result.text,
                result.status,
                result.extractor,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        self._conn.commit()
        return result

    def get_text(
        self, sha256: str, *, extractor: Optional[str] = None, force: bool = False
    ) -> ExtractResult:
        name = extractor or default_extractor_name()
        if not force:
            cached = self._cached_text(sha256, name)
            if cached is not None:
                return cached
        row = self._meta_row(sha256)
        if row is None:
            raise KeyError(f"unknown attachment {sha256}")
        return self._extract_and_cache(sha256, name, row)

    def fetch(self, sha256: str, *, extractor: Optional[str] = None, force: bool = False) -> dict:
        row = self._meta_row(sha256)
        if row is None:
            raise KeyError(f"unknown attachment {sha256}")
        name = extractor or default_extractor_name()
        result = None if force else self._cached_text(sha256, name)
        if result is None:
            result = self._extract_and_cache(sha256, name, row)
        return {
            "sha256": sha256,
            "filename": row["filename"],
            "mime": row["mime"],
            "size": row["size"],
            "text": result.text,
            "text_status": result.status,
            "path": self.path_for(sha256),
        }
