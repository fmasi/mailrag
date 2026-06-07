"""Source-agnostic, content-addressed attachment store.

Bytes live at ``<root>/blobs/{first-2}/{sha256}`` (write-once, dedup); metadata in
a sqlite index; extracted text cached by sha. Any ingester (.eml now; IMAP/AppleScript
later) feeds the same store. See docs/superpowers/specs/2026-06-07-attachments-1a-store-fetch-design.md.
"""
from __future__ import annotations

import hashlib
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional

from src.attachments.extract import extract_text, ExtractResult

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
    sha256 TEXT PRIMARY KEY, text TEXT, extractor TEXT, status TEXT, created_at TEXT
);
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
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def path_for(self, sha256: str) -> str:
        return os.path.join(self._blobs, sha256[:2], sha256)

    def put(self, data: bytes, *, message_id: str, thread_id: str, filename: str,
            mime: str, size: int, source_type: str, source_ref: str,
            inline: bool = False) -> str:
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
            (sha, message_id or "", thread_id or "", filename or "", mime or "",
             int(size), source_type, source_ref, int(bool(inline)),
             datetime.now(timezone.utc).isoformat()))
        self._conn.commit()
        return sha

    def _row_to_meta(self, r: sqlite3.Row) -> AttachmentMeta:
        return AttachmentMeta(
            sha256=r["sha256"], message_id=r["message_id"], thread_id=r["thread_id"],
            filename=r["filename"], mime=r["mime"], size=r["size"],
            source_type=r["source_type"], source_ref=r["source_ref"],
            inline=bool(r["inline"]))

    def list_for(self, *, message_id: Optional[str] = None,
                 thread_id: Optional[str] = None) -> List[AttachmentMeta]:
        if message_id is not None:
            rows = self._conn.execute(
                "SELECT * FROM attachments WHERE message_id=?", (message_id,))
        elif thread_id is not None:
            rows = self._conn.execute(
                "SELECT * FROM attachments WHERE thread_id=?", (thread_id,))
        else:
            rows = self._conn.execute("SELECT * FROM attachments")
        return [self._row_to_meta(r) for r in rows]

    def get_bytes(self, sha256: str) -> bytes:
        blob = self.path_for(sha256)
        if not os.path.exists(blob):
            raise KeyError(f"no attachment blob for {sha256}")
        with open(blob, "rb") as fh:
            return fh.read()

    def _meta_row(self, sha256: str):
        return self._conn.execute(
            "SELECT * FROM attachments WHERE sha256=? LIMIT 1", (sha256,)).fetchone()

    def get_text(self, sha256: str) -> ExtractResult:
        cached = self._conn.execute(
            "SELECT text, extractor, status FROM text_cache WHERE sha256=?",
            (sha256,)).fetchone()
        if cached is not None:
            return ExtractResult(text=cached["text"], status=cached["status"],
                                 extractor=cached["extractor"])
        row = self._meta_row(sha256)
        if row is None:
            raise KeyError(f"unknown attachment {sha256}")
        result = extract_text(self.get_bytes(sha256), row["mime"], row["filename"])
        self._conn.execute(
            """INSERT OR REPLACE INTO text_cache (sha256, text, extractor, status, created_at)
               VALUES (?,?,?,?,?)""",
            (sha256, result.text, result.extractor, result.status,
             datetime.now(timezone.utc).isoformat()))
        self._conn.commit()
        return result

    def fetch(self, sha256: str) -> dict:
        row = self._meta_row(sha256)
        if row is None:
            raise KeyError(f"unknown attachment {sha256}")
        result = self.get_text(sha256)
        return {
            "sha256": sha256, "filename": row["filename"], "mime": row["mime"],
            "size": row["size"], "text": result.text, "text_status": result.status,
            "path": self.path_for(sha256),
        }
