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
CREATE TABLE IF NOT EXISTS blob_signals (
    sha256 TEXT PRIMARY KEY, chars INTEGER, words INTEGER, unique_words INTEGER,
    digits INTEGER, width INTEGER, height INTEGER, status TEXT, extractor TEXT,
    measured_at TEXT
);
CREATE TABLE IF NOT EXISTS text_cache (
    sha256 TEXT, extractor TEXT, text TEXT, status TEXT,
    extractor_used TEXT, created_at TEXT,
    PRIMARY KEY (sha256, extractor)
);
"""


# Boilerplate thresholds, calibrated on a real 45k-row corpus and verified by
# LOOKING at the images at every boundary (2026-08-19). 70% of attachment rows
# are recurring inline decoration — signature strips, newsletter headers,
# marketing templates, spacer pixels — which drown the actual documents.
#
# Three things that look like they should work, and do not:
#
# * Filename/mime. `image002.png` is a 259-byte spacer in one message and a
#   12 MB pasted screenshot in another. Both inline, both image/png.
# * Aspect ratio. "Banner-shaped means signature" is wrong: a 2475x383 strip in
#   this corpus is a product-lifecycle table with EOL dates.
# * Counting messages. A real image quoted down an 18-message reply chain
#   appears in 18 messages of ONE thread. A message-based rule read 237 blobs
#   (36% of everything it removed) as decoration that way — including a Samsung
#   feature-request table. Count distinct THREADS: decoration is reused by
#   unrelated conversations, quoted content is not.
#
# What remains is that recurrence alone still catches genuinely reused content —
# a benchmark table shared into 5 threads because it is useful. Size separates
# them: decoration at that recurrence is uniformly tiny, while the reused-content
# false positives were 60-93 KB. So the bar scales with size — a big image must
# be far more widely reused before it counts as decoration.
#
# Verified at the boundary: what this removes at the top end is a newsletter
# header in 52 threads and a marketing template in 25; what it now keeps is that
# benchmark table (88.8 KB, 5 threads). Costs 1 point of noise removal (64% ->
# 63%) over the size-blind rule. The rule errs toward keeping: leaving some
# decoration in beats hiding one real document.
BOILERPLATE_SMALL_MAX_SIZE = 20_000  # below this, 5 threads is enough evidence
BOILERPLATE_SMALL_MIN_THREADS = 5
BOILERPLATE_MAX_SIZE = 100_000  # nothing bigger is ever treated as decoration
BOILERPLATE_LARGE_MIN_THREADS = 15  # 20-100KB must be this widely reused

_NOT_BOILERPLATE = """
    NOT (a.inline = 1 AND a.mime LIKE 'image/%' AND a.size < ?
         AND (SELECT COUNT(DISTINCT b.thread_id) FROM attachments b
              WHERE b.sha256 = a.sha256)
             >= (CASE WHEN a.size < ? THEN ? ELSE ? END))
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

    def put_signals(self, sha256: str, signals) -> None:
        """Record measured signals for one blob (keyed by content hash).

        Keyed by sha256 because that is content identity: measuring the one
        6.5KB logo once covers all 2,273 messages that carry it. 45,454 rows in
        this corpus are only 6,761 distinct blobs.
        """
        self._conn.execute(
            """INSERT OR REPLACE INTO blob_signals
               (sha256, chars, words, unique_words, digits, width, height, status,
                extractor, measured_at)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (
                sha256,
                signals.chars,
                signals.words,
                signals.unique_words,
                signals.digits,
                signals.width,
                signals.height,
                signals.status,
                signals.extractor,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        self._conn.commit()

    def get_signals(self, sha256: str):
        """Measured signals for one blob, or ``None`` if never measured."""
        from src.attachments.signals import BlobSignals

        r = self._conn.execute(
            """SELECT chars, words, unique_words, digits, width, height, status, extractor
               FROM blob_signals WHERE sha256=?""",
            (sha256,),
        ).fetchone()
        if r is None:
            return None
        return BlobSignals(*r)

    def unmeasured_blobs(self, *, max_size: Optional[int] = None, images_only: bool = True):
        """Distinct blobs with no recorded signals yet — the classify work list.

        Returns ``(sha256, mime, filename, size)`` rows. ``max_size`` bounds the
        pass to the cheap tier: OCR on a 6KB logo is ~0.05s, while a 200-page
        PDF is minutes, so bulk measurement is worth it only for the small
        inline images that actually pollute listings.
        """
        sql = """SELECT a.sha256, a.mime, a.filename, MIN(a.size)
                 FROM attachments a
                 LEFT JOIN blob_signals s ON s.sha256 = a.sha256
                 WHERE s.sha256 IS NULL"""
        params: List[Any] = []
        if images_only:
            sql += " AND a.mime LIKE 'image/%'"
        if max_size is not None:
            sql += " AND a.size < ?"
            params.append(max_size)
        sql += " GROUP BY a.sha256"
        return self._conn.execute(sql, params).fetchall()

    def thread_counts(self) -> dict:
        """sha256 -> number of distinct threads carrying it (the recurrence signal)."""
        return {
            r[0]: r[1]
            for r in self._conn.execute(
                "SELECT sha256, COUNT(DISTINCT thread_id) FROM attachments GROUP BY sha256"
            )
        }

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
        rule described at :data:`BOILERPLATE_SMALL_MIN_THREADS`. One-off inline
        images (pasted screenshots) are kept, as are images reused only *within* one
        thread — quoting down a reply chain is not evidence of decoration.
        """
        where = "1=1"
        params: List[Any] = []
        if message_id is not None:
            where, params = "a.message_id=?", [message_id]
        elif thread_id is not None:
            where, params = "a.thread_id=?", [thread_id]
        rows = list(self._conn.execute(f"SELECT a.* FROM attachments a WHERE {where}", params))
        metas = [self._row_to_meta(r) for r in rows]
        if include_boilerplate:
            return metas
        return [m for m in metas if not self._is_boilerplate(m)]

    def _is_boilerplate(self, meta) -> bool:
        """Decide one attachment, preferring measured signals over the heuristic.

        Measured OCR signals win when they have an opinion, because the
        heuristic is a guess about content made from metadata and is known to
        misfire both ways — it hid a quarterly reporting-deadline table (small,
        quoted into six threads because it is useful) while a text-poor "access
        denied" screenshot is content someone pasted deliberately. Blobs that
        were never measured, or whose extraction failed, fall back to the
        heuristic, so coverage gaps degrade to today's behaviour rather than to
        no filtering at all.
        """
        from src.attachments.signals import is_decoration

        threads = self._thread_count(meta.sha256)
        verdict = is_decoration(self.get_signals(meta.sha256), threads, meta.inline)
        if verdict is not None:
            return verdict
        return self._heuristic_boilerplate(meta, threads)

    def _thread_count(self, sha256: str) -> int:
        return int(
            self._conn.execute(
                "SELECT COUNT(DISTINCT thread_id) FROM attachments WHERE sha256=?", (sha256,)
            ).fetchone()[0]
        )

    def _heuristic_boilerplate(self, meta, threads: int) -> bool:
        """Metadata-only fallback: recurrence across threads, scaled by size."""
        if not (meta.inline and meta.mime.startswith("image/")):
            return False
        if meta.size >= BOILERPLATE_MAX_SIZE:
            return False
        needed = (
            BOILERPLATE_SMALL_MIN_THREADS
            if meta.size < BOILERPLATE_SMALL_MAX_SIZE
            else BOILERPLATE_LARGE_MIN_THREADS
        )
        return threads >= needed

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
