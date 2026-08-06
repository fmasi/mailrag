# src/llm/cache.py
"""Content-addressed SQLite cache for LLM Pass-2 per-email results.

Keyed by the sha256 of the raw ``.eml`` bytes (the same hash space as
:mod:`src.data.blacklist`), so a row survives file moves and is reused across
rebuilds. The heavy LLM sweep writes here once; dry-run, apply, and the build
all read from it.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from typing import Dict, Iterator, Optional

_SCHEMA = """
CREATE TABLE IF NOT EXISTS pass2 (
    sha256         TEXT PRIMARY KEY,
    summary        TEXT NOT NULL DEFAULT '',
    is_noise       INTEGER NOT NULL,
    confidence     REAL NOT NULL,
    reason         TEXT NOT NULL DEFAULT '',
    model          TEXT NOT NULL DEFAULT '',
    created_at     TEXT NOT NULL,
    message_id     TEXT,
    content_sha256 TEXT,
    quant          TEXT,
    endpoint       TEXT,
    source         TEXT
)
"""

# Stable fallback identifiers (see :mod:`src.data.identity`): the file-bytes
# ``sha256`` primary key is brittle against re-export, so we also index
# Message-ID and a normalized content hash and fall back to them in
# :meth:`Pass2Cache.get_resilient`.
_IDENTITY_COLUMNS = ("message_id", "content_sha256")

# Which judge produced a row. A model id alone does not identify one: the same
# model at a different quantisation, or served from a hosted endpoint rather than
# this machine, is a different judge — and a cache mixing them makes every
# noise-rate comparison over the corpus unattributable.
_PROVENANCE_COLUMNS = ("quant", "endpoint", "source")


class Pass2Cache:
    """A tiny keyed store for ``{summary, is_noise, confidence, reason}`` records."""

    def __init__(self, path: str):
        self._conn = sqlite3.connect(path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(_SCHEMA)
        self._migrate()
        self._conn.commit()

    def _migrate(self) -> None:
        """Add identity columns/indexes to a legacy DB, in place and idempotently."""
        existing = {r[1] for r in self._conn.execute("PRAGMA table_info(pass2)")}
        for col in _IDENTITY_COLUMNS:
            if col not in existing:
                self._conn.execute(f"ALTER TABLE pass2 ADD COLUMN {col} TEXT")
            self._conn.execute(f"CREATE INDEX IF NOT EXISTS idx_pass2_{col} ON pass2({col})")
        for col in _PROVENANCE_COLUMNS:
            if col not in existing:
                self._conn.execute(f"ALTER TABLE pass2 ADD COLUMN {col} TEXT")

    def close(self) -> None:
        self._conn.close()

    def has(self, sha256: str) -> bool:
        cur = self._conn.execute("SELECT 1 FROM pass2 WHERE sha256=?", (sha256,))
        return cur.fetchone() is not None

    def get(self, sha256: str) -> Optional[sqlite3.Row]:
        cur = self._conn.execute("SELECT * FROM pass2 WHERE sha256=?", (sha256,))
        return cur.fetchone()

    def put(
        self,
        sha256: str,
        record: Dict,
        model: str = "",
        message_id: Optional[str] = None,
        content_sha256: Optional[str] = None,
        quant: str = "",
        endpoint: str = "",
        source: str = "",
    ) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO pass2
               (sha256, summary, is_noise, confidence, reason, model, created_at,
                message_id, content_sha256, quant, endpoint, source)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                sha256,
                record.get("summary", ""),
                int(bool(record["is_noise"])),
                float(record.get("confidence", 0.0)),
                record.get("reason", ""),
                model,
                datetime.now(timezone.utc).isoformat(),
                message_id or None,
                content_sha256 or None,
                quant or None,
                endpoint or None,
                source or None,
            ),
        )
        self._conn.commit()

    def get_resilient(
        self, sha256: str, message_id: Optional[str] = None, content_sha256: Optional[str] = None
    ) -> Optional[sqlite3.Row]:
        """Look up a row by exact file hash, then by stable fallback identifiers.

        Order is deliberate: an exact file-bytes match is authoritative; only
        when that misses (e.g. a re-export changed the bytes) do we fall back to
        the Message-ID and then the normalized content hash.
        """
        row = self.get(sha256)
        if row is not None:
            return row
        if message_id:
            row = self._conn.execute(
                "SELECT * FROM pass2 WHERE message_id=?", (message_id,)
            ).fetchone()
            if row is not None:
                return row
        if content_sha256:
            row = self._conn.execute(
                "SELECT * FROM pass2 WHERE content_sha256=?", (content_sha256,)
            ).fetchone()
            if row is not None:
                return row
        return None

    def set_identity(
        self, sha256: str, message_id: Optional[str], content_sha256: Optional[str]
    ) -> None:
        """Backfill the stable identifiers on an existing row (no LLM cost)."""
        self._conn.execute(
            "UPDATE pass2 SET message_id=?, content_sha256=? WHERE sha256=?",
            (message_id or None, content_sha256 or None, sha256),
        )
        self._conn.commit()

    def iter_noise(self, min_confidence: float = 0.0) -> Iterator[sqlite3.Row]:
        yield from self._conn.execute(
            "SELECT * FROM pass2 WHERE is_noise=1 AND confidence>=? ORDER BY confidence DESC",
            (min_confidence,),
        )

    def iter_kept(self) -> Iterator[sqlite3.Row]:
        yield from self._conn.execute("SELECT * FROM pass2 WHERE is_noise=0")

    def judges(self) -> Dict[str, int]:
        """How many rows each distinct judge produced.

        The question this cache must be able to answer: is this corpus judged by
        ONE model at ONE quantisation, or silently by several?
        """
        rows = self._conn.execute(
            "SELECT COALESCE(model,'') m, COALESCE(quant,'') q, COALESCE(source,'') s, "
            "COUNT(*) n FROM pass2 GROUP BY m, q, s ORDER BY n DESC"
        ).fetchall()
        out: Dict[str, int] = {}
        for r in rows:
            key = r["m"] or "(unrecorded)"
            # The stored model may already be a `model@quant` fingerprint; only
            # append the quant when it is not already carried.
            if r["q"] and not key.endswith(f"@{r['q']}"):
                key += f"@{r['q']}"
            if r["s"]:
                key += f" [{r['s']}]"
            out[key] = out.get(key, 0) + r["n"]
        return out

    def stats(self) -> Dict[str, int]:
        row = self._conn.execute(
            "SELECT COUNT(*) AS total, COALESCE(SUM(is_noise),0) AS noise FROM pass2"
        ).fetchone()
        return {"total": row["total"], "noise": row["noise"], "kept": row["total"] - row["noise"]}
