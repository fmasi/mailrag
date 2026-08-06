"""Durable sync state: folder cursors, a per-message ledger, and run records.

Three tables, each earning its place:

``folders``
    Where each folder's cursor got to, plus the provider ``generation`` that
    cursor is valid under. A generation change (IMAP UIDVALIDITY, a recreated
    mailbox) invalidates the cursor and only the cursor — the ledger below makes
    the re-enumeration cost bandwidth rather than LLM calls.

``messages``
    Every message ever fetched, keyed on ``message_key`` — the *same* identity
    the indexer stamps on every Qdrant point — with a column per pipeline stage
    (``fetched`` / ``judged`` / ``indexed``). This is what lets a run be
    interrupted anywhere and resumed, and what lets a message that arrived while
    Qdrant was down get indexed on the next run instead of being lost.

    Keying on ``message_key`` rather than the content hash is deliberate.
    ``content_sha256`` intentionally excludes the Message-ID so that a re-export
    of the same email still hits the Pass-2 cache — which means two *different*
    emails with identical sender/subject/date/body (a newsletter sent twice, an
    automated alert) share it. Keying the ledger there would collapse them into
    one row, and the second would be spooled and indexed but never judged.

``sync_runs``
    Audit log and mid-run checkpoint in one (borrowed from msgvault, which is the
    part of its design worth taking). Starting a run supersedes any stale
    ``running`` row, so a crashed run cannot block the next one forever.

Deliberately mirrors :class:`src.llm.cache.Pass2Cache` in style — plain sqlite3,
no ORM, schema created on connect.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

from src.sync.sources import Cursor

_SCHEMA = """
CREATE TABLE IF NOT EXISTS folders (
    account_id   TEXT NOT NULL,
    name         TEXT NOT NULL,
    role         TEXT NOT NULL DEFAULT 'other',
    generation   TEXT NOT NULL DEFAULT '',
    cursor_kind  TEXT NOT NULL DEFAULT '',
    cursor_json  TEXT NOT NULL DEFAULT '{}',
    last_sync_at TEXT,
    PRIMARY KEY (account_id, name)
);

CREATE TABLE IF NOT EXISTS messages (
    account_id     TEXT NOT NULL,
    message_key    TEXT NOT NULL,
    content_sha256 TEXT NOT NULL DEFAULT '',
    message_id     TEXT,
    folder         TEXT NOT NULL DEFAULT '',
    source_uid     TEXT NOT NULL DEFAULT '',
    eml_path       TEXT NOT NULL DEFAULT '',
    internal_date  TEXT,
    fetched_at     TEXT NOT NULL,
    judged_at      TEXT,
    indexed_at     TEXT,
    error          TEXT,
    attempts       INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (account_id, message_key)
);
CREATE INDEX IF NOT EXISTS idx_messages_pending_judge
    ON messages(account_id, judged_at);
CREATE INDEX IF NOT EXISTS idx_messages_pending_index
    ON messages(account_id, indexed_at);
CREATE INDEX IF NOT EXISTS idx_messages_content ON messages(account_id, content_sha256);

CREATE TABLE IF NOT EXISTS poison (
    account_id  TEXT NOT NULL,
    folder      TEXT NOT NULL,
    source_uid  TEXT NOT NULL,
    error       TEXT NOT NULL DEFAULT '',
    seen_at     TEXT NOT NULL,
    PRIMARY KEY (account_id, folder, source_uid)
);

CREATE TABLE IF NOT EXISTS sync_runs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    account_id    TEXT NOT NULL,
    started_at    TEXT NOT NULL,
    completed_at  TEXT,
    status        TEXT NOT NULL,
    fetched       INTEGER NOT NULL DEFAULT 0,
    judged        INTEGER NOT NULL DEFAULT 0,
    indexed       INTEGER NOT NULL DEFAULT 0,
    errors        INTEGER NOT NULL DEFAULT 0,
    message       TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_runs_account ON sync_runs(account_id, started_at DESC);
"""

STATUS_RUNNING = "running"
STATUS_OK = "ok"
STATUS_PARTIAL = "partial"
STATUS_FAILED = "failed"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SyncState:
    """The sync ledger. One file per machine; safe to delete (costs a re-enumerate)."""

    def __init__(self, path: str):
        self._conn = sqlite3.connect(path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "SyncState":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ---------------------------------------------------------------- folders

    def get_cursor(self, account_id: str, folder: str) -> Tuple[Optional[Cursor], str]:
        """Return ``(cursor, generation)`` for a folder; ``(None, "")`` if unseen."""
        row = self._conn.execute(
            "SELECT cursor_kind, cursor_json, generation FROM folders "
            "WHERE account_id=? AND name=?",
            (account_id, folder),
        ).fetchone()
        if row is None:
            return None, ""
        if not row["cursor_kind"]:
            return None, row["generation"]
        cursor = Cursor(kind=row["cursor_kind"], value=json.loads(row["cursor_json"] or "{}"))
        return cursor, row["generation"]

    def set_cursor(
        self,
        account_id: str,
        folder: str,
        cursor: Cursor,
        *,
        generation: str = "",
        role: str = "other",
        touch: bool = True,
    ) -> None:
        """Persist a folder's cursor. Called after each spooled batch, not once at
        the end, so an interrupted run keeps the ground it gained."""
        self._conn.execute(
            """INSERT INTO folders
                   (account_id, name, role, generation, cursor_kind, cursor_json, last_sync_at)
               VALUES (?,?,?,?,?,?,?)
               ON CONFLICT(account_id, name) DO UPDATE SET
                   role=excluded.role,
                   generation=excluded.generation,
                   cursor_kind=excluded.cursor_kind,
                   cursor_json=excluded.cursor_json,
                   last_sync_at=COALESCE(excluded.last_sync_at, folders.last_sync_at)""",
            (
                account_id,
                folder,
                role,
                generation,
                cursor.kind,
                json.dumps(cursor.value, sort_keys=True),
                _now() if touch else None,
            ),
        )
        self._conn.commit()

    def reset_folder(self, account_id: str, folder: str, generation: str) -> None:
        """Void a folder's cursor after a generation change, keeping the ledger.

        Dropping the ledger too would be the expensive mistake: re-enumeration
        would then re-spool and re-judge everything. Keeping it means a
        UIDVALIDITY reset costs bandwidth alone.
        """
        self._conn.execute(
            """INSERT INTO folders (account_id, name, generation, cursor_kind, cursor_json)
               VALUES (?,?,?,'','{}')
               ON CONFLICT(account_id, name) DO UPDATE SET
                   generation=excluded.generation, cursor_kind='', cursor_json='{}'""",
            (account_id, folder, generation),
        )
        self._conn.commit()

    def folders(self, account_id: str) -> List[sqlite3.Row]:
        return list(
            self._conn.execute(
                "SELECT * FROM folders WHERE account_id=? ORDER BY name", (account_id,)
            )
        )

    # --------------------------------------------------------------- messages

    def have_message(self, account_id: str, message_key: str) -> bool:
        """Has this message already been spooled (in any folder)?

        Identity-keyed rather than UID-keyed, so the same message filed in two
        folders — or re-enumerated after a generation reset — is stored once.
        """
        return (
            self._conn.execute(
                "SELECT 1 FROM messages WHERE account_id=? AND message_key=?",
                (account_id, message_key),
            ).fetchone()
            is not None
        )

    def record_fetched(
        self,
        account_id: str,
        *,
        message_key: str,
        content_sha256: str = "",
        message_id: Optional[str] = None,
        folder: str = "",
        source_uid: str = "",
        eml_path: str = "",
        internal_date: Optional[str] = None,
    ) -> None:
        """Record a spooled message, or retarget an existing one to a new sighting.

        The retarget branch is how a message seen in a second folder (or moved)
        stays a single row instead of becoming a duplicate — the stage timestamps
        are deliberately left alone so re-sighting never re-triggers work.
        """
        self._conn.execute(
            """INSERT INTO messages
                   (account_id, message_key, content_sha256, message_id, folder,
                    source_uid, eml_path, internal_date, fetched_at)
               VALUES (?,?,?,?,?,?,?,?,?)
               ON CONFLICT(account_id, message_key) DO UPDATE SET
                   folder=excluded.folder, source_uid=excluded.source_uid""",
            (
                account_id,
                message_key,
                content_sha256,
                message_id,
                folder,
                source_uid,
                eml_path,
                internal_date,
                _now(),
            ),
        )
        self._conn.commit()

    def mark_judged(self, account_id: str, message_keys: Iterable[str]) -> int:
        return self._mark(account_id, message_keys, "judged_at")

    def mark_indexed(self, account_id: str, message_keys: Iterable[str]) -> int:
        return self._mark(account_id, message_keys, "indexed_at")

    def _mark(self, account_id: str, message_keys: Iterable[str], column: str) -> int:
        keys = [k for k in dict.fromkeys(message_keys) if k]
        if not keys:
            return 0
        now = _now()
        # Clearing `error` is the point of doing this in one statement: a message
        # that failed once and succeeded on retry must stop being counted, or the
        # error total only ever grows and stops meaning anything.
        self._conn.executemany(
            f"UPDATE messages SET {column}=?, error=NULL WHERE account_id=? AND message_key=?",
            [(now, account_id, k) for k in keys],
        )
        self._conn.commit()
        return len(keys)

    def record_poison(self, account_id: str, *, folder: str, source_uid: str, error: str) -> None:
        """Durably park a message that could not even be parsed.

        Its own table, deliberately. :meth:`record_error` cannot serve this case
        — it is an UPDATE keyed on ``message_key``, which an unparseable message
        does not have — but synthesising a key into the ``messages`` table would
        put a *sender-controlled* string in the same namespace as real keys: a
        crafted ``Message-ID`` could collide with a poison row and be silently
        suppressed. Keyed instead on the only identity such a message has, its
        location on the server.
        """
        self._conn.execute(
            """INSERT INTO poison (account_id, folder, source_uid, error, seen_at)
               VALUES (?,?,?,?,?)
               ON CONFLICT(account_id, folder, source_uid) DO UPDATE SET
                   error=excluded.error, seen_at=excluded.seen_at""",
            (account_id, folder, source_uid, error[:2000], _now()),
        )
        self._conn.commit()

    def poison_count(self, account_id: str) -> int:
        """How many messages are parked as unparseable."""
        row = self._conn.execute(
            "SELECT COUNT(*) AS n FROM poison WHERE account_id=?", (account_id,)
        ).fetchone()
        return row["n"] or 0

    def clear_indexed(self, account_id: str, message_keys: Iterable[str]) -> int:
        """Send messages back to the index queue. Returns how many were reset.

        ``indexed_at`` is otherwise write-once, which strands anything indexed
        before its summary existed: the vector stays summary-less forever even
        after a later run judges it. Judging therefore un-indexes, so the next
        index pass rewrites the point with its summary (cheap — deterministic
        ids make it a replacement, not a duplicate).
        """
        keys = [k for k in dict.fromkeys(message_keys) if k]
        if not keys:
            return 0
        cur = self._conn.executemany(
            "UPDATE messages SET indexed_at=NULL WHERE account_id=? AND message_key=? "
            "AND indexed_at IS NOT NULL",
            [(account_id, k) for k in keys],
        )
        self._conn.commit()
        return cur.rowcount if cur.rowcount and cur.rowcount > 0 else 0

    def indexed_keys(self, account_id: str, message_keys: Iterable[str]) -> List[str]:
        """Which of *message_keys* are already marked indexed."""
        keys = [k for k in dict.fromkeys(message_keys) if k]
        if not keys:
            return []
        marks = ",".join("?" * len(keys))
        rows = self._conn.execute(
            f"SELECT message_key FROM messages WHERE account_id=? AND indexed_at IS NOT NULL "
            f"AND message_key IN ({marks})",
            [account_id, *keys],
        ).fetchall()
        return [r["message_key"] for r in rows]

    def abandon(self, account_id: str, message_keys: Iterable[str], reason: str) -> int:
        """Give up on messages, terminally, with the reason recorded.

        Some failures are not transient and not retryable — the spooled ``.eml``
        has been deleted, or the message has failed the same stage too many
        times. Leaving those pending means re-loading them on every tick forever
        while ``--status`` reports a permanently dirty account and (because
        indexing waits on judging) blocks everything behind them.

        Both stage timestamps are set so the row leaves every queue, and ``error``
        is set so it is still counted and visible rather than silently passing as
        done.
        """
        keys = [k for k in dict.fromkeys(message_keys) if k]
        if not keys:
            return 0
        now = _now()
        self._conn.executemany(
            """UPDATE messages SET judged_at=COALESCE(judged_at, ?),
                   indexed_at=COALESCE(indexed_at, ?), error=?
               WHERE account_id=? AND message_key=?""",
            [(now, now, reason[:2000], account_id, k) for k in keys],
        )
        self._conn.commit()
        return len(keys)

    def bump_attempts(self, account_id: str, message_keys: Iterable[str]) -> Dict[str, int]:
        """Increment the failure counter for these messages; return the new counts.

        A deterministically-failing message otherwise re-issues a real (possibly
        paid) LLM call on every tick, forever, with nothing recording that it has
        already failed a hundred times.
        """
        keys = [k for k in dict.fromkeys(message_keys) if k]
        if not keys:
            return {}
        self._conn.executemany(
            "UPDATE messages SET attempts=attempts+1 WHERE account_id=? AND message_key=?",
            [(account_id, k) for k in keys],
        )
        self._conn.commit()
        marks = ",".join("?" * len(keys))
        rows = self._conn.execute(
            f"SELECT message_key, attempts FROM messages WHERE account_id=? "
            f"AND message_key IN ({marks})",
            [account_id, *keys],
        ).fetchall()
        return {r["message_key"]: r["attempts"] for r in rows}

    def last_successful_run(self, account_id: str) -> Optional[sqlite3.Row]:
        """The most recent run that actually completed cleanly.

        ``last_run`` answers "when did we last try", which is not the question
        ``--status`` is asking: 120 consecutive failed ticks would otherwise look
        fresh because the newest *attempt* is two hours old.
        """
        return self._conn.execute(
            "SELECT * FROM sync_runs WHERE account_id=? AND status=? ORDER BY id DESC LIMIT 1",
            (account_id, STATUS_OK),
        ).fetchone()

    def record_error(self, account_id: str, message_key: str, error: str) -> None:
        """Park a poison message with its error, without blocking the folder.

        The cursor still advances past it — one unparseable message must never
        wedge every later message behind it.
        """
        self._conn.execute(
            "UPDATE messages SET error=? WHERE account_id=? AND message_key=?",
            (error[:2000], account_id, message_key),
        )
        self._conn.commit()

    def pending(self, account_id: str, stage: str) -> List[sqlite3.Row]:
        """Messages spooled but not yet through *stage* (``judged``/``indexed``).

        The backbone of stage-skipping: mail fetched while LM Studio or Qdrant was
        down simply shows up here on the next run.
        """
        column = {"judged": "judged_at", "indexed": "indexed_at"}[stage]
        return list(
            self._conn.execute(
                f"SELECT * FROM messages WHERE account_id=? AND {column} IS NULL "
                "ORDER BY fetched_at",
                (account_id,),
            )
        )

    def counts(self, account_id: str) -> Dict[str, int]:
        row = self._conn.execute(
            """SELECT COUNT(*) AS total,
                      SUM(judged_at IS NULL) AS pending_judge,
                      SUM(indexed_at IS NULL) AS pending_index,
                      SUM(error IS NOT NULL) AS errors
               FROM messages WHERE account_id=?""",
            (account_id,),
        ).fetchone()
        return {
            "total": row["total"] or 0,
            "pending_judge": row["pending_judge"] or 0,
            "pending_index": row["pending_index"] or 0,
            # Poison messages never became ledger rows, so they must be counted
            # from their own table or --status under-reports the real trouble.
            "errors": (row["errors"] or 0) + self.poison_count(account_id),
        }

    # ------------------------------------------------------------- sync_runs

    def start_run(self, account_id: str) -> int:
        """Open a run, superseding any stale ``running`` row for this account.

        A run killed by a laptop sleeping, a `kill -9`, or a crash leaves its row
        ``running`` forever. Superseding on start (rather than trusting a
        cleanup-on-exit that by definition did not happen) is what keeps the next
        scheduled tick from being blocked by a ghost.
        """
        with self._conn:  # BEGIN/COMMIT: supersede + insert must be atomic
            self._conn.execute(
                """UPDATE sync_runs SET status=?, completed_at=?,
                       message='superseded by a later run'
                   WHERE account_id=? AND status=?""",
                (STATUS_FAILED, _now(), account_id, STATUS_RUNNING),
            )
            cur = self._conn.execute(
                "INSERT INTO sync_runs (account_id, started_at, status) VALUES (?,?,?)",
                (account_id, _now(), STATUS_RUNNING),
            )
        run_id = cur.lastrowid
        if run_id is None:  # pragma: no cover - sqlite always sets it after INSERT
            raise RuntimeError("could not open a sync run: sqlite returned no row id")
        return int(run_id)

    def finish_run(
        self,
        run_id: int,
        *,
        status: str = STATUS_OK,
        fetched: int = 0,
        judged: int = 0,
        indexed: int = 0,
        errors: int = 0,
        message: str = "",
    ) -> None:
        self._conn.execute(
            """UPDATE sync_runs
               SET completed_at=?, status=?, fetched=?, judged=?, indexed=?, errors=?, message=?
               WHERE id=?""",
            (_now(), status, fetched, judged, indexed, errors, message[:2000], run_id),
        )
        self._conn.commit()

    def last_run(self, account_id: str) -> Optional[sqlite3.Row]:
        return self._conn.execute(
            "SELECT * FROM sync_runs WHERE account_id=? ORDER BY id DESC LIMIT 1",
            (account_id,),
        ).fetchone()

    def recent_runs(self, account_id: str, limit: int = 10) -> List[sqlite3.Row]:
        return list(
            self._conn.execute(
                "SELECT * FROM sync_runs WHERE account_id=? ORDER BY id DESC LIMIT ?",
                (account_id, limit),
            )
        )

    def status(self, account_id: str) -> Dict[str, Any]:
        """Everything ``mailrag sync --status`` needs, in one call."""
        last = self.last_run(account_id)
        return {
            "account_id": account_id,
            "counts": self.counts(account_id),
            "folders": [dict(r) for r in self.folders(account_id)],
            "last_run": dict(last) if last is not None else None,
        }
