"""IMAP source — covers iCloud, Fastmail, Dovecot, and most of the world (#101).

Notes that shaped this, from probing ``imap.mail.me.com`` and from prior art:

* **Capabilities must be re-read after login.** iCloud sits behind a frontend
  proxy that rewrites the pre-auth ``CAPABILITY`` response: CONDSTORE, QRESYNC,
  UIDPLUS and IDLE appear only once authenticated. Assuming the pre-auth list
  would permanently downgrade the connection.
* **CONDSTORE is an optimisation, never a requirement.** The UID watermark path
  is the primary mechanism and works unaided; MODSEQ is recorded alongside it for
  future flag/delete work. A server without CONDSTORE syncs identically.
* **One connection.** iCloud caps concurrent connections at roughly five and
  Mail.app on the same machine is already holding some. No pipelining, no
  parallel folder fetches.
* **``BODY.PEEK[]``, never ``BODY[]``.** Archiving your own mailbox must not mark
  it read.
* **Advance the cursor even when a message fails.** msgvault's one genuinely
  hard-won lesson: a single poison message that blocks the watermark wedges the
  folder forever.

``imapclient`` is imported lazily so the rest of mailrag stays importable without
it, and so a user who never syncs over IMAP need not install it.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Iterator, List, Optional

from src.sync.sources import (
    CURSOR_UID,
    CURSOR_UID_MODSEQ,
    Cursor,
    Folder,
    RawMessage,
    SourceCaps,
    resolve_role,
)

log = logging.getLogger(__name__)

# Messages fetched per round trip. Small enough that an interrupted run loses
# little, large enough that per-command latency does not dominate.
FETCH_BATCH = 50


class ImapError(RuntimeError):
    """An IMAP conversation failed in a way the runner should surface."""


def _decode(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return str(value)


class ImapSource:
    """Incremental IMAP reader driven by a per-folder UID watermark."""

    name = "imap"

    def __init__(
        self,
        *,
        host: str,
        login: str,
        password: str,
        port: int = 993,
        ssl: bool = True,
        folder_roles: Optional[dict] = None,
        start_from=None,
        client=None,
        timeout: int = 60,
    ):
        self.host = host
        self.login = login
        self._password = password
        self.port = port
        self.ssl = ssl
        self._folder_roles = folder_roles or {}
        self._start_from = start_from
        self._timeout = timeout
        self._client = client  # injectable for tests; otherwise connected lazily
        self._connected = client is not None
        self._caps: Optional[frozenset] = None
        self._selected: Optional[str] = None

    # ----------------------------------------------------------- connection

    def _connect(self):
        if not self._connected or self._client is None:
            try:
                from imapclient import IMAPClient  # noqa: PLC0415 — optional dependency
            except ImportError as exc:  # pragma: no cover - depends on the environment
                raise ImapError(
                    "IMAP sync needs the 'imapclient' package (pip install imapclient)"
                ) from exc
            try:
                self._client = IMAPClient(
                    self.host, port=self.port, ssl=self.ssl, timeout=self._timeout
                )
                self._client.login(self.login, self._password)
            except Exception as exc:  # noqa: BLE001 — auth and transport failures alike
                raise ImapError(f"IMAP login to {self.host} as {self.login} failed: {exc}") from exc
            self._connected = True
        # Negotiation is keyed off _caps rather than the connect branch, so an
        # injected client (tests, or a caller managing its own connection) is
        # negotiated exactly like one we dialled ourselves.
        if self._caps is None:
            self._negotiate()
        return self._client

    def _negotiate(self) -> None:
        """Re-read capabilities post-login and enable CONDSTORE when offered."""
        self._caps = self._capabilities()
        if "CONDSTORE" in self._caps:
            try:
                self._client.enable("CONDSTORE")
            except Exception as exc:  # noqa: BLE001 — an ENABLE refusal is not fatal
                log.warning("CONDSTORE advertised but ENABLE failed (%s); using UID only", exc)
                self._caps = frozenset(c for c in self._caps if c != "CONDSTORE")

    def _capabilities(self) -> frozenset:
        try:
            raw = self._client.capabilities()
        except Exception as exc:  # noqa: BLE001
            log.warning("could not read IMAP capabilities (%s); assuming a bare server", exc)
            return frozenset()
        return frozenset(_decode(c).upper() for c in raw or ())

    def capabilities(self) -> SourceCaps:
        self._connect()
        condstore = bool(self._caps and "CONDSTORE" in self._caps)
        return SourceCaps(
            name=self.name,
            cursor_kind=CURSOR_UID_MODSEQ if condstore else CURSOR_UID,
            incremental=True,
            # Detecting deletions needs a full UID-set diff, which the
            # archive-forever policy makes unnecessary. Reported honestly so a
            # future reconciliation sweep can ask.
            supports_delete_detection=False,
            max_connections=1,
        )

    # ---------------------------------------------------------------- folders

    def list_folders(self) -> List[Folder]:
        """List folders and classify them, without selecting any.

        Generations stay blank: obtaining one means SELECTing, and selecting
        every folder only to discard the out-of-scope ones is a wasted round trip
        each on a connection we are deliberately keeping to one.
        """
        client = self._connect()
        try:
            listed = client.list_folders()
        except Exception as exc:  # noqa: BLE001
            raise ImapError(f"could not list folders on {self.host}: {exc}") from exc
        out: List[Folder] = []
        for entry in listed:
            flags, _delimiter, name = entry
            name = _decode(name)
            if any(_decode(f).lower() == "\\noselect" for f in flags or ()):
                continue  # a container, not a mailbox
            role = resolve_role(
                name,
                flags=[_decode(f) for f in flags or ()],
                overrides=self._folder_roles,
            )
            out.append(Folder(name=name, role=role))
        return out

    def open_folder(self, folder: Folder) -> Folder:
        """SELECT the folder read-only and return it carrying its UIDVALIDITY.

        Read-only matters: syncing must never mutate the mailbox it is archiving.
        UIDVALIDITY is the generation — when it changes, every UID previously
        issued for this folder is meaningless and the cursor must be reset.
        """
        client = self._connect()
        try:
            info = client.select_folder(folder.name, readonly=True)
        except Exception as exc:  # noqa: BLE001
            raise ImapError(f"could not select folder {folder.name!r}: {exc}") from exc
        self._selected = folder.name
        generation = str(info.get(b"UIDVALIDITY") or info.get("UIDVALIDITY") or "")
        return Folder(name=folder.name, role=folder.role, generation=generation)

    def initial_cursor(self, folder: Folder) -> Cursor:
        """The starting watermark for a folder never synced before.

        With no ``start_from`` this is UID 0 — sync *is* the backfill and the
        whole folder gets downloaded. When ``start_from`` is set (because a
        backup export already covers history) the date is resolved to a UID here,
        so the first run fetches only what the export missed: ``UID SEARCH SINCE``
        gives the oldest message on/after that date, and the watermark is set
        just below it. Resolving server-side rather than filtering client-side is
        the point — otherwise the first run still downloads everything to throw
        most of it away.
        """
        kind = CURSOR_UID_MODSEQ if (self._caps and "CONDSTORE" in self._caps) else CURSOR_UID
        if self._start_from is None:
            return Cursor(kind, {"last_uid": 0})

        client = self._connect()
        if self._selected != folder.name:
            self.open_folder(folder)
        since = self._start_from.strftime("%d-%b-%Y")
        try:
            uids = [int(u) for u in client.search(["SINCE", since])]
        except Exception as exc:  # noqa: BLE001 — fall back to a full sync, never to silence
            log.warning(
                "SINCE search failed in %s (%s); starting from the beginning", folder.name, exc
            )
            return Cursor(kind, {"last_uid": 0})

        if uids:
            return Cursor(kind, {"last_uid": min(uids) - 1})
        # Nothing since that date. Park the watermark at the newest message so
        # the folder is considered caught up and only genuinely new mail arrives.
        try:
            existing = [int(u) for u in client.search(["ALL"])]
        except Exception:  # noqa: BLE001
            existing = []
        return Cursor(kind, {"last_uid": max(existing) if existing else 0})

    # ----------------------------------------------------------------- fetch

    def fetch_delta(self, folder: Folder, cursor: Cursor) -> Iterator[RawMessage]:
        """Yield messages with a UID above the watermark, oldest first."""
        client = self._connect()
        if self._selected != folder.name:
            self.open_folder(folder)

        last_uid = int(cursor.value.get("last_uid", 0) or 0)
        try:
            uids = client.search(["UID", f"{last_uid + 1}:*"])
        except Exception as exc:  # noqa: BLE001
            raise ImapError(f"UID search failed in {folder.name!r}: {exc}") from exc

        # `<n>:*` is defined to return the highest UID even when it is below n, so
        # an idle folder answers with its last message every run. Filtering here
        # keeps that from being re-fetched forever.
        uids = sorted(u for u in (int(x) for x in uids) if u > last_uid)
        if not uids:
            return

        for i in range(0, len(uids), FETCH_BATCH):
            batch = uids[i : i + FETCH_BATCH]
            try:
                resp = client.fetch(batch, ["BODY.PEEK[]", "INTERNALDATE"])
            except Exception as exc:  # noqa: BLE001
                raise ImapError(f"fetch failed in {folder.name!r}: {exc}") from exc
            for uid in batch:
                data = resp.get(uid)
                if not data:
                    continue
                raw = data.get(b"BODY[]") or data.get("BODY[]")
                if not raw:
                    continue
                yield RawMessage(
                    raw=bytes(raw),
                    source_uid=str(uid),
                    folder=folder.name,
                    internal_date=_as_utc(data.get(b"INTERNALDATE") or data.get("INTERNALDATE")),
                )

    def advance(self, cursor: Cursor, message: RawMessage) -> Cursor:
        """Move the watermark past *message*, never backwards."""
        try:
            uid = int(message.source_uid)
        except (TypeError, ValueError):
            return cursor
        value = dict(cursor.value)
        value["last_uid"] = max(int(value.get("last_uid", 0) or 0), uid)
        return Cursor(cursor.kind, value)

    def close(self) -> None:
        """Log out, tolerating an already-dead socket. Safe to call twice."""
        client, self._client, self._connected = self._client, None, False
        if client is None:
            return
        try:
            client.logout()
        except Exception:  # noqa: BLE001 — closing a broken connection is not an error
            pass


def _as_utc(value) -> Optional[datetime]:
    if not isinstance(value, datetime):
        return None
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
