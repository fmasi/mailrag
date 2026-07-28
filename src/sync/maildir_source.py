"""A local Maildir source — the network-free reference implementation (issue #101).

Two jobs. It is genuinely useful (mbsync/offlineimap/getmail users already have a
local Maildir, and syncing from it needs no credentials and no rate limits), and
it makes the entire runner testable end to end with nothing but a temp directory —
no fake IMAP server, no network, no secrets. Every property the IMAP path relies
on — cursors advancing, resumption after a crash, dedup across folders, stage
skipping — is exercised here first.

Cursor is a modification-time watermark. Maildir filenames are unique and files
are never modified in place, so "everything whose mtime is newer than the last one
I handled" is a sound delta. The tie-breaking filename in the cursor prevents a
message written in the same mtime tick as the watermark from being skipped.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Iterator, List

from src.sync.sources import (
    CURSOR_MTIME,
    Cursor,
    Folder,
    RawMessage,
    SourceCaps,
    resolve_role,
)

_MAILDIR_SUBDIRS = ("cur", "new")


def _is_maildir(path: str) -> bool:
    return all(os.path.isdir(os.path.join(path, sub)) for sub in _MAILDIR_SUBDIRS)


class MaildirSource:
    """Read new mail out of a Maildir tree."""

    name = "maildir"

    def __init__(self, root: str, *, folder_roles: dict | None = None):
        self.root = os.path.abspath(os.path.expanduser(root))
        self._folder_roles = folder_roles or {}

    # ------------------------------------------------------------------ seam

    def capabilities(self) -> SourceCaps:
        return SourceCaps(
            name=self.name,
            cursor_kind=CURSOR_MTIME,
            incremental=True,
            # A file vanishing is indistinguishable from a file never fetched,
            # and the project's policy is archive-forever anyway.
            supports_delete_detection=False,
            max_connections=1,
        )

    def list_folders(self) -> List[Folder]:
        """The root (if it is a Maildir) plus any Maildir subdirectory.

        Handles both layouts: a plain ``Maildir/`` with ``cur``/``new``, and
        Maildir++ where sub-folders are dot-prefixed siblings (``.Sent``).
        """
        folders: List[Folder] = []
        if _is_maildir(self.root):
            folders.append(Folder("INBOX", resolve_role("INBOX", overrides=self._folder_roles)))
        if not os.path.isdir(self.root):
            return folders
        for entry in sorted(os.listdir(self.root)):
            path = os.path.join(self.root, entry)
            if entry in _MAILDIR_SUBDIRS or entry.startswith(".tmp") or not os.path.isdir(path):
                continue
            if not _is_maildir(path):
                continue
            # Maildir++ names sub-folders ".Sent"; present the readable form.
            name = entry.lstrip(".") or entry
            folders.append(Folder(name, resolve_role(name, overrides=self._folder_roles)))
        return folders

    def open_folder(self, folder: Folder) -> Folder:
        """No-op besides confirming existence: a Maildir has no generation.

        Nothing here can invalidate previously issued cursors the way an IMAP
        UIDVALIDITY bump does, so the generation stays constant and the runner
        never resets.
        """
        if not _is_maildir(self._folder_path(folder.name)):
            raise FileNotFoundError(f"not a maildir: {self._folder_path(folder.name)}")
        return Folder(folder.name, folder.role, generation="maildir")

    def initial_cursor(self, folder: Folder) -> Cursor:
        return Cursor(CURSOR_MTIME, {"mtime": 0.0, "name": ""})

    def fetch_delta(self, folder: Folder, cursor: Cursor) -> Iterator[RawMessage]:
        """Yield messages newer than the cursor, oldest first.

        Ordering by ``(mtime, filename)`` — not mtime alone — is what makes the
        watermark safe: several messages routinely share an mtime, and without the
        filename tie-break, resuming would skip every one of them but the first.
        """
        last_mtime = float(cursor.value.get("mtime", 0.0))
        last_name = str(cursor.value.get("name", ""))
        base = self._folder_path(folder.name)

        entries = []
        for sub in _MAILDIR_SUBDIRS:
            d = os.path.join(base, sub)
            if not os.path.isdir(d):
                continue
            for fn in os.listdir(d):
                p = os.path.join(d, fn)
                try:
                    mtime = os.path.getmtime(p)
                except OSError:
                    continue  # vanished between listing and stat — next run will see it
                if (mtime, fn) <= (last_mtime, last_name):
                    continue
                entries.append((mtime, fn, p))

        for mtime, fn, path in sorted(entries):
            try:
                with open(path, "rb") as fh:
                    raw = fh.read()
            except OSError:
                continue
            if not raw:
                continue
            yield RawMessage(
                raw=raw,
                source_uid=fn,
                folder=folder.name,
                internal_date=datetime.fromtimestamp(mtime, tz=timezone.utc),
            )

    def advance(self, cursor: Cursor, message: RawMessage) -> Cursor:
        mtime = message.internal_date.timestamp() if message.internal_date else 0.0
        current = (float(cursor.value.get("mtime", 0.0)), str(cursor.value.get("name", "")))
        candidate = (mtime, message.source_uid)
        # max() rather than blind assignment: a source that ever yields out of
        # order must not be able to walk the watermark backwards.
        best = max(current, candidate)
        return Cursor(CURSOR_MTIME, {"mtime": best[0], "name": best[1]})

    def close(self) -> None:
        """Nothing to release — present so the seam is uniform."""

    # --------------------------------------------------------------- helpers

    def _folder_path(self, name: str) -> str:
        if name == "INBOX" and _is_maildir(self.root):
            return self.root
        direct = os.path.join(self.root, name)
        if _is_maildir(direct):
            return direct
        return os.path.join(self.root, f".{name}")
