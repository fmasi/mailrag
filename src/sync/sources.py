"""The provider-agnostic seam every sync source implements (issue #101).

iCloud IMAP is the first implementation, not the shape of the API. A source
contributes exactly two provider-specific things — **how you enumerate new mail**
and **what a cursor is** — and everything downstream (spool, noise filter,
summaries, chunking, embedding, Qdrant) is unchanged because it only ever sees
``.eml`` files.

Cursors are deliberately **opaque**. Hard-coding IMAP's ``UIDVALIDITY`` /
``UID`` / ``MODSEQ`` into the state schema would have to be unpicked the first
time someone syncs Gmail (``historyId``), Fastmail (JMAP ``state``) or a
Microsoft tenant (delta tokens). Instead a cursor is ``(kind, value)`` plus a
**generation** — the provider's "everything you knew is void" signal, which is
UIDVALIDITY for IMAP, a 404-on-historyId for Gmail, ``cannotCalculateChanges``
for JMAP. One concept, one reset path.

A generation reset is cheap by construction: the ledger is keyed on content hash
and the Pass-2 cache on content identity, so re-enumerating a folder costs
bandwidth but no LLM calls and no index churn.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Iterator, List, Optional, Protocol, runtime_checkable


class FolderRole(str, Enum):
    """What a folder *is*, independent of what the provider calls it.

    iCloud has no ``SPECIAL-USE`` and uses literal names ("Sent Messages");
    Gmail exposes labels; Dovecot advertises ``\\Sent``. Scope decisions are
    expressed in roles so "everything except junk and trash" means the same thing
    on every provider.
    """

    INBOX = "inbox"
    SENT = "sent"
    ARCHIVE = "archive"
    DRAFTS = "drafts"
    JUNK = "junk"
    TRASH = "trash"
    OTHER = "other"


# Literal folder names seen in the wild, lowercased. Consulted only when the
# server advertises no SPECIAL-USE flag. Deliberately conservative: an unknown
# name becomes OTHER (which the default scope *includes*) rather than being
# guessed into JUNK and silently dropped.
_NAME_TO_ROLE = {
    "inbox": FolderRole.INBOX,
    "sent": FolderRole.SENT,
    "sent messages": FolderRole.SENT,  # iCloud
    "sent items": FolderRole.SENT,  # Exchange
    "sent mail": FolderRole.SENT,  # Gmail
    "archive": FolderRole.ARCHIVE,
    "all mail": FolderRole.ARCHIVE,
    "drafts": FolderRole.DRAFTS,
    "junk": FolderRole.JUNK,
    "junk e-mail": FolderRole.JUNK,
    "spam": FolderRole.JUNK,
    "bulk mail": FolderRole.JUNK,
    "trash": FolderRole.TRASH,
    "deleted messages": FolderRole.TRASH,  # iCloud
    "deleted items": FolderRole.TRASH,  # Exchange
    "bin": FolderRole.TRASH,
}

# IMAP SPECIAL-USE attributes (RFC 6154) -> role. Authoritative when present.
_FLAG_TO_ROLE = {
    "\\inbox": FolderRole.INBOX,
    "\\sent": FolderRole.SENT,
    "\\archive": FolderRole.ARCHIVE,
    "\\all": FolderRole.ARCHIVE,
    "\\drafts": FolderRole.DRAFTS,
    "\\junk": FolderRole.JUNK,
    "\\trash": FolderRole.TRASH,
}


def resolve_role(
    name: str,
    flags: Optional[List[str]] = None,
    overrides: Optional[Dict[str, str]] = None,
) -> FolderRole:
    """Classify a folder. Precedence: user override > SPECIAL-USE flag > name.

    The user override wins outright — no name table survives contact with every
    mailbox, and a mis-scoped folder is the user's to correct.
    """
    if overrides:
        # Case-insensitive lookup so a config need not match the server's casing.
        lowered = {k.lower(): v for k, v in overrides.items()}
        raw = lowered.get(name.lower())
        if raw:
            try:
                return FolderRole(raw.lower())
            except ValueError as exc:
                raise ValueError(
                    f"unknown folder role {raw!r} for folder {name!r}; "
                    f"expected one of {[r.value for r in FolderRole]}"
                ) from exc
    for f in flags or []:
        role = _FLAG_TO_ROLE.get(str(f).lower())
        if role:
            return role
    # Leaf name only: "INBOX/Receipts" is a sub-folder, not the inbox itself, so
    # only an exact whole-name match promotes it out of OTHER.
    return _NAME_TO_ROLE.get(name.strip().lower(), FolderRole.OTHER)


@dataclass(frozen=True)
class Folder:
    """A syncable folder, with the provider's current generation for it.

    ``generation`` is captured at enumeration time so the runner can compare it
    against the stored one and reset the cursor when the provider has invalidated
    it (IMAP UIDVALIDITY change, mailbox recreated, account re-added).
    """

    name: str
    role: FolderRole = FolderRole.OTHER
    generation: str = ""


@dataclass(frozen=True)
class Cursor:
    """Opaque per-folder position. ``kind`` names the scheme; ``value`` is its state.

    Sources own the contents of ``value`` entirely — the state store persists it
    as JSON and never interprets it.
    """

    kind: str
    value: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> Dict[str, Any]:
        return {"kind": self.kind, "value": dict(self.value)}

    @classmethod
    def from_json(cls, data: Optional[Dict[str, Any]]) -> Optional["Cursor"]:
        if not data:
            return None
        return cls(kind=data.get("kind", ""), value=dict(data.get("value") or {}))


# Known cursor schemes. Free-form by design — a new source may introduce its own;
# these exist so ``--status`` can render something meaningful.
CURSOR_UID = "uid"  # IMAP without CONDSTORE
CURSOR_UID_MODSEQ = "uid+modseq"  # IMAP with CONDSTORE
CURSOR_HISTORY_ID = "history_id"  # Gmail API
CURSOR_JMAP_STATE = "jmap_state"  # JMAP Email/changes
CURSOR_DELTA_TOKEN = "delta_token"  # Microsoft Graph
CURSOR_MTIME = "mtime"  # local filesystem sources


@dataclass(frozen=True)
class SourceCaps:
    """What a source can actually do, probed rather than assumed.

    iCloud advertises CONDSTORE/QRESYNC only *after* login (a frontend proxy
    rewrites the pre-auth CAPABILITY response), and the evidence for QRESYNC is
    years old — so capabilities are reported by the live connection and the
    runner must work with the weakest of them.
    """

    name: str
    cursor_kind: str
    incremental: bool = True
    supports_delete_detection: bool = False
    max_connections: int = 1


@dataclass(frozen=True)
class RawMessage:
    """One fetched message, before anything has been parsed out of it.

    Deliberately just bytes plus provenance: parsing is the existing loader's
    job, and re-deriving identity from the same parse the indexer uses is what
    keeps the sync ledger and the index agreeing on what an email is.
    """

    raw: bytes
    source_uid: str
    folder: str
    internal_date: Optional[datetime] = None


@runtime_checkable
class MessageSource(Protocol):
    """Enumerate new mail from somewhere. The only provider-aware interface.

    Implementations must be safe to call repeatedly: ``fetch_delta`` is expected
    to be re-run after a crash, and the runner relies on the ledger — not the
    source — to avoid re-spooling what it already has.
    """

    name: str

    def capabilities(self) -> SourceCaps:
        """Report what this connection can do (probe, don't assume)."""
        ...

    def list_folders(self) -> List[Folder]:
        """Enumerate folders with their roles. ``generation`` may be blank here.

        Cheap by design: on IMAP the generation only comes from SELECTing a
        folder, and selecting every folder just to discard the out-of-scope ones
        would be a round trip per folder for nothing. The runner filters by role
        first, then calls :meth:`open_folder` on the survivors.
        """
        ...

    def open_folder(self, folder: Folder) -> Folder:
        """Prepare a folder for fetching and return it with its live generation."""
        ...

    def initial_cursor(self, folder: Folder) -> Cursor:
        """The cursor meaning "I have nothing from this folder yet"."""
        ...

    def fetch_delta(self, folder: Folder, cursor: Cursor) -> Iterator[RawMessage]:
        """Yield messages newer than *cursor*, oldest first.

        Yielding lazily is load-bearing: the runner spools and commits each
        message as it arrives, so an interrupted fetch keeps everything it had
        already written.
        """
        ...

    def advance(self, cursor: Cursor, message: RawMessage) -> Cursor:
        """Return the cursor that accounts for *message* having been handled.

        Called after each message is durably spooled, so a run that dies halfway
        resumes from where it got to — and a single poison message can be parked
        without wedging the folder forever.
        """
        ...

    def close(self) -> None:
        """Release the connection. Must be safe to call twice."""
        ...
