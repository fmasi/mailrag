"""Account configuration — accounts are **data**, not code (issue #101).

One YAML file describes every mailbox mailrag keeps fresh: where it lives, how to
authenticate, which folder roles are in scope, which collection and cleaning
profile it feeds, and how often to poll. Multi-account and multi-collection then
fall out of the loader rather than needing special cases: N entries produce N
ledgers, each targeting its own collection (or several sharing one — the ledger
dedupes on content hash).

Example::

    accounts:
      - id: personal-icloud
        source: imap
        host: imap.mail.me.com
        login: appleid@example.com
        secret: keychain:mailrag.imap.personal-icloud
        collection: personal
        profile: ~/.mailrag/personal.json
        spool_root: ~/mail/personal/incoming
        exclude_roles: [junk, trash]
        cadence: 12h
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.sync.sources import FolderRole

DEFAULT_ACCOUNTS_PATH = "~/.mailrag/accounts.yaml"

# Everything except junk and trash. Deliberately includes OTHER (arbitrary filed
# folders) and SENT — a thread without your own replies reads one-sided, which is
# exactly the context a personal RAG needs.
DEFAULT_INCLUDE_ROLES = [
    FolderRole.INBOX,
    FolderRole.SENT,
    FolderRole.ARCHIVE,
    FolderRole.OTHER,
]

_CADENCE_RE = re.compile(r"^\s*(\d+)\s*([smhd])\s*$", re.IGNORECASE)
_CADENCE_UNITS = {"s": 1, "m": 60, "h": 3600, "d": 86400}


def parse_cadence(text: str) -> int:
    """``"12h"`` -> 43200 seconds. Raises ValueError on anything else."""
    m = _CADENCE_RE.match(str(text))
    if not m:
        raise ValueError(f"cadence {text!r} is not of the form <n><s|m|h|d>, e.g. '12h'")
    n = int(m.group(1))
    if n <= 0:
        raise ValueError(f"cadence {text!r} must be positive")
    return n * _CADENCE_UNITS[m.group(2).lower()]


@dataclass
class AccountConfig:
    """One mailbox mailrag keeps fresh."""

    id: str
    source: str = "imap"
    collection: str = "email-rag"
    profile: Optional[str] = None
    spool_root: Optional[str] = None
    # Connection (source-specific; unused keys are simply ignored by a source).
    host: str = ""
    port: int = 993
    ssl: bool = True
    login: str = ""
    secret: str = ""
    path: str = ""  # maildir / local sources
    # Scope
    include_roles: List[FolderRole] = field(default_factory=lambda: list(DEFAULT_INCLUDE_ROLES))
    exclude_roles: List[FolderRole] = field(
        default_factory=lambda: [FolderRole.JUNK, FolderRole.TRASH]
    )
    folder_roles: Dict[str, str] = field(default_factory=dict)  # explicit name -> role overrides
    cadence: str = "12h"
    options: Dict[str, Any] = field(default_factory=dict)

    def cadence_seconds(self) -> int:
        return parse_cadence(self.cadence)

    def resolved_spool_root(self) -> str:
        if not self.spool_root:
            raise ValueError(f"account {self.id!r} has no spool_root")
        return os.path.abspath(os.path.expanduser(self.spool_root))

    def wants(self, role: FolderRole) -> bool:
        """Is a folder of this role in scope? Exclusion always wins."""
        if role in self.exclude_roles:
            return False
        return role in self.include_roles


def _coerce_roles(raw, field_name: str, account_id: str) -> List[FolderRole]:
    out: List[FolderRole] = []
    for item in raw or []:
        try:
            out.append(FolderRole(str(item).lower()))
        except ValueError as exc:
            raise ValueError(
                f"account {account_id!r}: unknown role {item!r} in {field_name}; "
                f"expected one of {[r.value for r in FolderRole]}"
            ) from exc
    return out


def account_from_dict(data: Dict[str, Any]) -> AccountConfig:
    """Build an :class:`AccountConfig`, validating eagerly.

    Config mistakes must surface at load time, not eight hours later inside a
    scheduled run whose output nobody is watching.
    """
    if not isinstance(data, dict):
        raise ValueError(f"each account must be a mapping, got {type(data).__name__}")
    account_id = str(data.get("id") or "").strip()
    if not account_id:
        raise ValueError("every account needs a unique 'id'")

    known = {
        "id",
        "source",
        "collection",
        "profile",
        "spool_root",
        "host",
        "port",
        "ssl",
        "login",
        "secret",
        "path",
        "include_roles",
        "exclude_roles",
        "folder_roles",
        "cadence",
        "options",
    }
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"account {account_id!r}: unknown key(s) {sorted(unknown)}")

    cfg = AccountConfig(
        id=account_id,
        source=str(data.get("source", "imap")).lower(),
        collection=str(data.get("collection", "email-rag")),
        profile=data.get("profile"),
        spool_root=data.get("spool_root"),
        host=str(data.get("host", "")),
        port=int(data.get("port", 993)),
        ssl=bool(data.get("ssl", True)),
        login=str(data.get("login", "")),
        secret=str(data.get("secret", "")),
        path=str(data.get("path", "")),
        folder_roles={str(k): str(v) for k, v in (data.get("folder_roles") or {}).items()},
        cadence=str(data.get("cadence", "12h")),
        options=dict(data.get("options") or {}),
    )
    if "include_roles" in data:
        cfg.include_roles = _coerce_roles(data["include_roles"], "include_roles", account_id)
    if "exclude_roles" in data:
        cfg.exclude_roles = _coerce_roles(data["exclude_roles"], "exclude_roles", account_id)

    cfg.cadence_seconds()  # validate now, not at schedule time
    for name, role in cfg.folder_roles.items():
        try:
            FolderRole(str(role).lower())
        except ValueError as exc:
            raise ValueError(
                f"account {account_id!r}: unknown role {role!r} for folder {name!r}"
            ) from exc
    return cfg


def load_accounts(path: Optional[str] = None) -> List[AccountConfig]:
    """Load and validate every account. Missing file -> empty list.

    A missing config is a normal state ("sync isn't set up yet"), not an error;
    the CLI turns it into an actionable message.
    """
    import yaml  # noqa: PLC0415 — keep the import cost off the hot CLI path

    resolved = os.path.expanduser(path or DEFAULT_ACCOUNTS_PATH)
    if not os.path.exists(resolved):
        return []
    with open(resolved, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict) or "accounts" not in data:
        raise ValueError(f"{resolved}: expected a top-level 'accounts:' list")
    accounts = [account_from_dict(a) for a in (data.get("accounts") or [])]
    ids = [a.id for a in accounts]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        # Duplicate ids would silently share a ledger and fight over cursors.
        raise ValueError(f"{resolved}: duplicate account id(s) {sorted(dupes)}")
    return accounts


def get_account(account_id: str, path: Optional[str] = None) -> AccountConfig:
    """Look up one account by id, with a helpful error listing what does exist."""
    accounts = load_accounts(path)
    for a in accounts:
        if a.id == account_id:
            return a
    known = ", ".join(a.id for a in accounts) or "(none configured)"
    raise KeyError(f"no account {account_id!r} in {path or DEFAULT_ACCOUNTS_PATH}; known: {known}")
