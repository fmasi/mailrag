"""Build a :class:`~src.sync.sources.MessageSource` from an account's config.

The single place a ``source:`` string becomes an object. Adding a provider means
adding one entry here and one module — nothing in the runner, the state store,
the spool or the pipeline changes.
"""

from __future__ import annotations

from src.sync.accounts import AccountConfig
from src.sync.secrets import resolve_secret
from src.sync.sources import MessageSource


def build_source(account: AccountConfig) -> MessageSource:
    """Construct the configured source, resolving any secret at connect time."""
    kind = (account.source or "").lower()

    if kind == "maildir":
        from src.sync.maildir_source import MaildirSource  # noqa: PLC0415

        if not account.path:
            raise ValueError(f"account {account.id!r}: maildir source needs 'path'")
        return MaildirSource(account.path, folder_roles=account.folder_roles)

    if kind == "imap":
        from src.sync.imap_source import ImapSource  # noqa: PLC0415

        missing = [f for f in ("host", "login", "secret") if not getattr(account, f)]
        if missing:
            raise ValueError(f"account {account.id!r}: imap source needs {', '.join(missing)}")
        return ImapSource(
            host=account.host,
            login=account.login,
            # Dereferenced here, at the last moment, so the secret never sits in
            # the config object being logged or printed by --status.
            password=resolve_secret(account.secret),
            port=account.port,
            ssl=account.ssl,
            folder_roles=account.folder_roles,
        )

    raise ValueError(
        f"account {account.id!r}: unknown source {account.source!r} (expected 'imap' or 'maildir')"
    )
