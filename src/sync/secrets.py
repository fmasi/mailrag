"""Resolve an account's password from a secret **reference**, never a literal.

Account config is a file users are likely to copy into a repo, paste into an
issue, or sync to a backup. So it holds a reference — ``keychain:...``,
``env:...``, ``file:...`` — and this module dereferences it at connect time. A
plaintext password in ``accounts.yaml`` is rejected rather than merely
discouraged: the config is not a safe place for it, and silently accepting one
teaches the wrong habit.

macOS Keychain is the default on darwin but never the only option — mailrag runs
on Linux too, and a resolver that only understood ``security(1)`` would make the
whole sync feature macOS-only.
"""

from __future__ import annotations

import os
import shutil
import subprocess

_SCHEMES = ("keychain", "env", "file")


class SecretError(RuntimeError):
    """A secret reference could not be resolved."""


def resolve_secret(ref: str) -> str:
    """Dereference *ref* and return the secret.

    Supported forms::

        keychain:<service>   macOS Keychain generic password, looked up by service
        env:<VAR>            environment variable (CI, containers)
        file:<path>          first line of a file (expect 0600 on POSIX)

    Raises :class:`SecretError` for an unknown scheme, a missing target, or a
    bare literal.
    """
    if not ref or ":" not in ref:
        # NEVER echo the rejected value: if it is a literal, it is the password,
        # and this message reaches the sync log and the run record in the state
        # DB, replayed on every scheduled tick.
        raise SecretError(
            "secret must be a reference, not a literal value "
            f"({'it is empty' if not ref else 'value withheld'}); "
            f"use one of {', '.join(s + ':...' for s in _SCHEMES)}"
        )
    scheme, _, target = ref.partition(":")
    scheme = scheme.strip().lower()
    target = target.strip()
    # Validate the scheme BEFORE quoting it back. A value like "hunter2:" parses
    # as scheme="hunter2" — echoing that leaks the password just as surely as
    # echoing the whole reference would.
    if scheme not in _SCHEMES:
        raise SecretError(
            f"unknown secret scheme (value withheld); expected one of "
            f"{', '.join(s + ':...' for s in _SCHEMES)}"
        )
    if not target:
        raise SecretError(f"secret reference with scheme {scheme!r} names no target")

    if scheme == "env":
        value = os.environ.get(target)
        if not value:
            raise SecretError(f"environment variable {target} is unset or empty")
        return value

    if scheme == "file":
        path = os.path.expanduser(target)
        try:
            with open(path, encoding="utf-8") as fh:
                value = fh.readline().strip()
        except OSError as exc:
            raise SecretError(f"cannot read secret file {path}: {exc}") from exc
        if not value:
            raise SecretError(f"secret file {path} is empty")
        return value

    return _from_keychain(target)  # the only remaining validated scheme


def _from_keychain(service: str) -> str:
    """Read a generic password from the macOS Keychain by service name."""
    if not shutil.which("security"):
        raise SecretError(
            f"keychain: secrets need the macOS `security` tool, which is not on PATH "
            f"(on Linux use env: or file: instead of keychain:{service})"
        )
    try:
        out = subprocess.run(
            ["security", "find-generic-password", "-s", service, "-w"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise SecretError(f"keychain lookup for {service!r} failed: {exc}") from exc
    if out.returncode != 0:
        raise SecretError(
            f"no keychain item for service {service!r}. Store one with:\n"
            f"  security add-generic-password -U -a <account> -s {service} -w"
        )
    # -w prints the password followed by a newline; nothing else goes to stdout.
    value = out.stdout.rstrip("\n")
    if not value:
        raise SecretError(f"keychain item {service!r} holds an empty password")
    return value


def store_keychain_secret(service: str, account: str, password: str) -> None:
    """Create/replace a Keychain item — the write half, used by ``sync --setup``.

    ``-U`` updates in place so re-running setup after rotating an app-specific
    password does not fail on a duplicate item.
    """
    if not shutil.which("security"):
        raise SecretError("the macOS `security` tool is not on PATH")
    res = subprocess.run(
        ["security", "add-generic-password", "-U", "-a", account, "-s", service, "-w", password],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if res.returncode != 0:
        raise SecretError(f"could not store keychain item {service!r}: {res.stderr.strip()}")
