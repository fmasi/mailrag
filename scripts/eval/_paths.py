"""Shared setup for the private eval scripts (issue #97 follow-up).

These scripts reproduce the numbers the README and landing page publish from the
**private** corpus — the thread-reconstruction ladder, the contextual-summary
lift, and the TREC Legal comparison. They are not runnable by a stranger (they
need a private mailbox, a private query set, and in places a paid API key); the
public, reproducible benchmark is ``bench_public.py``.

They previously began with a hardcoded absolute path into a git worktree::

    WT = "/Users/fmasi/Git/mailrag/.claude/worktrees/p2-backend-agnostic"
    sys.path.insert(0, WT)
    os.chdir(WT)

That worktree no longer exists, so every one of them failed on import. Worse, the
`chdir` meant they silently ran against a *stale checkout* of the code rather than
the working tree — so a number produced after any code change was not necessarily
a number about the current code.

This module replaces that with a repo-relative root plus environment overrides, so
the scripts run against the checkout they live in, on any machine.

Every data location can be overridden by an environment variable, because the
private corpora do not live in the repo and their location is personal:

===========================  ==========================================
``MAILRAG_EVAL_QUERIES``     private labelled query set (JSONL)
``MAILRAG_EVAL_TREC``        TREC Legal directory (qrels + topics)
``MAILRAG_EVAL_TREC_MBOX``   TREC Legal mbox
``QDRANT_URL``               vector store (defaults to localhost)
===========================  ==========================================
"""

from __future__ import annotations

import os
import pathlib
import sys

# The repo root, derived from this file's location — never a hardcoded home dir.
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


def bootstrap(*, offline: bool = True) -> pathlib.Path:
    """Put the repo on ``sys.path`` and set the usual eval environment defaults.

    Deliberately does NOT ``chdir``: the old scripts changed directory into a
    worktree, which is how they ended up measuring stale code. Returns the repo
    root so callers can build paths from it.
    """
    root = str(REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    os.environ.setdefault("QDRANT_URL", "http://localhost:6333")
    if offline:
        # The eval runs against already-downloaded weights; a surprise network
        # fetch mid-benchmark is a timing artefact, not a measurement.
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    return REPO_ROOT


def data_path(env_var: str, default: str, *, what: str) -> pathlib.Path:
    """Resolve a private-data location, honouring *env_var*.

    Fails loudly and specifically when the data is absent. These scripts are run
    rarely, months apart, to re-verify published numbers — a clear "this corpus
    is missing, set this variable" beats a stack trace from three frames deeper.
    """
    p = pathlib.Path(os.environ.get(env_var, default)).expanduser()
    if not p.exists():
        raise SystemExit(
            f"{what} not found at {p}\n"
            f"  Set {env_var} to its location, e.g.  {env_var}=/path/to/data python -m {__package__}...\n"
            f"  (This is a PRIVATE eval script. The public benchmark is `make bench`.)"
        )
    return p


def require_key(
    env_var: str = "NVIDIA_API_KEY", *, what: str = "the NVIDIA rerank endpoint"
) -> str:
    """Return an API key or exit with an actionable message.

    The scripts used to `assert KEY`, which fails with a bare AssertionError and
    no hint about which key, why, or what it costs.
    """
    raw = os.environ.get(env_var, "").strip()
    if not raw:
        raise SystemExit(
            f"{env_var} is not set — required for {what}.\n"
            f"  This arm calls a PAID endpoint. Either export the key for this run:\n"
            f"      {env_var}=nvapi-... python -m scripts.eval.<script>\n"
            f"  or point it at a stored secret, which is the project convention:\n"
            f"      {env_var}=keychain:mailrag.nvidia.token python -m scripts.eval.<script>\n"
            f"  The public benchmark (`make bench`) needs no key and no private data."
        )
    # A reference (keychain:/env:/file:) is dereferenced through the same resolver
    # the sync passwords and RAG_LLM_API_KEY use, so the token never has to sit in
    # a shell history or a dotfile. A literal is still accepted — exporting a key
    # for one manual run is normal, and this variable is not read from config.
    if raw.split(":", 1)[0] in ("keychain", "env", "file"):
        bootstrap()  # the resolver lives in src/, which may not be importable yet
        from src.config.secrets import SecretError, resolve_secret  # noqa: PLC0415

        try:
            return resolve_secret(raw)
        except SecretError as exc:
            raise SystemExit(
                f"{env_var} is a {raw.split(':', 1)[0]}: reference that could not be resolved: {exc}"
            ) from exc
    return raw
