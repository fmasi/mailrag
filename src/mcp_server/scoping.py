"""Map a collection to the raw ``.eml`` files that belong to it.

``grep_email`` walks files on disk rather than an index, and every corpus on this
machine shares one root: the work and personal profiles both resolve to
``~/rag_eml`` and differ only by their selection rules. So an unscoped walk reads
every corpus at once — an agent researching work greps a string and gets personal
mail back, however well the vector collections and attachment stores are
separated.

Scoping therefore means applying the corpus profile's own selection rules to the
walk, which is exactly what indexing does. Resolution is cheap (~0.2s for 73k
files) and the two profiles here select disjoint sets.

A collection's profile is found from the onboarding manifest that recorded it,
falling back to scanning ``*.profile.json`` under ``$MAILRAG_PROFILE_DIR``
(default ``~``) and indexing them by the ``collection`` they name.

Both lookups are memoised, and both invalidate on the profile's mtime, because
the MCP server outlives the profiles it reads: re-onboarding a collection
rewrites its profile underneath a running process, and nothing on that path
clears these caches.
"""

from __future__ import annotations

import glob
import os
from typing import Dict, List, NamedTuple, Optional, Tuple

DEFAULT_PROFILE_DIR = "~"

# (profile-dir, mtime-signature) -> {collection: profile path}
_PROFILE_CACHE: Dict[tuple, Dict[str, str]] = {}
# profile path -> (profile mtime when resolved, selected files, corpus root)
_SCOPE_CACHE: Dict[str, Tuple[float, List[str], str]] = {}


class CorpusScope(NamedTuple):
    """One collection's walk: the files to read and the root they came from.

    Deliberately one value rather than two lookups. The files and the root are
    the same fact read from the same profile, and resolving them separately let
    them disagree: a long-running server answered from a cached file list while
    reporting a freshly-read root, so a re-onboarded collection was grepped in
    its old corpus under its new corpus's name.
    """

    files: List[str]
    root: str


def profile_dir() -> str:
    return os.path.expanduser(os.environ.get("MAILRAG_PROFILE_DIR") or DEFAULT_PROFILE_DIR)


def _mtime(path: str) -> float:
    """``path``'s mtime, or ``0.0`` when it cannot be stat'ed."""
    try:
        return os.path.getmtime(path)
    except OSError:
        return 0.0


def _signature(paths: List[str]) -> tuple:
    """Cache key that changes when a profile is edited or added."""
    return tuple((p, _mtime(p)) for p in paths)


def collection_profiles() -> Dict[str, str]:
    """``{collection: profile path}`` for every readable profile on disk."""
    paths = sorted(glob.glob(os.path.join(profile_dir(), "*.profile.json")))
    key = (profile_dir(), _signature(paths))
    cached = _PROFILE_CACHE.get(key)
    if cached is not None:
        return cached

    from src.profile import CorpusProfile

    # Recorded mappings win: a manifest states which profile built a collection,
    # while directory scanning only infers it from whatever files happen to sit
    # in one place under whatever names.
    found: Dict[str, str] = {}
    try:
        from src.onboard import manifest_profile_paths

        found.update(manifest_profile_paths())
    except Exception:
        pass
    for path in paths:
        try:
            prof = CorpusProfile.load(path)
        except Exception:
            continue  # an unreadable profile must not break scoping for the others
        if getattr(prof, "collection", None) and prof.collection not in found:
            found[prof.collection] = path
    _PROFILE_CACHE[key] = found
    return found


def scope_for_collection(collection: Optional[str]) -> Optional[CorpusScope]:
    """The files ``collection`` selects and the root they came from, or ``None``.

    ``None`` means "no profile names this collection", which the caller must
    treat as a refusal to scope rather than a licence to scan everything. A
    falsy ``collection`` is the same refusal in advance: nothing to scope to.

    The walk reads the profile's own root, which need not be
    ``$MAILRAG_EML_ROOT`` — so a caller reporting "which corpus answered this"
    reads the root off the scope it walked, never by re-resolving the default.

    Resolution is memoised per profile and invalidated by the profile's mtime.
    That matters because the server is long-lived: re-onboarding a collection
    rewrites its profile while the process keeps running, and a cache with no
    invalidation would go on grepping the previous corpus. Nothing in the
    onboarding path calls :func:`clear_cache`, so the mtime is what makes a
    warm cache safe.

    Raises:
        ValueError: when a profile is named but cannot be read. Scoping has
            failed, and both alternatives are worse than saying so — scanning
            everything defeats the point, and reporting "unknown collection"
            would send the caller looking for a profile that is right there.
    """
    if not collection:
        return None
    path = collection_profiles().get(collection)
    if path is None:
        return None
    stamp = _mtime(path)
    cached = _SCOPE_CACHE.get(path)
    if cached is not None and cached[0] == stamp:
        return CorpusScope(cached[1], cached[2])

    from src.ingest.local_source import resolve_index_files
    from src.profile import CorpusProfile

    try:
        prof = CorpusProfile.load(path)
    except Exception as exc:
        raise ValueError(
            f"corpus profile for collection {collection!r} at {path!r} could not be read: {exc}"
        ) from exc
    root = prof.resolved_root()
    kept, _ = resolve_index_files(root, prof.selection_rules, getattr(prof, "blacklist", None))
    files = sorted(kept)
    _SCOPE_CACHE[path] = (stamp, files, root)
    return CorpusScope(files, root)


def clear_cache() -> None:
    """Drop memoised profiles and resolved scopes (tests, and after re-onboarding)."""
    _PROFILE_CACHE.clear()
    _SCOPE_CACHE.clear()
