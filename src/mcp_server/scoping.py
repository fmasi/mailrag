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

Profiles are discovered by reading ``*.profile.json`` under
``$MAILRAG_PROFILE_DIR`` (default ``~``) and indexing them by the ``collection``
they name, because nothing else records that mapping — onboarding manifests carry
the collection but not the profile that produced it.
"""

from __future__ import annotations

import glob
import os
from typing import Dict, List, Optional

DEFAULT_PROFILE_DIR = "~"

# (profile-dir, mtime-signature) -> {collection: profile path}
_PROFILE_CACHE: Dict[tuple, Dict[str, str]] = {}
# profile path -> selected files
_FILES_CACHE: Dict[str, List[str]] = {}


def profile_dir() -> str:
    return os.path.expanduser(os.environ.get("MAILRAG_PROFILE_DIR") or DEFAULT_PROFILE_DIR)


def _signature(paths: List[str]) -> tuple:
    """Cache key that changes when a profile is edited or added."""
    out = []
    for p in paths:
        try:
            out.append((p, os.path.getmtime(p)))
        except OSError:
            out.append((p, 0.0))
    return tuple(out)


def collection_profiles() -> Dict[str, str]:
    """``{collection: profile path}`` for every readable profile on disk."""
    paths = sorted(glob.glob(os.path.join(profile_dir(), "*.profile.json")))
    key = (profile_dir(), _signature(paths))
    cached = _PROFILE_CACHE.get(key)
    if cached is not None:
        return cached

    from src.profile import CorpusProfile

    found: Dict[str, str] = {}
    for path in paths:
        try:
            prof = CorpusProfile.load(path)
        except Exception:
            continue  # an unreadable profile must not break scoping for the others
        if getattr(prof, "collection", None):
            found[prof.collection] = path
    _PROFILE_CACHE[key] = found
    return found


def files_for_collection(collection: str) -> Optional[List[str]]:
    """The ``.eml`` files belonging to ``collection``, or ``None`` if unknown.

    ``None`` means "no profile names this collection", which the caller must
    treat as a refusal to scope rather than a licence to scan everything.
    """
    path = collection_profiles().get(collection)
    if path is None:
        return None
    cached = _FILES_CACHE.get(path)
    if cached is not None:
        return cached

    from src.ingest.local_source import resolve_index_files
    from src.profile import CorpusProfile

    prof = CorpusProfile.load(path)
    kept, _ = resolve_index_files(
        prof.resolved_root(), prof.selection_rules, getattr(prof, "blacklist", None)
    )
    files = sorted(kept)
    _FILES_CACHE[path] = files
    return files


def clear_cache() -> None:
    """Drop memoised profiles and file lists (tests, and after re-onboarding)."""
    _PROFILE_CACHE.clear()
    _FILES_CACHE.clear()
