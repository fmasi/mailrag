"""Content-addressed blacklist of unwanted .eml files.

Instead of deleting (or copying) source emails, we record the
``sha256`` of each unwanted file's raw bytes in a plain-text blacklist.
The indexer skips any file whose hash is listed; cleaning passes (regex
noise rules, LLM deep-clean) *append* to it. Hashing the whole file makes
the key survive folder moves/renames and doubles as exact-file dedup,
while leaving the originals byte-for-byte untouched.

Blacklist file format: one hex sha256 per line; blank lines and lines
starting with ``#`` are ignored.

Stdlib-only and host-testable.
"""

import hashlib
import os
from typing import Iterable, List, Set, Tuple

_CHUNK = 1 << 16


def file_sha256(path: str) -> str:
    """Return the hex sha256 of a file's raw bytes (streamed)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(_CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def load_blacklist(path: str) -> Set[str]:
    """Load the set of blacklisted hashes; missing file -> empty set."""
    if not os.path.exists(path):
        return set()
    hashes: Set[str] = set()
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#"):
                hashes.add(line)
    return hashes


def append_to_blacklist(path: str, hashes: Iterable[str]) -> int:
    """Append hashes not already present. Returns how many were added."""
    existing = load_blacklist(path)
    new = [h for h in dict.fromkeys(hashes) if h not in existing]
    if new:
        with open(path, "a") as fh:
            for h in new:
                fh.write(h + "\n")
    return len(new)


def filter_blacklisted(
    file_paths: Iterable[str], blacklist_path: str
) -> Tuple[List[str], List[str]]:
    """Split *file_paths* into (kept, skipped) by blacklist membership."""
    blacklisted = load_blacklist(blacklist_path)
    if not blacklisted:
        return list(file_paths), []
    kept: List[str] = []
    skipped: List[str] = []
    for path in file_paths:
        (skipped if file_sha256(path) in blacklisted else kept).append(path)
    return kept, skipped
