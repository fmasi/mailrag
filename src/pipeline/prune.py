"""Prune stage: turn noise verdicts (or pass-1 tags) into blacklist entries.

`prune` appends the content hashes of confident-noise files to the profile's
content-addressed blacklist (src/data/blacklist.py). `summarize`/`index` then skip
those files via ``resolve_index_files``, so the drop happens BEFORE the expensive
LLM pass. Nothing is deleted — only a text file of hashes is written, and a
verify-before-prune confirmation shows a sample first.
"""

from __future__ import annotations

from typing import Callable, List, Tuple

from src.data.blacklist import append_to_blacklist, file_sha256
from src.data.loaders.mail_archive_x import MailArchiveXLoader
from src.data.noise_filter import NoiseFilter
from src.ingest.local_source import resolve_index_files
from src.llm.cache import Pass2Cache
from src.pipeline import pass1


def _from_cache(profile, min_confidence: float) -> Tuple[List[str], List[str]]:
    cache = Pass2Cache(profile.pass2_cache)
    try:
        rows = list(cache.iter_noise(min_confidence))
    finally:
        cache.close()
    hashes = [r["sha256"] for r in rows]
    preview = [f"{r['confidence']:.2f}  {(r['reason'] or '')[:80]}" for r in rows[:10]]
    return hashes, preview


def _from_tag(profile) -> Tuple[List[str], List[str]]:
    kept, _ = resolve_index_files(profile.resolved_root(), profile.selection_rules, None)
    emails = MailArchiveXLoader(eml_files=kept).load()
    pass1.run(emails, NoiseFilter.from_project_rules())
    hashes: List[str] = []
    preview: List[str] = []
    for e in emails:
        if getattr(e, "noise_candidate", False) or getattr(e, "is_bulk", False):
            src = getattr(e, "source_id", "") or ""  # the .eml file path
            if src:
                hashes.append(file_sha256(src))
                if len(preview) < 10:
                    preview.append(f"{(e.sender or '')[:30]}  {(e.subject or '')[:60]}")
    return hashes, preview


def collect(profile, *, source: str, min_confidence: float = 0.7) -> Tuple[List[str], List[str]]:
    """Return ``(hashes, preview_lines)`` for the requested *source*."""
    if source in ("judge", "summarize"):
        return _from_cache(profile, min_confidence)
    if source == "tag":
        return _from_tag(profile)
    raise ValueError(f"unknown prune source {source!r} (use tag|judge|summarize)")


def run(
    profile,
    *,
    source: str,
    min_confidence: float = 0.7,
    confirm: Callable[[List[str]], bool] = lambda preview: True,
) -> int:
    """Collect drop hashes, confirm, and append them to the profile's blacklist.

    Returns the number of hashes newly added (0 if nothing to drop or declined)."""
    if not profile.blacklist:
        raise ValueError("profile has no blacklist path set")
    hashes, preview = collect(profile, source=source, min_confidence=min_confidence)
    if not hashes:
        return 0
    if not confirm(preview):
        return 0
    return append_to_blacklist(profile.blacklist, hashes)
