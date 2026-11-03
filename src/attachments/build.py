"""Fill a collection's attachment store from the mail a profile selects.

One implementation for every front-end that builds a store over a whole
profile: ``mailrag attachments build`` and the persona recipes' ``attach``
step (the wizard and ``mailrag run``). Sync fills the store incrementally for
the mail it brings in, through its own index stage.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

#: Only blobs smaller than this are OCR-measured in the build pass.
CLASSIFY_MAX_SIZE = 100_000


def build_store(
    profile: Any,
    store_path: str,
    *,
    limit: Optional[int] = None,
    classify: bool = True,
    extractor: Optional[str] = None,
    classify_max_size: int = CLASSIFY_MAX_SIZE,
    progress: bool = True,
) -> Dict[str, Any]:
    """Store every attachment of the profile's selected mail; return counts.

    The returned dict carries the ingest counts (``emails``, ``attachments``,
    ``skipped``, ``bad_parts``), ``classified`` (the noise-signal pass, or
    ``None`` when skipped) and ``complete``: whether the store now covers the
    whole profile, so that an empty lookup in it means "no attachments".
    """
    from src.attachments.ingest_eml import ingest_eml
    from src.attachments.store import AttachmentStore
    from src.ingest.local_source import resolve_index_files

    kept, _ = resolve_index_files(
        profile.resolved_root(), profile.selection_rules, getattr(profile, "blacklist", None)
    )
    if limit:
        kept = kept[:limit]
    store = AttachmentStore(os.path.expanduser(store_path))
    try:
        # A fresh store is partial until a build finishes over the whole
        # profile. Left unmarked, the rows from a limited or interrupted run
        # made it read as complete.
        store.mark_partial()
        counts: Dict[str, Any] = dict(ingest_eml(kept, store, progress=progress))
        counts["classified"] = None
        if classify:
            # Measure the cheap tier in the same pass: the small inline images
            # that pollute listings are ~0.05s each to OCR and are measured once
            # per content hash, so this is minutes even on a large corpus. The
            # expensive documents stay on the lazy get_attachment path.
            from src.attachments.classify import classify_blobs

            # Only build an extractor when one was explicitly requested:
            # build_default_extractor(None) resolves $RAG_ATTACH_EXTRACTOR, which
            # defaults to the vision LLM, and would silently override the
            # tesseract default that makes this pass cheap enough to run in bulk.
            from src.attachments.extract import build_default_extractor

            explicit = build_default_extractor(extractor) if extractor else None
            stats = classify_blobs(
                store, extractor=explicit, max_size=classify_max_size, progress=progress
            )
            counts["classified"] = stats.as_dict()
        # Only a build over the whole profile makes an empty lookup mean "no
        # attachments". A limited run, like sync, covers part of the corpus, and
        # a build that found no mail at all (a missing or unmounted root)
        # covered none of it.
        counts["complete"] = not limit and counts.get("emails", 0) > 0
        if counts["complete"]:
            store.mark_built()
    finally:
        store.close()
    return counts
