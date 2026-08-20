"""Bulk measurement of attachment blobs — the cheap tier of noise judgement.

Extraction is otherwise lazy: ``get_attachment`` runs OCR/LLM on first fetch and
caches it, so nothing expensive happens until something is actually read. That
policy is right for the 3,059 real documents in a corpus this size, where a
single PDF can take minutes.

It does not, however, help the listing. Filtering happens when attachments are
*listed*, extraction when one is *fetched* — so the blobs nobody has fetched are
exactly the ones polluting listings, and lazy measurement never reaches them.

The resolution is that the noisy tier is not expensive. Decoration is small: OCR
on a 6KB signature strip is ~0.05s, and the whole small-image pool of a 45k-row
corpus is roughly 1,700 blobs — minutes, once, ever, because measurement is keyed
by content hash and every repeat of that logo is the same blob. So this module
measures the cheap tier in bulk at build time, and leaves the expensive tier to
the lazy path.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from src.attachments.signals import measure_blob

# Only blobs below this are measured in bulk. Above it, extraction cost stops
# being negligible and the lazy path should own it — and an image that large is
# never treated as decoration anyway, so measuring it buys the listing nothing.
DEFAULT_MAX_SIZE = 100_000


@dataclass
class ClassifyStats:
    measured: int = 0
    skipped: int = 0
    failed: int = 0

    def as_dict(self) -> dict:
        return {"measured": self.measured, "skipped": self.skipped, "failed": self.failed}


def classify_blobs(
    store,
    *,
    extractor=None,
    max_size: int = DEFAULT_MAX_SIZE,
    images_only: bool = True,
    limit: Optional[int] = None,
    progress: bool = False,
) -> ClassifyStats:
    """Measure every not-yet-measured blob under ``max_size``; record the signals.

    Idempotent and resumable: it works from ``unmeasured_blobs()``, so an
    interrupted run picks up where it stopped and a second run over an unchanged
    corpus does nothing. Failures are counted, not raised — one unreadable image
    must not abort a bulk pass.
    """
    todo = store.unmeasured_blobs(max_size=max_size, images_only=images_only)
    if limit is not None:
        todo = todo[:limit]
    if not todo:
        # Nothing to measure: return before constructing an engine. Tesseract
        # surfaces a missing binary as OCR_UNAVAILABLE per blob rather than
        # raising here, but a re-run over an already-measured corpus should not
        # depend on that being true of every future provider.
        return ClassifyStats()

    if extractor is None:
        from src.attachments.extract import build_default_extractor

        # Tesseract, not the global default. This pass IS the cheap tier — it
        # exists because measuring a signature strip costs ~0.05s — and the
        # global default is `llm`, which would turn a five-minute bulk pass into
        # thousands of model calls to answer "does this image contain words".
        extractor = build_default_extractor(
            os.environ.get("RAG_ATTACH_CLASSIFY_EXTRACTOR", "tesseract")
        )

    bar = None
    if progress:
        try:
            from tqdm import tqdm

            bar = tqdm(total=len(todo), unit="blob", desc="classify")
        except ImportError:
            bar = None

    stats = ClassifyStats()
    for sha256, mime, filename, _size in todo:
        try:
            path = store.path_for(sha256)
            if not os.path.exists(path):
                stats.skipped += 1
                continue
            with open(path, "rb") as fh:
                data = fh.read()
            store.put_signals(sha256, measure_blob(data, mime, filename or "", extractor))
            stats.measured += 1
        except Exception:
            # A blob that cannot be measured simply keeps no signals, and the
            # listing falls back to the recurrence heuristic for it.
            stats.failed += 1
        finally:
            if bar is not None:
                bar.update(1)
    if bar is not None:
        bar.close()
    return stats
