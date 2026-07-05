"""Shared page-capped PDF rendering for OCR providers.

``RAG_ATTACH_MAX_PAGES`` (default 10) bounds how many pages any OCR backend renders
and reads; longer PDFs are truncated and the truncation is logged so the context
loss is never silent. Pages past the cap are never rendered (``last_page``), so a
huge scanned PDF cannot exhaust memory or hang an ingest run.
"""

from __future__ import annotations

import logging
import os

LOGGER = logging.getLogger("mailrag.attachments")
_DEFAULT_MAX_PAGES = 10


def max_pages() -> int:
    try:
        return max(1, int(os.getenv("RAG_ATTACH_MAX_PAGES", str(_DEFAULT_MAX_PAGES))))
    except ValueError:
        return _DEFAULT_MAX_PAGES


def render_pdf_pages(data: bytes, log=None):
    """Render at most max_pages() pages of a PDF to PIL images.

    Logs (warning) when the document is longer than the cap. Raises if
    pdf2image/poppler are unavailable — callers translate that to their
    unavailable status.
    """
    import pdf2image

    log = log or LOGGER.warning
    cap = max_pages()
    try:
        total = int(pdf2image.pdfinfo_from_bytes(data).get("Pages", 0))
    except Exception:
        total = 0  # pdfinfo failed; the cap below still bounds the render
    if total > cap:
        log(f"PDF has {total} pages; reading only the first {cap} (RAG_ATTACH_MAX_PAGES={cap})")
    return pdf2image.convert_from_bytes(data, last_page=cap)
