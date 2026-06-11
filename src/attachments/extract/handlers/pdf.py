"""PDF handler: prefer the embedded text layer (pypdf); if there is none, hand the
bytes to the configured OCR provider (which renders pages and reads them)."""
from __future__ import annotations

import io

from src.attachments.extract.mime import is_pdf
from src.attachments.extract.ocr.base import OcrProvider
from src.attachments.extract.result import ExtractResult, ok


def _pdf_text(data: bytes) -> str:
    """Return the embedded text layer, or '' if absent/unreadable. Never raises."""
    try:
        import pypdf
    except Exception:
        return ""
    try:
        reader = pypdf.PdfReader(io.BytesIO(data))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    except Exception:
        return ""


class PdfHandler:
    def __init__(self, ocr: OcrProvider):
        self._ocr = ocr

    def can_handle(self, mime: str, filename: str) -> bool:
        return is_pdf(mime, filename)

    def extract(self, data: bytes, mime: str, filename: str) -> ExtractResult:
        text = _pdf_text(data)
        if text.strip():
            return ok(text, "pdf")
        out = self._ocr.read(data, "application/pdf", filename)   # image-only -> OCR
        return ExtractResult(out.text, out.status, f"pdf+{out.provider}")
