"""The extraction facade: an ordered handler registry behind a single Extractor object.

`Extractor` is constructed with its OCR provider (an explicit dependency) and dispatches
each blob to the first handler whose can_handle() matches, else returns UNSUPPORTED.
"""

from __future__ import annotations

from typing import List

from src.attachments.extract.handlers.base import Handler
from src.attachments.extract.handlers.docx import DocxHandler
from src.attachments.extract.handlers.html import HtmlHandler
from src.attachments.extract.handlers.image import ImageHandler
from src.attachments.extract.handlers.pdf import PdfHandler
from src.attachments.extract.handlers.plaintext import PlaintextHandler
from src.attachments.extract.handlers.pptx import PptxHandler
from src.attachments.extract.handlers.xlsx import XlsxHandler
from src.attachments.extract.ocr.base import OcrProvider
from src.attachments.extract.result import ExtractResult, Status


class Extractor:
    def __init__(self, ocr: OcrProvider):
        # Order matters only where can_handle() could overlap; these are disjoint.
        self._handlers: List[Handler] = [
            PlaintextHandler(),
            HtmlHandler(),
            DocxHandler(),
            XlsxHandler(),
            PptxHandler(),
            PdfHandler(ocr),
            ImageHandler(ocr),
        ]

    def extract(self, data: bytes, mime: str, filename: str) -> ExtractResult:
        for h in self._handlers:
            if h.can_handle(mime, filename):
                try:
                    return h.extract(data, mime, filename)
                except Exception:
                    return ExtractResult("", Status.ERROR, type(h).__name__)
        return ExtractResult("", Status.UNSUPPORTED, "none")
