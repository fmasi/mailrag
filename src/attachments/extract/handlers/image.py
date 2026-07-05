"""Image handler: delegate raster images to the configured OCR provider."""

from __future__ import annotations

from src.attachments.extract.mime import mime_base
from src.attachments.extract.ocr.base import OcrProvider
from src.attachments.extract.result import ExtractResult


class ImageHandler:
    def __init__(self, ocr: OcrProvider):
        self._ocr = ocr

    def can_handle(self, mime: str, filename: str) -> bool:
        return mime_base(mime).startswith("image/")

    def extract(self, data: bytes, mime: str, filename: str) -> ExtractResult:
        out = self._ocr.read(data, mime, filename)
        return ExtractResult(out.text, out.status, out.provider)
