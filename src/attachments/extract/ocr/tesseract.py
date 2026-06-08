"""Local tesseract OCR provider (no network). Handles raster images and scanned
(text-less) PDFs by rendering pages with poppler/pdf2image first.

Import-missing errors (pytesseract/PIL not installed) yield OCR_UNAVAILABLE so
the ChainedOcr can fall through to the next provider. Runtime failures (tesseract
binary crashed, corrupt image, etc.) yield ERROR — a distinct status per #37.
"""
from __future__ import annotations

import io

from src.attachments.extract.ocr.base import OcrResult
from src.attachments.extract.result import Status

_NAME = "tesseract"


def _ok(text: str) -> OcrResult:
    return OcrResult(text, Status.EXTRACTED if text.strip() else Status.EMPTY, _NAME)


class TesseractOcr:
    def read(self, data: bytes, mime: str, filename: str) -> OcrResult:
        name = (filename or "").lower()
        if mime == "application/pdf" or name.endswith(".pdf"):
            return self._pdf(data)
        return self._image(data)

    def _image(self, data: bytes) -> OcrResult:
        try:
            import pytesseract
            from PIL import Image
        except Exception:
            return OcrResult("", Status.OCR_UNAVAILABLE, _NAME)
        try:
            return _ok(pytesseract.image_to_string(Image.open(io.BytesIO(data))))
        except Exception:
            return OcrResult("", Status.ERROR, _NAME)

    def _pdf(self, data: bytes) -> OcrResult:
        try:
            import pdf2image
            import pytesseract
        except Exception:
            return OcrResult("", Status.OCR_UNAVAILABLE, _NAME)
        try:
            images = pdf2image.convert_from_bytes(data)
            text = "\n".join(pytesseract.image_to_string(im) for im in images)
            return _ok(text)
        except Exception:
            return OcrResult("", Status.ERROR, _NAME)
