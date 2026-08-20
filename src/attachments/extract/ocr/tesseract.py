"""Local tesseract OCR provider (no network). Handles raster images and scanned
(text-less) PDFs by rendering pages with poppler/pdf2image first.

Import-missing errors (pytesseract/PIL not installed) yield OCR_UNAVAILABLE so
the ChainedOcr can fall through to the next provider. Runtime failures (tesseract
binary crashed, corrupt image, etc.) yield ERROR — a distinct status per #37.
"""

from __future__ import annotations

import io

from src.attachments.extract.mime import is_pdf
from src.attachments.extract.ocr.base import OcrResult
from src.attachments.extract.ocr.pages import render_pdf_pages
from src.attachments.extract.result import Status

_NAME = "tesseract"


def _ok(text: str) -> OcrResult:
    return OcrResult(text, Status.EXTRACTED if text.strip() else Status.EMPTY, _NAME)


class TesseractOcr:
    def read(self, data: bytes, mime: str, filename: str) -> OcrResult:
        if is_pdf(mime, filename):
            return self._pdf(data)
        return self._image(data)

    def _image(self, data: bytes) -> OcrResult:
        try:
            import pytesseract
            from PIL import Image, ImageFile
        except Exception:
            return OcrResult("", Status.OCR_UNAVAILABLE, _NAME)
        # Real mail carries damaged images. PIL refuses a JPEG missing its last
        # few bytes outright, which threw away whole photographs over 20 unread
        # bytes; decoding what is there is strictly better than nothing, and the
        # missing tail is the bottom edge of the picture.
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        try:
            with Image.open(io.BytesIO(data)) as im:
                im.load()
                # Normalise the mode before OCR. pytesseract rejects some modes
                # outright with "Unsupported image format/type" — a 4032x3024
                # photo in this corpus failed that way and yielded 562
                # characters once converted.
                return _ok(pytesseract.image_to_string(im.convert("RGB")))
        except Exception:
            return OcrResult("", Status.ERROR, _NAME)

    def _pdf(self, data: bytes) -> OcrResult:
        try:
            import pdf2image  # noqa: F401 — probed here so a missing lib is UNAVAILABLE, not ERROR
            import pytesseract
        except Exception:
            return OcrResult("", Status.OCR_UNAVAILABLE, _NAME)
        try:
            text = "\n".join(pytesseract.image_to_string(im) for im in render_pdf_pages(data))
            return _ok(text)
        except Exception:
            return OcrResult("", Status.ERROR, _NAME)
