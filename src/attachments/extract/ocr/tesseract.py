"""Local tesseract OCR provider (no network). Handles raster images and scanned
(text-less) PDFs by rendering pages with poppler/pdf2image first.

A *missing engine* yields OCR_UNAVAILABLE so the ChainedOcr can fall through to
the next provider — whether it is missing as a Python import (pytesseract/PIL not
installed) or as a binary (``tesseract``/``poppler`` absent from PATH). Genuine
runtime failures (tesseract crashed, corrupt image) yield ERROR — a distinct
status per #37.

The binary case cannot be caught by the import probe: ``pytesseract`` imports
happily without ``tesseract`` and only fails when it shells out. Reporting that
as ERROR is not cosmetic — :class:`AttachmentStore` deliberately does not cache
OCR_UNAVAILABLE so that a later run with a working PATH retries, while ERROR is
cached permanently. A scheduled sync inherits no PATH, so misclassifying here
froze "unreadable" onto attachments that were merely unread.

Nor can the binary case be caught by exception type alone: poppler is missing in
*halves*, and pdf2image names an exception for only one of them. See
:data:`_POPPLER_BINARIES`.
"""

from __future__ import annotations

import io
import os

from src.attachments.extract.mime import is_pdf
from src.attachments.extract.ocr.base import OcrResult
from src.attachments.extract.ocr.pages import render_pdf_pages
from src.attachments.extract.result import Status

_NAME = "tesseract"

# Matched by NAME, not by class, for two reasons. ``TesseractNotFoundError``
# subclasses ``OSError`` — and so does Pillow's "image file is truncated" — so
# the type hierarchy cannot tell a missing engine from a damaged image. And the
# tests inject a MagicMock in place of ``pytesseract``, which makes
# ``except pytesseract.TesseractNotFoundError`` a TypeError rather than a match.
_ENGINE_MISSING = frozenset(
    {
        "TesseractNotFoundError",  # pytesseract: no `tesseract` on PATH
        "PDFInfoNotInstalledError",  # pdf2image: no `pdfinfo` (poppler)
        "PopplerNotInstalledError",  # pdf2image: no `pdftoppm` (poppler)
    }
)

# Not every missing binary gets a named exception. pdf2image guards only its
# `pdfinfo` call (OSError -> PDFInfoNotInstalledError); the `pdftoppm`/
# `pdftocairo` version probe and the render itself are unguarded `Popen` calls.
# So a PARTIAL poppler install — pdfinfo present, pdftoppm absent, the shape a
# minimal container or a half-finished `brew install` produces — surfaces as a
# bare FileNotFoundError that no exception NAME can identify.
_POPPLER_BINARIES = frozenset({"pdfinfo", "pdftoppm", "pdftocairo"})


def _is_missing_binary(exc: BaseException) -> bool:
    """True when *exc* is ``Popen`` failing to exec one of poppler's binaries.

    Matched on the *executable* that failed, not on the exception type: a bare
    FileNotFoundError is equally what an unreadable input file raises, and
    calling that "environment issue, retry later" would be the same
    misclassification in reverse — the attachment would be re-OCR'd forever and
    never settle. ``Popen`` sets ``filename`` to argv[0], which is the bare
    command name, or an absolute path when ``poppler_path`` is configured.
    """
    name = getattr(exc, "filename", None)
    return (
        isinstance(exc, FileNotFoundError)
        and bool(name)
        and (os.path.basename(str(name)) in _POPPLER_BINARIES)
    )


def _is_engine_missing(exc: BaseException) -> bool:
    """True when *exc* means "the OCR engine is not installed here".

    Walks the ``__cause__``/``__context__`` chain so a wrapped error still counts.
    """
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if type(current).__name__ in _ENGINE_MISSING or _is_missing_binary(current):
            return True
        current = current.__cause__ or current.__context__
    return False


def _failed(exc: BaseException) -> OcrResult:
    """Classify a failure: a missing engine is UNAVAILABLE, anything else ERROR."""
    status = Status.OCR_UNAVAILABLE if _is_engine_missing(exc) else Status.ERROR
    return OcrResult("", status, _NAME)


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
        except Exception as exc:
            return _failed(exc)

    def _pdf(self, data: bytes) -> OcrResult:
        try:
            import pdf2image  # noqa: F401 — probed here so a missing lib is UNAVAILABLE, not ERROR
            import pytesseract
        except Exception:
            return OcrResult("", Status.OCR_UNAVAILABLE, _NAME)
        try:
            text = "\n".join(pytesseract.image_to_string(im) for im in render_pdf_pages(data))
            return _ok(text)
        except Exception as exc:
            return _failed(exc)
