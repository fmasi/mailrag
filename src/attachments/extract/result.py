"""Extraction result + status vocabulary, shared by every handler and OCR provider."""

from __future__ import annotations

from dataclasses import dataclass


class Status:
    EXTRACTED = "extracted"  # ran and produced text
    EMPTY = "empty"  # ran successfully, no text found
    BINARY = "binary"  # a handler matched the type but its parsing library is unavailable
    UNSUPPORTED = "unsupported"  # no handler matched the content type
    OCR_UNAVAILABLE = "ocr_unavailable"  # OCR chain had no available engine
    ERROR = "error"  # a handler/provider ran and raised at runtime


@dataclass(frozen=True)
class ExtractResult:
    text: str
    status: str
    extractor: str


def ok(text: str, extractor: str) -> ExtractResult:
    """Build an EXTRACTED/EMPTY result based on whether there is non-blank text."""
    return ExtractResult(
        text=text, status=Status.EXTRACTED if text.strip() else Status.EMPTY, extractor=extractor
    )
