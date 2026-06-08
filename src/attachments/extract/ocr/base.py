"""OCR/vision provider protocol and a fallback composition.

The one runtime-swappable axis of extraction: turn image bytes (or a scanned PDF)
into text. Implementations: TesseractOcr, LlmVision, (future) cloud. Fallback is
expressed by composing providers in a ChainedOcr, not by nested try/except.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Protocol

from src.attachments.extract.result import Status


@dataclass
class OcrResult:
    text: str
    status: str        # extracted | empty | ocr_unavailable | error
    provider: str


class OcrProvider(Protocol):
    def read(self, data: bytes, mime: str, filename: str) -> OcrResult: ...


# Statuses that mean "this provider produced nothing usable; try the next one".
_FALL_THROUGH = {Status.OCR_UNAVAILABLE, Status.EMPTY, Status.ERROR}


class ChainedOcr:
    """Try each provider in order; return the first usable result.

    Falls through on ocr_unavailable / empty / error. If every provider falls
    through, returns the last provider's result, or an ocr_unavailable sentinel
    when the chain is empty.
    """

    def __init__(self, providers: List[OcrProvider]):
        self._providers = list(providers)

    def read(self, data: bytes, mime: str, filename: str) -> OcrResult:
        last = OcrResult("", Status.OCR_UNAVAILABLE, "none")
        for p in self._providers:
            last = p.read(data, mime, filename)
            if last.status not in _FALL_THROUGH:
                return last
        return last
