"""Attachment text-extraction subsystem (clean handler + OCR-provider design)."""

from src.attachments.extract.ocr.registry import default_extractor_name, resolve
from src.attachments.extract.registry import Extractor
from src.attachments.extract.result import ExtractResult, Status


def build_default_extractor(extractor_name: str | None = None) -> Extractor:
    """Build an Extractor whose OCR provider is the configured (or named) backend."""
    return Extractor(resolve(extractor_name or default_extractor_name()))


__all__ = [
    "ExtractResult",
    "Status",
    "Extractor",
    "resolve",
    "default_extractor_name",
    "build_default_extractor",
]
