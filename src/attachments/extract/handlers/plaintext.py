"""Plain-text family: text/plain, text/csv, text/calendar (.txt/.csv/.ics)."""
from __future__ import annotations

from src.attachments.extract.result import ExtractResult, Status, ok

_MIMES = {"text/plain", "text/csv", "text/calendar"}
_EXTS = (".txt", ".csv", ".ics")


def _decode(data: bytes) -> str:
    """Decode bytes to text, preferring utf-8; fall back to latin-1 (which maps every
    byte, so it never raises)."""
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("latin-1")


class PlaintextHandler:
    def can_handle(self, mime: str, filename: str) -> bool:
        m = (mime or "").lower()
        return m in _MIMES or (filename or "").lower().endswith(_EXTS)

    def extract(self, data: bytes, mime: str, filename: str) -> ExtractResult:
        try:
            return ok(_decode(data), "plaintext")
        except Exception:
            return ExtractResult("", Status.ERROR, "plaintext")
