"""Plain-text family: text/plain, text/csv, text/calendar (.txt/.csv/.ics)."""
from __future__ import annotations

from src.attachments.extract.mime import mime_base, mime_charset
from src.attachments.extract.result import ExtractResult, Status, ok

_MIMES = {"text/plain", "text/csv", "text/calendar"}
_EXTS = (".txt", ".csv", ".ics")


def decode_text(data: bytes, charset: str | None = None) -> str:
    """Decode bytes to text: the declared charset first (when given and valid),
    then utf-8, then latin-1 (which maps every byte, so this never raises)."""
    if charset:
        try:
            return data.decode(charset)
        except (UnicodeDecodeError, LookupError):
            pass   # mislabeled or unknown charset -> fall through
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("latin-1")


class PlaintextHandler:
    def can_handle(self, mime: str, filename: str) -> bool:
        return mime_base(mime) in _MIMES or (filename or "").lower().endswith(_EXTS)

    def extract(self, data: bytes, mime: str, filename: str) -> ExtractResult:
        try:
            return ok(decode_text(data, mime_charset(mime)), "plaintext")
        except Exception:
            return ExtractResult("", Status.ERROR, "plaintext")
