"""HTML handler: strip tags to visible text."""

from __future__ import annotations

from html.parser import HTMLParser

from src.attachments.extract.handlers.plaintext import decode_text
from src.attachments.extract.mime import mime_base, mime_charset
from src.attachments.extract.result import ExtractResult, Status, ok


class _Stripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_data(self, data):
        self.parts.append(data)


class HtmlHandler:
    def can_handle(self, mime: str, filename: str) -> bool:
        return mime_base(mime) == "text/html" or (filename or "").lower().endswith(
            (".html", ".htm")
        )

    def extract(self, data: bytes, mime: str, filename: str) -> ExtractResult:
        try:
            p = _Stripper()
            p.feed(decode_text(data, mime_charset(mime)))
            return ok(" ".join(" ".join(p.parts).split()), "html")
        except Exception:
            return ExtractResult("", Status.ERROR, "html")
