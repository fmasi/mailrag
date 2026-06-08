"""HTML handler: strip tags to visible text."""
from __future__ import annotations

from html.parser import HTMLParser

from src.attachments.extract.result import ExtractResult, Status, ok
from src.attachments.extract.handlers.plaintext import _decode


class _Stripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_data(self, data):
        self.parts.append(data)


class HtmlHandler:
    def can_handle(self, mime: str, filename: str) -> bool:
        m = (mime or "").lower()
        return m == "text/html" or (filename or "").lower().endswith((".html", ".htm"))

    def extract(self, data: bytes, mime: str, filename: str) -> ExtractResult:
        try:
            p = _Stripper()
            p.feed(_decode(data))
            return ok(" ".join(" ".join(p.parts).split()), "html")
        except Exception:
            return ExtractResult("", Status.ERROR, "html")
