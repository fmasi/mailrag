"""DOCX handler (python-docx)."""

from __future__ import annotations

import io

from src.attachments.extract.result import ExtractResult, Status, ok


class DocxHandler:
    def can_handle(self, mime: str, filename: str) -> bool:
        return (filename or "").lower().endswith(".docx") or "wordprocessingml" in (
            mime or ""
        ).lower()

    def extract(self, data: bytes, mime: str, filename: str) -> ExtractResult:
        try:
            import docx
        except Exception:
            return ExtractResult("", Status.BINARY, "docx")
        try:
            d = docx.Document(io.BytesIO(data))
            return ok("\n".join(p.text for p in d.paragraphs), "docx")
        except Exception:
            return ExtractResult("", Status.ERROR, "docx")
