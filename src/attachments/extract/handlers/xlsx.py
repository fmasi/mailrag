"""XLSX handler (openpyxl)."""
from __future__ import annotations

import io

from src.attachments.extract.result import ExtractResult, Status, ok


class XlsxHandler:
    def can_handle(self, mime: str, filename: str) -> bool:
        return (filename or "").lower().endswith(".xlsx") or "spreadsheetml" in (mime or "").lower()

    def extract(self, data: bytes, mime: str, filename: str) -> ExtractResult:
        try:
            import openpyxl
        except Exception:
            return ExtractResult("", Status.BINARY, "xlsx")
        try:
            wb = openpyxl.load_workbook(io.BytesIO(data), read_only=True, data_only=True)
            out = []
            for ws in wb.worksheets:
                for row in ws.iter_rows(values_only=True):
                    out.append("\t".join("" if c is None else str(c) for c in row))
            return ok("\n".join(out), "xlsx")
        except Exception:
            return ExtractResult("", Status.ERROR, "xlsx")
