"""Handler protocol: each content-type extractor declares what it handles and how."""

from __future__ import annotations

from typing import Protocol

from src.attachments.extract.result import ExtractResult


class Handler(Protocol):
    def can_handle(self, mime: str, filename: str) -> bool: ...
    def extract(self, data: bytes, mime: str, filename: str) -> ExtractResult: ...
