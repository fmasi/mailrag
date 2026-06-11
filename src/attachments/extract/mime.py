"""Small content-type helpers shared by handlers and OCR providers.

Stored mimes may carry parameters (``text/plain; charset=iso-8859-1`` — the ingester
preserves the declared charset for text parts), so anything matching on a content
type must compare against the parameter-stripped base.
"""
from __future__ import annotations

from typing import Optional


def mime_base(mime: Optional[str]) -> str:
    """The lowercased base content type, with any ;-parameters stripped."""
    return (mime or "").split(";", 1)[0].strip().lower()


def mime_charset(mime: Optional[str]) -> Optional[str]:
    """The charset= parameter of a content type (lowercased), or None."""
    for seg in (mime or "").split(";")[1:]:
        key, _, val = seg.partition("=")
        if key.strip().lower() == "charset":
            val = val.strip().strip('"').strip("'").lower()
            return val or None
    return None


def is_pdf(mime: Optional[str], filename: Optional[str]) -> bool:
    """The one predicate for "this blob is a PDF" (by mime or .pdf suffix)."""
    return mime_base(mime) == "application/pdf" or (filename or "").lower().endswith(".pdf")
