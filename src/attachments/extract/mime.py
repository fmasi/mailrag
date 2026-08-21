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


# Extensions that mean "image" when the declared mime does not. Senders and
# exporters label attachments application/octet-stream routinely, and a photo of
# a whiteboard is exactly the content OCR exists for.
_IMAGE_SUFFIXES = (
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
    ".heic",
    ".heif",
)


def is_image(mime: Optional[str], filename: Optional[str]) -> bool:
    """The one predicate for "this blob is an image" (by mime or suffix).

    Mirrors :func:`is_pdf`. Images used to be routed by mime alone, so anything
    declared ``application/octet-stream`` never reached OCR and came back
    ``unsupported`` — while a PDF in the same situation extracted fine, because
    that predicate already had the filename fallback. Same problem, opposite
    outcome, for no reason other than which predicate was written first.
    """
    return mime_base(mime).startswith("image/") or (filename or "").lower().endswith(
        _IMAGE_SUFFIXES
    )


def is_pdf(mime: Optional[str], filename: Optional[str]) -> bool:
    """The one predicate for "this blob is a PDF" (by mime or .pdf suffix)."""
    return mime_base(mime) == "application/pdf" or (filename or "").lower().endswith(".pdf")
