"""Cheap, corpus-agnostic signals for judging whether an attachment is decoration.

The listing problem: 63% of attachment rows on a real corpus are signature
strips, newsletter headers and spacer pixels. Recurrence heuristics get most of
them, but they are guesses about content made from metadata, and they misfire in
both directions — a quarterly reporting-deadline table quoted into six threads
looks exactly like a logo to them.

What settles it is the text the image actually contains. A logo OCRs to a
company name or nothing; a table OCRs to dozens of words and numbers. That
observation is corpus-agnostic in a way the thresholds are not, which is why
this module **measures and records** rather than judging: the raw signals go in
the store, and the verdict is computed from them at read time. Re-calibrating
against a different corpus (personal mail behaves nothing like work mail — 7%
bulk vs 41%) then costs a SQL query rather than re-OCRing thousands of blobs.

Deliberately no LLM. The signals below are arithmetic over OCR output, so there
is no prompt to get subtly wrong per corpus — the failure mode that cost this
project 8.5 hours on the email-cleaning rubric.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

# Words of 2+ letters; digits counted separately because tabular content is
# digit-dense in a way decoration never is.
_WORD_RE = re.compile(r"[A-Za-z]{2,}")
_DIGIT_RE = re.compile(r"\d")

# The LLM vision provider answers in a fixed two-part shape (see
# ``ocr/llm_vision.py``): a DESCRIPTION preamble, then TEXT: and the
# transcription. Only the transcription is comparable to what tesseract
# returns — measured, the preamble inflates a three-word newsletter header from
# 22 characters to 159, which is enough to flip it from decoration to content.
# So strip it, and the signal means the same thing whichever engine produced it.
_LLM_TEXT_MARKER = re.compile(r"^\s*DESCRIPTION:.*?\bTEXT:\s*", re.IGNORECASE | re.DOTALL)

# Extractors whose output these thresholds have been calibrated against. Signals
# from anything else are recorded but not trusted to judge — better to fall back
# to the metadata heuristic than to apply a number tuned for a different engine.
CALIBRATED_EXTRACTORS = frozenset({"tesseract", "llm_vision", "llm"})


@dataclass(frozen=True)
class BlobSignals:
    """What was measured about one blob. Verdict-free by design."""

    chars: int
    words: int
    unique_words: int
    digits: int
    width: int
    height: int
    status: str
    extractor: str

    @property
    def pixels(self) -> int:
        return self.width * self.height


def transcription_only(text: str) -> str:
    """Drop an LLM vision DESCRIPTION preamble, keeping the transcribed text.

    A no-op for tesseract output, which has no preamble.
    """
    return _LLM_TEXT_MARKER.sub("", text or "", count=1)


def measure_text(text: str) -> tuple:
    """(chars, words, unique_words, digits) for extracted text.

    Measures the *transcription* — what the image actually says — so the numbers
    mean the same thing regardless of which OCR engine produced them.
    """
    stripped = transcription_only(text).strip()
    words = _WORD_RE.findall(stripped)
    return (
        len(stripped),
        len(words),
        len({w.lower() for w in words}),
        len(_DIGIT_RE.findall(stripped)),
    )


def image_dimensions(data: bytes) -> tuple:
    """(width, height) of an image blob, or (0, 0) when it cannot be read.

    Dimensions are recorded but deliberately NOT used to judge: "banner-shaped
    means signature" was tested against this corpus and failed — a 2475x383
    strip turned out to be a product-lifecycle table with end-of-life dates.
    They are kept because they are free at measurement time and may inform a
    future rule.
    """
    try:
        import io

        from PIL import Image

        with Image.open(io.BytesIO(data)) as im:
            return im.size
    except Exception:
        return (0, 0)


def measure_blob(data: bytes, mime: str, filename: str, extractor) -> BlobSignals:
    """Run extraction over one blob and reduce it to signals.

    ``extractor`` is the usual :class:`Extractor` facade, so images go through
    the same OCR chain used by ``get_attachment`` — one code path, one set of
    failure modes.
    """
    result = extractor.extract(data, mime, filename)
    chars, words, uwords, digits = measure_text(result.text)
    width, height = image_dimensions(data) if mime.startswith("image/") else (0, 0)
    return BlobSignals(
        chars=chars,
        words=words,
        unique_words=uwords,
        digits=digits,
        width=width,
        height=height,
        status=result.status,
        extractor=result.extractor,
    )


# Verdict thresholds — the corpus-specific half, kept apart from the measuring
# above so they can be retuned without re-running OCR.
#
# Calibrated on the work corpus (2026-08-19) against images labelled by eye:
#   * newsletter header, 52 threads, 22 chars              -> decoration
#   * signature strips / spacers, 0 chars                  -> decoration
#   * quarterly reporting-deadline table, 209 chars        -> CONTENT, and the
#     recurrence heuristic had it wrong (small, in 6 threads because it is a
#     useful reference people quote)
#   * "401 Access is denied" screenshot, 29 chars, 1 thread -> CONTENT. Text-poor
#     is NOT decoration: someone pasted it to report a problem. Only the
#     combination of text-poor AND reuse across unrelated threads is.
TEXT_RICH_CHARS = 100  # at or above this it carries information...
TEXT_POOR_CHARS = 30  # below this it says nothing on its own — recurrence decides
DECOR_MIN_THREADS = 5
# ...unless it is ubiquitous. Text-richness alone wrongly rescued a legal
# confidentiality disclaimer rendered as an image (748 chars, 829 threads) and
# signature blocks carrying a name, title and phone numbers (195 chars, 61
# threads). Nothing genuine appears in that many unrelated conversations.
#
# The 15-25 band is genuinely ambiguous on recurrence alone: a real quarterly
# reporting-deadline table sits at 15 threads and a disclaimer at 18. This cut
# errs toward keeping, so a handful of signature images survive rather than one
# real table being hidden. Separating that band properly needs a content rule
# (disclaimer phrasing, contact-detail patterns), not a bigger threshold.
UBIQUITOUS_THREADS = 20


def is_decoration(
    signals: Optional[BlobSignals], thread_count: int, inline: bool
) -> Optional[bool]:
    """Verdict from measured signals, or ``None`` when they cannot decide.

    ``None`` means "no opinion" — the caller falls back to its recurrence
    heuristic. That keeps an unmeasured or unreadable blob behaving exactly as
    it does today rather than silently changing category.

    The rule is deliberately asymmetric: text-rich content is rescued
    regardless of how often it recurs (that is the heuristic's known failure
    mode), while calling something decoration requires BOTH that it says
    nothing and that unrelated conversations reuse it.
    """
    if signals is None or signals.status not in ("extracted", "empty"):
        return None
    if signals.extractor not in CALIBRATED_EXTRACTORS:
        return None
    if inline and thread_count >= UBIQUITOUS_THREADS:
        return True
    if signals.chars >= TEXT_RICH_CHARS:
        return False
    if not inline:
        return False
    if signals.chars < TEXT_POOR_CHARS and thread_count >= DECOR_MIN_THREADS:
        return True
    return None
