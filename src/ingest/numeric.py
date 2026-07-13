"""Normalise exact numeric surface forms so the sparse/dense legs can match them.

Motivation (issue #82). bge-m3's learned-sparse tokeniser carries almost no signal
for bare numbers, and the many *surface forms* of one figure never collide:
``$210,000,000``, ``210,000,000``, ``210M`` and ``210 million`` all tokenise
differently, so a query for one form does not retrieve a document that spells the
figure another way.

The tractable fix applied here is **surface-form normalisation**: for any currency
amount or magnitude-suffixed number in a piece of text we append a canonical
integer token (``210000000``) to the text *before* it is embedded. Because the same
augmentation runs at index time (on the chunk) and query time (on the query
string), the canonical token lands in the same token-id vocabulary on both sides
and gives the sparse leg a near-exact hit, while the original surface form is left
untouched so ordinary lexical matches still work.

What this does NOT do (documented on #82): it does not teach the tokeniser that
``200M`` is *close to* ``210M`` — only exact-figure recall is improved. Fuzzy /
range numeric matching and a raw-corpus ``grep_email`` escape hatch remain open on
#82.
"""

from __future__ import annotations

import re
from typing import List

# A number with optional thousands separators and an optional decimal part,
# optionally followed by a magnitude word/suffix. Currency symbols are matched
# separately so a bare "210M" is normalised too.
_MAGNITUDES = {
    "k": 1_000,
    "thousand": 1_000,
    "m": 1_000_000,
    "mm": 1_000_000,  # finance shorthand for "million"
    "mn": 1_000_000,
    "million": 1_000_000,
    "b": 1_000_000_000,
    "bn": 1_000_000_000,
    "billion": 1_000_000_000,
    "t": 1_000_000_000_000,
    "tn": 1_000_000_000_000,
    "trillion": 1_000_000_000_000,
}

# Group 1: the numeric core (digits, optional thousands commas, optional decimal).
# Group 2: an optional magnitude suffix (letters) — a space between the number and
# a word suffix ("210 million") is allowed; a glued suffix ("210M") is also allowed.
_NUMBER_RE = re.compile(
    r"(?<![\w.])"  # not preceded by a word char or a dot (avoid version strings)
    r"(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?"  # 1,234,567 or 1234567, optional .dd
    r"(?:\s*([A-Za-z]{1,8}))?",  # optional magnitude suffix / word
)


def _canonical_int(core: str, frac: str | None, suffix: str | None) -> str | None:
    """Return the canonical integer string for one matched number, or None.

    ``core`` is the integer part with any thousands commas; ``frac`` is the
    fractional digits (without the dot) or None; ``suffix`` is a trailing
    magnitude word/letter or None. Returns None when the value is not an exact
    integer (e.g. a decimal with no magnitude to scale it up).
    """
    digits = core.replace(",", "")
    if not digits.isdigit():
        return None
    value = float(digits)
    if frac:
        value += float(f"0.{frac}")
    mult = 1
    if suffix is not None:
        key = suffix.strip().lower()
        # A trailing word that is not a magnitude (e.g. "people") is not part of
        # the figure — ignore it and normalise the number on its own merits.
        if key in _MAGNITUDES:
            mult = _MAGNITUDES[key]
        else:
            suffix = None  # treat as a plain number followed by an unrelated word
    scaled = value * mult
    if scaled != int(scaled):
        return None
    return str(int(scaled))


def normalized_numeric_tokens(text: str) -> List[str]:
    """Return the de-duplicated canonical integer tokens found in *text*.

    Preserves first-seen order. Only produces a token for numbers with either a
    thousands separator, a decimal, or a magnitude suffix (a bare ``42`` is left
    alone so we do not flood the vocabulary with every stray integer)."""
    seen: dict[str, None] = {}
    for m in _NUMBER_RE.finditer(text):
        core, frac, suffix = m.group(1), m.group(2), m.group(3)
        is_magnitude = suffix is not None and suffix.strip().lower() in _MAGNITUDES
        # Only normalise numbers that carry a *marker*: a thousands separator, a
        # decimal, or a real magnitude suffix. A bare "42" (even followed by an
        # unrelated word like "people") is left alone to avoid vocab flooding.
        has_marker = ("," in core) or bool(frac) or is_magnitude
        if not has_marker:
            continue
        canon = _canonical_int(core, frac, suffix)
        if canon is None:
            continue
        # Skip the no-op case where the canonical form already equals the surface
        # form (a plain comma-free integer with no scaling) — nothing to add.
        if canon == core.replace(",", "") and "," not in core and not frac and not is_magnitude:
            continue
        seen.setdefault(canon, None)
    return list(seen.keys())


def augment_numeric(text: str) -> str:
    """Append canonical numeric tokens to *text* for embedding.

    Idempotent-ish: appends a single space-joined block of canonical integers so
    the sparse/dense legs gain an exact-figure token without disturbing the
    original surface text. Returns *text* unchanged when no normalisable number is
    present.
    """
    tokens = normalized_numeric_tokens(text)
    if not tokens:
        return text
    return f"{text} {' '.join(tokens)}"
