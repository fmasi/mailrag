"""Final body normalization before a message is chunked and embedded.

Runs after HTML→text conversion and reply-chain stripping, and removes the
three kinds of content that survive those passes while carrying no retrievable
meaning:

* **base64 / data: blobs** — inline images and attachments that leaked into the
  text body, tens of KB each;
* **URL tracking parameters** — `utm_*`, `fbclid`, `mc_cid` and friends, which
  make thirty copies of one campaign URL look like thirty distinct strings;
* **signature blocks** — the RFC 3676 ``-- `` delimiter and everything after it.

This matters more here than in a store-and-search tool. mailrag chunks each
body and spends one LLM call per email, so a 30 KB base64 blob is not merely a
diluted vector: it is several junk chunks, real embedding compute, and burned
LLM tokens in the summarize pass. Tracking parameters cost twice over — they
defeat the exact-content chunk dedup in :mod:`src.data.dedup` (two newsletter
chunks differing only by ``?utm_campaign=`` both survive) and they pollute the
learned-sparse vocabulary.

----

Attribution
    The base64 detection strategy and the tracking-parameter key list are
    adapted from **msgvault** (https://github.com/kenn-io/msgvault),
    ``internal/vector/embed/preprocess.go``, MIT licensed,
    Copyright (c) 2025-2026 Wes McKinney. The two-threshold split for base64
    (see :data:`_BASE64_RUN` / :data:`_BASE64_RUN_WITH_SLASH`) is their insight
    and is the non-obvious part of getting this right. Reimplemented in Python
    here rather than copied; the accompanying MIT notice is reproduced in the
    repository ``NOTICE`` file.
"""

from __future__ import annotations

import re
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

# ---------------------------------------------------------------- base64 blobs

# A ``data:<mime>;base64,<payload>`` URI, typically an inline image that ended
# up in the text body. The content-type length cap is defensive; the payload
# match is greedy so the whole blob goes.
_DATA_URI = re.compile(r"(?i)data:[a-zA-Z0-9./+\-]{0,128};base64,[A-Za-z0-9+/]+={0,2}")

# Bare base64 (no ``data:`` prefix). Two patterns, not one, because ``/`` is in
# both the base64 alphabet AND every URL path — so a single threshold either
# eats URL paths or misses real base64 containing slashes:
#
#   * without ``/``: 200+ chars. Prose never produces runs that long, and every
#     ``/`` in a URL resets the run, so paths are structurally safe.
#   * with ``/``: 300+ chars. The higher bar is what protects long signed S3 /
#     CloudFront URLs, which almost always hit a ``.``, ``?``, ``&``, ``_``,
#     ``-`` or ``~`` before 300 unbroken base64-alphabet characters — while an
#     inline image runs to thousands.
#
# (Strategy adapted from msgvault; see the module docstring.)
_BASE64_RUN = re.compile(r"[A-Za-z0-9+]{200,}={0,2}")
_BASE64_RUN_WITH_SLASH = re.compile(r"[A-Za-z0-9+/]{300,}={0,2}")


def strip_base64_blobs(text: str) -> str:
    """Remove data: URIs and bare base64 payloads from *text*.

    Order matters: the ``data:`` form goes first so its ``;base64,`` prefix
    cannot be left behind as an orphan once the payload is gone.
    """
    if not text:
        return text
    text = _DATA_URI.sub(" ", text)
    text = _BASE64_RUN_WITH_SLASH.sub(" ", text)
    return _BASE64_RUN.sub(" ", text)


# ------------------------------------------------------------- tracking params

# Query keys that exist purely for analytics/attribution. Stripping them
# collapses many visually-distinct copies of one campaign URL into a single
# canonical string, which is what lets exact-content chunk dedup actually fire.
# Key list adapted from msgvault (see the module docstring). Compared
# lowercase — the lookup normalises the parameter name first.
_TRACKING_PARAMS = frozenset(
    {
        "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
        "utm_id", "utm_name", "utm_brand", "utm_social",
        "fbclid", "gclid", "dclid", "gbraid", "wbraid", "msclkid", "yclid", "twclid",
        "mc_cid", "mc_eid", "ml_subscriber",
        "_hsenc", "_hsmi", "hsctatracking",
        "vero_conv", "vero_id", "ck_subscriber_id",
        "_branch_match_id", "ref", "ref_src", "s_cid", "icid", "spm",
    }
)  # fmt: skip

# A URL runs to the next whitespace, quote or bracket. Deliberately a coarse
# seed — the real work is done by urlsplit below, not by this pattern.
_URL = re.compile(r"https?://[^\s\"'<>)]+")


def _clean_url(match: "re.Match[str]") -> str:
    url = match.group(0)
    # Trailing sentence punctuation is part of the prose, not the URL.
    trailing = ""
    while url and url[-1] in ".,;:!?":
        trailing = url[-1] + trailing
        url = url[:-1]
    try:
        parts = urlsplit(url)
    except ValueError:
        return match.group(0)
    if not parts.query:
        return url + trailing
    kept = [
        (k, v)
        for k, v in parse_qsl(parts.query, keep_blank_values=True)
        if k.lower() not in _TRACKING_PARAMS
    ]
    cleaned = urlunsplit(
        (parts.scheme, parts.netloc, parts.path, urlencode(kept, doseq=True), parts.fragment)
    )
    return cleaned + trailing


def strip_tracking_params(text: str) -> str:
    """Drop analytics query parameters from every URL in *text*.

    Only the listed keys are removed — an unknown parameter may well be
    meaningful (an order id, a document reference), and silently dropping it
    would make the mail less searchable, not more.
    """
    if not text or "http" not in text:
        return text
    return _URL.sub(_clean_url, text)


# ---------------------------------------------------------------- signatures

# RFC 3676 §4.3: a line containing exactly "-- " delimits the signature block.
# Tolerant of the trailing space being stripped in transit, which is common.
_SIGNATURE_DELIM = re.compile(r"\n-- ?\n")

# Below this many characters of surviving body, a signature strip is more
# likely to have eaten the message than cleaned it — a two-line "Thanks,\n--\nJ"
# reply is mostly signature, and keeping it whole beats indexing an empty body.
_MIN_BODY_AFTER_SIGNATURE = 40

# A real signature is small. These bound what may be REMOVED — measuring only
# what is kept lets a long tail after a stray "--" divider be deleted from a
# long message without tripping anything.
_MAX_SIGNATURE_CHARS = 600
_MAX_SIGNATURE_LINES = 12


def strip_signature_block(text: str) -> str:
    """Remove a trailing ``-- `` signature block — conservatively, and only once.

    Three conditions, all required, because this rule deletes user content:

    * there is **exactly one** ``-- `` delimiter. Two or more means at least one
      is being used as a divider, and no rule can tell which. Stripping the last
      one would also make the function non-idempotent: the earlier delimiter
      would become "the last one" on a second pass and take more of the body with
      it — which would change content hashes on re-index and churn the whole
      collection.
    * what would be removed looks like a signature — bounded in length and line
      count — rather than prose that happened to follow a divider;
    * what remains is still a usable body.

    The cost of this conservatism is that an email containing both a divider and
    a signature keeps its signature. That is the right trade: a stray signature
    is noise, a deleted paragraph is data loss.
    """
    if not text:
        return text
    delimiters = list(_SIGNATURE_DELIM.finditer(text))
    if len(delimiters) != 1:
        return text
    match = delimiters[0]
    removed = text[match.start() :]
    if len(removed) > _MAX_SIGNATURE_CHARS or removed.count("\n") > _MAX_SIGNATURE_LINES:
        return text
    stripped = text[: match.start()]
    if len(stripped.strip()) < _MIN_BODY_AFTER_SIGNATURE:
        return text
    return stripped


# ------------------------------------------------------------------ whitespace

_TRAILING_HWS = re.compile(r"(?m)[ \t]+$")
_MULTI_NEWLINE = re.compile(r"\n{3,}")
_HORIZONTAL_RUN = re.compile(r"[ \t]{2,}")


def normalize_whitespace(text: str) -> str:
    """Collapse the whitespace bloat that HTML→text conversion leaves behind.

    Trailing horizontal whitespace goes first, so a "blank" line that actually
    contains spaces collapses together with its neighbours. Three or more
    newlines become two — two preserves the paragraph break that carries
    structure; more is noise.
    """
    if not text:
        return text
    text = _TRAILING_HWS.sub("", text)
    text = _HORIZONTAL_RUN.sub(" ", text)
    return _MULTI_NEWLINE.sub("\n\n", text).strip()


def clean_body(text: str) -> str:
    """Full normalization pass, in the order the stages depend on.

    Blobs go before whitespace collapsing (so the space left behind is tidied
    up), and the signature strip runs before whitespace normalization so its
    ``\\n-- \\n`` anchor is still intact.
    """
    if not text:
        return text
    text = strip_base64_blobs(text)
    text = strip_tracking_params(text)
    text = strip_signature_block(text)
    return normalize_whitespace(text)
