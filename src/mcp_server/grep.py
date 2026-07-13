"""Literal / regex search over the raw email corpus — the no-embeddings escape hatch.

``grep_email`` walks a directory of ``.eml`` files, decodes each message's body
(honouring ``Content-Transfer-Encoding``: quoted-printable + base64, and stripping
HTML), and matches ``pattern`` line-by-line. It bypasses Qdrant and embeddings
entirely, so needle hunts — a number, an ID, an email address, an error string —
hit exactly where dense/hybrid retrieval is blind (issue #82).

This module is deliberately **self-contained**: it decodes MIME parts with the
Python stdlib ``email`` package and an ``html.parser`` HTML-to-text pass, rather
than importing the ingest loader — the loader is owned by another change and must
not be modified. Decoding here mirrors the loader's approach (prefer ``text/plain``,
fall back to stripped ``text/html``) closely enough for literal matching.

**Scope note:** grep covers the message *envelope* (subject/from/to/date) and the
decoded *body* text only. It does **not** yet read attachment bytes — spreadsheet
/ PDF / doc contents are out of scope here (that is attachment indexing, issue #80).
The result flags this via ``attachment_names`` so a caller knows an attachment
exists even though its cell/text contents were not searched.
"""

from __future__ import annotations

import email
import email.header
import os
import re
from email import policy
from html.parser import HTMLParser
from typing import Any, Dict, List, Optional

# Default corpus root when neither an explicit ``root`` arg nor an env override is
# given. ``$MAILRAG_EML_ROOT`` is the canonical override (the directory of raw
# ``.eml`` files that was onboarded); ``~/rag_eml`` is the conventional default.
DEFAULT_EML_ROOT = "~/rag_eml"

# Hard caps so a broad pattern can never produce an unbounded payload (the same
# discipline applied to search_email in issue #84).
_DEFAULT_MAX_MATCHES = 50
_HARD_MAX_MATCHES = 500
_SNIPPET_CHARS = 200  # chars of context kept around each matched line


class _HTMLTextExtractor(HTMLParser):
    """Minimal HTML→text: drop script/style, keep readable prose.

    Self-contained (mirrors the ingest loader's extractor) so grep does not
    depend on loader internals.
    """

    _SKIP_TAGS = frozenset({"style", "script", "head"})
    _BLOCK_TAGS = frozenset({"p", "br", "div", "tr", "li", "h1", "h2", "h3", "h4"})
    _ENTITIES = {"nbsp": " ", "amp": "&", "lt": "<", "gt": ">", "quot": '"'}

    def __init__(self) -> None:
        super().__init__()
        self._parts: List[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag in self._SKIP_TAGS:
            self._skip_depth += 1
        elif tag in self._BLOCK_TAGS and not self._skip_depth:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP_TAGS and self._skip_depth:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self._skip_depth:
            self._parts.append(data)

    def handle_entityref(self, name: str) -> None:
        if not self._skip_depth:
            self._parts.append(self._ENTITIES.get(name, ""))

    def get_text(self) -> str:
        return re.sub(r"\n{3,}", "\n\n", "".join(self._parts)).strip()


def resolve_eml_root(root: Optional[str] = None) -> str:
    """Resolve the raw-``.eml`` corpus root grep searches over.

    Precedence: explicit ``root`` arg > ``$MAILRAG_EML_ROOT`` > ``~/rag_eml``.
    ``~`` is expanded. Raises ``ValueError`` if the resolved path is not a
    directory, so the caller sees a clear error rather than an empty result set.
    """
    raw = root or os.environ.get("MAILRAG_EML_ROOT") or DEFAULT_EML_ROOT
    path = os.path.expanduser(raw)
    if not os.path.isdir(path):
        raise ValueError(
            f"raw email corpus not found at {path!r}: set MAILRAG_EML_ROOT to the "
            "directory of .eml files, or pass an explicit root"
        )
    return path


def _decode_header(raw: Optional[str]) -> str:
    """Decode an RFC 2047 header to a plain Unicode string ('' when absent)."""
    if not raw:
        return ""
    try:
        parts = email.header.decode_header(raw)
        out = []
        for text, enc in parts:
            if isinstance(text, bytes):
                out.append(text.decode(enc or "utf-8", errors="replace"))
            else:
                out.append(text)
        return "".join(out).strip()
    except Exception:
        return str(raw).strip()


def _decode_part(part: email.message.Message) -> str:
    """Decode one leaf part's payload (QP/base64 handled by ``decode=True``)."""
    raw = part.get_payload(decode=True)
    if not isinstance(raw, (bytes, bytearray)):
        return ""
    charset = part.get_content_charset() or "utf-8"
    try:
        return bytes(raw).decode(charset, errors="replace")
    except (LookupError, UnicodeDecodeError):
        return bytes(raw).decode("utf-8", errors="replace")


def _decode_body(msg: email.message.Message) -> str:
    """Extract decoded body text: prefer ``text/plain``, fall back to HTML→text.

    Honours ``Content-Transfer-Encoding`` per part (via ``get_payload(decode=True)``)
    and deduplicates ``multipart/alternative`` by preferring the plain part.
    """
    plain: List[str] = []
    html: List[str] = []
    if msg.is_multipart():
        for part in msg.walk():
            if part.is_multipart():
                continue
            ctype = part.get_content_type()
            if ctype == "text/plain":
                plain.append(_decode_part(part))
            elif ctype == "text/html":
                html.append(_decode_part(part))
    else:
        decoded = _decode_part(msg)
        if msg.get_content_type() == "text/plain":
            plain.append(decoded)
        else:
            html.append(decoded)

    if plain:
        return "\n".join(plain).strip()
    extractor = _HTMLTextExtractor()
    extractor.feed("\n".join(html))
    return extractor.get_text()


def _attachment_names(msg: email.message.Message) -> List[str]:
    """Filenames of non-inline attachment parts (names only — bytes not read)."""
    names: List[str] = []
    if not msg.is_multipart():
        return names
    for part in msg.walk():
        if part.is_multipart():
            continue
        disp = (part.get_content_disposition() or "").lower()
        filename = part.get_filename()
        if filename and disp != "inline":
            names.append(_decode_header(filename))
    return names


def _discover_eml(root: str):
    """Yield every ``.eml`` path under ``root`` (recursive), in sorted order."""
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in sorted(filenames):
            if name.lower().endswith(".eml"):
                yield os.path.join(dirpath, name)


def _parse(path: str):
    """Parse one ``.eml`` into ``(meta, body)`` or ``None`` when unreadable."""
    try:
        with open(path, "rb") as fh:
            raw = fh.read()
        msg = email.message_from_bytes(raw, policy=policy.compat32)
    except Exception:
        return None
    meta = {
        "subject": _decode_header(msg.get("Subject")) or "(no subject)",
        "from": _decode_header(msg.get("From")),
        "to": _decode_header(msg.get("To")),
        "date": (msg.get("Date") or "").strip(),
        "message_id": " ".join(str(msg.get("Message-ID") or "").split()),
        "attachment_names": _attachment_names(msg),
        "path": path,
    }
    return meta, _decode_body(msg)


def _compile(pattern: str, regex: bool) -> re.Pattern:
    """Compile ``pattern`` (literal by default) with a clear error on bad regex."""
    flags = re.IGNORECASE
    if regex:
        try:
            return re.compile(pattern, flags)
        except re.error as exc:
            raise ValueError(f"invalid regex pattern {pattern!r}: {exc}") from exc
    return re.compile(re.escape(pattern), flags)


def _line_matches(body: str, rx: re.Pattern) -> List[str]:
    """Return matched lines from ``body`` as bounded snippets (± context)."""
    out: List[str] = []
    for line in body.splitlines():
        m = rx.search(line)
        if not m:
            continue
        snippet = line.strip()
        if len(snippet) > _SNIPPET_CHARS:
            # Centre the window on the match so the needle stays visible.
            start = max(0, m.start() - _SNIPPET_CHARS // 2)
            snippet = ("…" if start else "") + snippet[start : start + _SNIPPET_CHARS] + "…"
        out.append(snippet)
    return out


def grep_email(
    pattern: str,
    collection: Optional[str] = None,
    max_matches: int = _DEFAULT_MAX_MATCHES,
    regex: bool = False,
    *,
    root: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Literal / regex search over the raw email corpus (no embeddings).

    Walks the raw ``.eml`` corpus, decodes each body (QP/base64 + HTML→text),
    and returns one row per **matching message** with the matched line snippets
    and message metadata. This is the escape hatch for exact needle hunts where
    dense/hybrid retrieval is blind to numerals and identifiers (issue #82).

    Args:
        pattern: The string (or regex, when ``regex=True``) to find. Matching is
            case-insensitive and covers subject + decoded body text.
        collection: Accepted for API symmetry with the other MCP tools; grep is
            corpus-directory based, so it is used only to scope the root when a
            per-collection mapping is wired up (currently a no-op label).
        max_matches: Maximum matching **messages** to return. Clamped to
            ``[1, 500]`` (hard cap) so a broad pattern can never flood output.
        regex: When true, ``pattern`` is a Python regex; otherwise it is matched
            literally (special characters are escaped).
        root: Explicit corpus root (tests/advanced use); defaults to
            ``$MAILRAG_EML_ROOT`` or ``~/rag_eml``.

    Returns:
        Up to ``max_matches`` rows, each
        ``{subject, from, to, date, message_id, attachment_names, matches, path}``
        where ``matches`` is a list of matched-line snippets. ``attachment_names``
        lists attachment filenames present on the message; their *contents* are
        NOT searched (attachment indexing is issue #80).

    Raises:
        ValueError: on a blank ``pattern``, an invalid regex, or a missing corpus.
    """
    if not pattern or not pattern.strip():
        raise ValueError("pattern must be a non-empty string")
    limit = max(1, min(int(max_matches), _HARD_MAX_MATCHES))
    rx = _compile(pattern, regex)
    corpus = resolve_eml_root(root)

    results: List[Dict[str, Any]] = []
    for path in _discover_eml(corpus):
        parsed = _parse(path)
        if parsed is None:
            continue
        meta, body = parsed
        # Match against subject too, so a subject-only hit still surfaces.
        haystack_extra = meta["subject"]
        matches = _line_matches(body, rx)
        if not matches and rx.search(haystack_extra):
            matches = [f"[subject] {haystack_extra.strip()}"]
        if not matches:
            continue
        row = dict(meta)
        row["matches"] = matches[:20]  # cap per-message snippets too
        results.append(row)
        if len(results) >= limit:
            break
    return results
