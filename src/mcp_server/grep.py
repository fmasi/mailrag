"""Literal / regex search over the raw email corpus — the no-embeddings escape hatch.

``grep_email`` walks a directory of ``.eml`` files, decodes each message's body
(honouring ``Content-Transfer-Encoding``: quoted-printable + base64, and stripping
HTML), and matches ``pattern`` line-by-line. It bypasses Qdrant and embeddings
entirely, so needle hunts — a number, an ID, an email address, an error string —
hit exactly where dense/hybrid retrieval is blind (issue #82).

Decoding mirrors the loader's approach (prefer ``text/plain``, fall back to
stripped ``text/html``). It shares exactly one thing with the loader, and must:
:meth:`MailArchiveXLoader._strip_mbox_preamble`. Exports in this corpus prepend
an mbox ``From `` separator and a stray byte-count line before the RFC 2822
headers, and Python's parser stops reading headers at the first line that is not
``Name: value`` — so without stripping, EVERY message parses as headerless. This
module previously kept its own copy of the parse for isolation reasons that no
longer hold, and the result was that grep reported ``(no subject)`` with empty
sender/date/message-id for 100% of a real corpus while its synthetic tests
(preamble-free, hand-written) passed. One parser, one bug to fix.

**Bounded by design:** the scan is capped by matches, by files and by wall
clock, and every result reports ``scanned`` / ``corpus_files`` / ``complete`` so
a caller can tell "this needle is not in the corpus" apart from "the scan ran
out of budget before it got there". See :func:`grep_email` for the cost model.

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
import time
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

# Work bounds. The match cap alone does NOT bound the scan: it stops the walk
# early only when the pattern actually *hits*. A pattern that matches rarely --
# or not at all, which is exactly the shape of a "does X appear anywhere?"
# existence check -- walks every file in the corpus. On a real personal corpus
# (tens of thousands of messages, several GB) that is minutes of CPU at best,
# and it once ran a caller into a 30-minute MCP client timeout with nothing to
# show for it. So the scan is also bounded by wall-clock and by file count, and
# the result reports how far it actually got -- see ``complete`` below.
_DEFAULT_MAX_SECONDS = 60.0
_HARD_MAX_SECONDS = 900.0


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


def _scoped_files(collection: Optional[str], root: Optional[str]):
    """Files belonging to ``collection``, or ``None`` to walk the whole root.

    Corpora on one machine share a root and differ only by selection rules, so an
    unscoped walk crosses them: a work-scoped session greps a string and reads
    personal mail. When a collection is named we walk exactly the files its
    profile selects — the same set indexing used.

    Refusing matters as much as scoping. If a collection is named but no profile
    claims it, falling back to the full root would quietly do the very thing the
    caller asked to avoid, so this raises instead. An explicit ``root`` is an
    override and skips all of it.
    """
    if root or not collection:
        return None
    from src.mcp_server.scoping import collection_profiles, files_for_collection

    files = files_for_collection(collection)
    if files is None:
        known = ", ".join(sorted(collection_profiles())) or "none found"
        raise ValueError(
            f"cannot scope a grep to collection {collection!r}: no corpus profile names it "
            f"(profiles found: {known}). Scanning the whole raw corpus instead would read "
            "every collection on this machine, which is what scoping exists to prevent. "
            "Point MAILRAG_PROFILE_DIR at your profiles, or pass an explicit root."
        )
    return files


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
        # Strip the mbox envelope first or every header is lost — see the module
        # docstring. Imported from the loader rather than reimplemented so the
        # two paths cannot drift apart again.
        from src.data.loaders.mail_archive_x import MailArchiveXLoader

        raw = MailArchiveXLoader._strip_mbox_preamble(raw)
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
    max_files: Optional[int] = None,
    max_seconds: Optional[float] = _DEFAULT_MAX_SECONDS,
) -> Dict[str, Any]:
    """Literal / regex search over the raw email corpus (no embeddings).

    Walks the raw ``.eml`` corpus, decodes each body (QP/base64 + HTML->text),
    and returns one row per **matching message** with the matched line snippets
    and message metadata. This is the escape hatch for exact needle hunts where
    dense/hybrid retrieval is blind to numerals and identifiers (issue #82).

    **Cost model.** There is no index: every call decodes raw ``.eml`` files one
    at a time (~2ms per message). The walk stops early as soon as ``max_matches``
    messages have matched, so a *frequent* pattern returns almost immediately
    while a *rare or absent* one scans the whole corpus. ``regex=True`` is not
    inherently slower than a literal -- what costs is how rarely the pattern
    hits. ``max_seconds`` / ``max_files`` therefore bound the work directly, and
    the result says how far the scan actually got.

    Args:
        pattern: The string (or regex, when ``regex=True``) to find. Matching is
            case-insensitive and covers subject + decoded body text.
        collection: Restrict the walk to the files this collection's profile
            selects, so a session scoped to one corpus cannot read another. When
            no profile names it this raises rather than scanning everything. An
            explicit ``root`` overrides scoping entirely. When scoping applies,
            ``$MAILRAG_EML_ROOT`` is not consulted at all.
        max_matches: Maximum matching **messages** to return. Clamped to
            ``[1, 500]`` (hard cap) so a broad pattern can never flood output.
            Set to 1 for an existence check -- it returns on the first hit.
        regex: When true, ``pattern`` is a Python regex; otherwise it is matched
            literally (special characters are escaped).
        root: Explicit corpus root (tests/advanced use); defaults to
            ``$MAILRAG_EML_ROOT`` or ``~/rag_eml``.
        max_files: Stop after scanning this many messages (``None`` = no file
            bound; the deadline still applies).
        max_seconds: Wall-clock budget, clamped to ``(0, 900]``, measured from
            entry -- so it covers the corpus walk as well as the scan, matching
            ``elapsed_s`` and the time a caller actually waits. The walk is
            ~0.25s over 73k files, so this only matters for very short budgets.
            ``None`` disables the deadline -- only safe on a small corpus.

    Returns:
        ``{matches, scanned, corpus_files, complete, stop_reason, elapsed_s, root}``:

        * ``matches`` -- up to ``max_matches`` rows, each
          ``{subject, from, to, date, message_id, attachment_names, matches, path}``
          where the inner ``matches`` is a list of matched-line snippets.
          ``attachment_names`` lists attachment filenames present on the message;
          their *contents* are NOT searched (attachment indexing is issue #80).
        * ``scanned`` / ``corpus_files`` -- messages examined, out of the total
          discovered under ``root``.
        * ``complete`` -- true only when the **entire** corpus was scanned. An
          empty ``matches`` means "not present in this corpus" only when this is
          true; otherwise the needle is merely absent from the first ``scanned``
          messages, which is a different claim.
        * ``stop_reason`` -- ``complete`` | ``max_matches`` | ``max_files`` |
          ``deadline``.

    Raises:
        ValueError: on a blank ``pattern``, an invalid regex, or a missing corpus.
    """
    if not pattern or not pattern.strip():
        raise ValueError("pattern must be a non-empty string")
    limit = max(1, min(int(max_matches), _HARD_MAX_MATCHES))
    if max_files is None:
        file_budget = None
    else:
        file_budget = int(max_files)
        if file_budget <= 0:
            # Rejected rather than clamped to 1, to match max_seconds: a caller
            # passing 0 means "none", and silently scanning one file instead
            # would answer a question they did not ask.
            raise ValueError("max_files must be > 0 (or None for no file bound)")
    if max_seconds is None:
        budget_s = None
    else:
        budget_s = min(float(max_seconds), _HARD_MAX_SECONDS)
        if budget_s <= 0:
            raise ValueError("max_seconds must be > 0 (or None to disable the deadline)")
    rx = _compile(pattern, regex)
    # Scope BEFORE resolving the default root, because scoping decides whether
    # that root is needed at all. A scoped walk takes its files from the
    # profile's own root (see ``scoping.files_for_collection``), so requiring
    # ``$MAILRAG_EML_ROOT`` to exist first failed two ways: a perfectly valid
    # scoped grep died on a default root it was never going to read, and naming
    # an unscopable collection reported "corpus not found" instead of the
    # refusal that actually applied — hiding a scoping error behind a config one.
    scoped_files = _scoped_files(collection, root)
    if scoped_files is None:
        corpus = resolve_eml_root(root)
    else:
        # Report the root the scan really covered, not the default it ignored.
        from src.mcp_server.scoping import root_for_collection

        corpus = root_for_collection(collection) or ""

    # Materialise the file list first: the walk is cheap next to parsing (~0.25s
    # for 73k files) and it buys the caller a denominator, so a partial scan can
    # be reported as "3,000 of 73,251" rather than an unqualified empty result.
    # The clock starts before it, so elapsed_s is the wall time the CALLER waited.
    started = time.monotonic()
    paths = list(_discover_eml(corpus)) if scoped_files is None else scoped_files
    deadline = started + budget_s if budget_s is not None else None

    results: List[Dict[str, Any]] = []
    scanned = 0
    stop_reason = "complete"
    for path in paths:
        if file_budget is not None and scanned >= file_budget:
            stop_reason = "max_files"
            break
        if deadline is not None and time.monotonic() >= deadline:
            stop_reason = "deadline"
            break
        scanned += 1
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
            # Attachment filenames are searchable too. They were, accidentally,
            # while the mbox envelope bug left raw MIME headers in the body text
            # — and the usage log shows callers relying on it to find documents
            # by name. Fixing the parse removed that; matching the decoded
            # filenames gives it back deliberately, without the base64 noise.
            named = [n for n in meta["attachment_names"] if rx.search(n)]
            if named:
                matches = [f"[attachment] {n}" for n in named]
        if not matches:
            continue
        row = dict(meta)
        row["matches"] = matches[:20]  # cap per-message snippets too
        results.append(row)
        if len(results) >= limit:
            stop_reason = "max_matches"
            break
    return {
        "matches": results,
        "scanned": scanned,
        "corpus_files": len(paths),
        "complete": stop_reason == "complete",
        "stop_reason": stop_reason,
        "elapsed_s": round(time.monotonic() - started, 3),
        "root": corpus,
        # Which corpus answered, echoed back so a caller can see the scope it
        # actually got rather than the one it assumed.
        "collection": collection,
        "scoped": scoped_files is not None,
    }
