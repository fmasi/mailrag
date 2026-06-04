"""Mail Archive X backup loader implementation."""

from __future__ import annotations

import email
import email.header
import os
import re
from datetime import datetime
from email import policy
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser
from typing import Any, Dict, List, Optional

# Matches lines that definitively start a quoted/forwarded block.
_REPLY_SEPARATOR_RE = re.compile(
    r'^('
    r'>[ >]?'                               # "> " or ">>" quoted lines
    r'|-----\s*original message\s*-----'    # Outlook "-----Original Message-----"
    r'|-----\s*forwarded'                   # "-----Forwarded message-----"
    r'|_{10,}'                              # Outlook "________________________"
    r'|[-*_]{10,}'                          # any long divider line
    r'|-{3,}\s*$'                           # line of only dashes (Globex "---------")
    r'|.*\boriginal\s+message\b'            # "Original Message ---------" on its own line
    r')',
    re.IGNORECASE,
)

# "On <date/name> ... wrote:" reply attribution — may span up to 3 lines.
_ON_WROTE_RE = re.compile(r'^on\s', re.IGNORECASE)
_WROTE_END_RE = re.compile(r'wrote:\s*$', re.IGNORECASE)

# Outlook inline reply header — English and Korean variants.
# English: "From: Name\nDate: ...\nTo: ...\nSubject: ..."
# Korean:  "보낸 사람: Name\n날짜: ...\n받는 사람: ...\n주제: ..."
# Triggered when we see the "From" line mid-body followed by ≥2 continuation
# headers nearby.
# Outlook inline reply header — "From" trigger line, all supported locales.
# Only fires mid-body (result non-empty) so first-line false positives
# are impossible.  Requires ≥2 continuation header matches (see below).
_OUTLOOK_FROM_RE = re.compile(
    r'^('
    r'from:\s+\S'   # English
    r'|보낸 사람:'    # Korean   (보낸 사람 = Sent by)
    r'|差出人:'       # Japanese (差出人 = Sender)
    r'|发件人:'       # Chinese Simplified
    r'|寄件人:'       # Chinese Traditional
    r'|จาก:'         # Thai
    r'|Từ:'          # Vietnamese
    r'|Dari:'        # Indonesian / Malay
    r')',
    re.IGNORECASE,
)

# Continuation headers that follow the "From" line.  Two or more matches
# within the next 5 lines confirm this is a reply header block, not prose.
_OUTLOOK_CONTINUATION_RE = re.compile(
    r'^('
    r'date|to|subject|sent'                      # English
    r'|날짜|받는 사람|참조|주제'                   # Korean
    r'|日付|宛先|件名'                             # Japanese
    r'|日期|收件人|收件者|抄送|副本|主题|主旨'       # Chinese Simplified / Traditional
    r'|วันที่|ถึง|สำเนา|หัวเรื่อง'               # Thai
    r'|Ngày|Tới|Chủ đề|Đã gửi'                  # Vietnamese
    r'|Tanggal|Kepada|Perihal|Subjek'             # Indonesian / Malay
    r'):\s',
    re.IGNORECASE,
)


class _HTMLTextExtractor(HTMLParser):
    """Minimal HTML-to-text converter using only the stdlib."""

    # Tags that should introduce a line break in the output.
    _BLOCK_TAGS = frozenset({
        "p", "div", "br", "tr", "li", "h1", "h2", "h3", "h4", "h5", "h6",
        "blockquote", "pre",
    })
    # Tags whose content we skip entirely (not readable prose).
    _SKIP_TAGS = frozenset({"style", "script", "head"})

    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []
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
        # Handle common named entities not covered by handle_data in compat mode.
        _ENTITIES = {"nbsp": " ", "amp": "&", "lt": "<", "gt": ">", "quot": '"'}
        if not self._skip_depth:
            self._parts.append(_ENTITIES.get(name, ""))

    def get_text(self) -> str:
        raw = "".join(self._parts)
        # Collapse runs of blank lines to at most two.
        return re.sub(r'\n{3,}', '\n\n', raw).strip()

from src.data.loaders.base import EmailLoader
from src.data.models import NormalizedEmail
from src.data import calendar_summary


class MailArchiveXLoader(EmailLoader):
    """Load emails from a Mail Archive X backup directory."""

    def __init__(
        self,
        backup_dir: Optional[str] = None,
        eml_files: Optional[List[str]] = None,
    ):
        """
        Initialize the Mail Archive X loader.

        Provide either:
            backup_dir: path to a backup root directory to walk for .eml files, OR
            eml_files:  an explicit list of .eml paths to parse (used by the local
                        guided-selection flow so only chosen files are loaded).
        """
        if eml_files is None and backup_dir is None:
            raise ValueError("Provide either backup_dir or eml_files")
        self.backup_dir = backup_dir
        self.eml_files = list(eml_files) if eml_files is not None else None
        if self.eml_files is None and not os.path.isdir(backup_dir):
            raise ValueError(f"Backup directory not found: {backup_dir}")

    # Flow overview:
    # load()
    #   -> _discover_eml_files()
    #   -> for each file:
    #        _parse_eml_file()
    #          -> _extract_email_body_from_message()
    #        _normalize_mail_archive_x_file()
    #   -> return List[NormalizedEmail]

    def load(self, num_samples: Optional[int] = None) -> List[NormalizedEmail]:
        """Load Mail Archive X backup and normalize each email."""
        source_desc = (
            self.backup_dir
            if self.eml_files is None
            else f"{len(self.eml_files)} selected file(s)"
        )
        print(f"Loading Mail Archive X from {source_desc}...")

        # Discover all .eml files recursively
        eml_files = self._discover_eml_files()
        print(f"  Found {len(eml_files)} .eml files")

        # Limit samples if specified.
        if num_samples:
            eml_files = eml_files[:num_samples]

        emails: List[NormalizedEmail] = []
        for i, eml_path in enumerate(eml_files):
            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(eml_files)} emails...")

            try:
                parsed = self._parse_eml_file(eml_path)
                if not parsed.get("error"):
                    normalized = self._normalize_mail_archive_x_file(parsed, eml_path)
                    emails.append(normalized)
            except Exception as e:
                print(f"    ⚠ Error processing {eml_path}: {e}")
                continue

        print(f"Loaded {len(emails)} emails from Mail Archive X")
        return emails

    def get_source_info(self) -> Dict[str, Any]:
        """Return Mail Archive X source metadata."""
        eml_count = len(self._discover_eml_files())
        return {
            "source": "mail_archive_x",
            "location": self.backup_dir,
            "eml_file_count": eml_count,
        }

    def _discover_eml_files(self) -> List[str]:
        """Return the .eml files to load.

        Uses the explicit ``eml_files`` list when one was supplied; otherwise
        recursively discovers every .eml under ``backup_dir``.
        """
        if self.eml_files is not None:
            return self.eml_files
        eml_files = []
        for dirpath, dirnames, filenames in os.walk(self.backup_dir):
            for filename in filenames:
                if filename.endswith(".eml"):
                    eml_files.append(os.path.join(dirpath, filename))
        return eml_files

    @staticmethod
    def _decode_header(raw_value: Optional[str]) -> str:
        """Decode an RFC 2047 encoded header value to a plain string.

        With policy.compat32, header values may contain encoded words like
        ``=?UTF-8?B?SGVsbG8=?=``.  This helper decodes them to a regular
        Unicode string so the result can be stored as clean metadata.
        """
        if not raw_value:
            return ""
        try:
            parts = email.header.decode_header(raw_value)
            return str(email.header.make_header(parts))
        except Exception:
            return raw_value

    @staticmethod
    def _strip_mbox_preamble(data: bytes) -> bytes:
        """Strip mbox envelope and any non-header lines before the RFC 2822 headers.

        Some Mail Archive X exports prepend an mbox ``From `` separator line
        *and* an extra numeric field (e.g. ``188035    ``) before the actual
        RFC 2822 headers.  Python's email parser stops reading headers when it
        encounters a line that does not match ``Name: value`` format, so any
        such preamble causes every header (including ``Subject``) to be silently
        lost.

        This method advances past any leading lines that are not valid RFC 2822
        header lines (i.e. lines that neither start with ``HeaderName:`` nor are
        continuation lines beginning with whitespace).
        """
        _HEADER_RE = re.compile(rb'^[A-Za-z][A-Za-z0-9\-]*\s*:')
        lines = data.splitlines(keepends=True)
        for i, line in enumerate(lines):
            if _HEADER_RE.match(line):
                return b"".join(lines[i:])
        return data

    def _parse_eml_file(self, eml_path: str) -> Dict[str, Any]:
        """Parse a single .eml file and extract metadata and content."""
        try:
            with open(eml_path, "rb") as f:
                raw = f.read()
            # Strip any mbox envelope / non-header preamble lines that appear
            # before the RFC 2822 headers.  Some Mail Archive X exports insert
            # an extra numeric field (e.g. "188035    ") after the "From "
            # separator line, which breaks Python's email parser and causes all
            # headers — including Subject — to be silently dropped.
            raw = self._strip_mbox_preamble(raw)
            # Use policy.compat32 (the classic lenient parser) instead of
            # policy.default.  The stricter policy.default can silently
            # drop headers from real-world .eml files that use older
            # RFC 2047 encoding or non-standard formatting, causing every
            # email to fall back to "unknown" / "(no subject)".
            msg = email.message_from_bytes(raw, policy=policy.compat32)
        except Exception as e:
            return {
                "sender": "unknown",
                "subject": "unknown",
                "date": None,
                "body": "",
                "error": str(e),
            }

        # Extract and decode header fields (RFC 2047 encoded words are decoded
        # by _decode_header so the stored values are plain Unicode strings).
        sender = self._decode_header(msg.get("From")) or "unknown"
        subject = self._decode_header(msg.get("Subject")) or "(no subject)"
        date_str = msg.get("Date", None)
        recipients = self._decode_header(msg.get("To", ""))
        cc = self._decode_header(msg.get("Cc", ""))

        # RFC 5322 threading headers — ASCII identifiers; collapse any folding
        # whitespace so References becomes a clean space-separated id list.
        message_id = " ".join(str(msg.get("Message-ID") or "").split())
        in_reply_to = " ".join(str(msg.get("In-Reply-To") or "").split())
        references = " ".join(str(msg.get("References") or "").split())

        # Bulk-mail markers (RFC 2369 List-Unsubscribe, RFC 2076 Precedence):
        # marketing / newsletter mail carries these, human mail essentially
        # never does. Surfaced so the conservative Pass-1 noise filter can act.
        precedence = str(msg.get("Precedence") or "").strip().lower()
        is_bulk = (
            msg.get("List-Unsubscribe") is not None
            or precedence in ("bulk", "list")
        )

        # Parse date to datetime if possible.
        date_obj = None
        if date_str:
            try:
                date_obj = parsedate_to_datetime(date_str)
            except Exception:
                date_obj = None

        # Extract body content and strip quoted reply chains.
        body = self._extract_email_body_from_message(msg)
        body = self._strip_reply_chain(body)

        # Collapse calendar invites/notifications to a one-line summary so the
        # (often huge) ics/HTML body doesn't explode into hundreds of chunks.
        calendar_text = self._extract_calendar_text(msg)
        if calendar_summary.is_calendar(subject, body, calendar_text):
            body = calendar_summary.summarize_calendar(subject, calendar_text or body)

        return {
            "sender": sender,
            "subject": subject,
            "date": date_obj,
            "date_str": date_str,
            "recipients": recipients,
            "cc": cc,
            "message_id": message_id,
            "in_reply_to": in_reply_to,
            "references": references,
            "is_bulk": is_bulk,
            "body": body,
            "error": None,
        }

    @staticmethod
    def _decode_part_payload(part: email.message.Message) -> str:
        """Decode a single message part's payload to a Unicode string.

        Works with policy.compat32 Message objects whose ``get_payload(decode=True)``
        returns raw bytes that must be decoded using the part's charset.
        """
        raw = part.get_payload(decode=True)
        if not raw:
            return ""
        charset = part.get_content_charset() or "utf-8"
        try:
            return raw.decode(charset, errors="replace")
        except (LookupError, UnicodeDecodeError):
            return raw.decode("utf-8", errors="replace")

    def _extract_email_body_from_message(self, msg: email.message.Message) -> str:
        """
        Extract body content from email message, handling multipart.

        Works with policy.compat32 Message objects (not the policy.default
        EmailMessage API).  Priority: plain text > HTML > raw payload.
        """
        plain_parts: List[str] = []
        html_parts: List[str] = []

        if msg.is_multipart():
            for part in msg.walk():
                if part.is_multipart():
                    continue
                content_type = part.get_content_type()
                if content_type == "text/plain":
                    plain_parts.append(self._decode_part_payload(part))
                elif content_type == "text/html":
                    html_parts.append(self._decode_part_payload(part))
        else:
            content_type = msg.get_content_type()
            decoded = self._decode_part_payload(msg)
            if content_type == "text/plain":
                plain_parts.append(decoded)
            else:
                html_parts.append(decoded)

        if plain_parts:
            return "\n".join(plain_parts).strip()

        # No plain-text part — convert HTML to readable text so that reply
        # chain patterns ("> ", "-----Original Message-----", etc.) become
        # visible and can be stripped by _strip_reply_chain.
        raw_html = "\n".join(html_parts)
        extractor = _HTMLTextExtractor()
        extractor.feed(raw_html)
        return extractor.get_text()

    def _extract_calendar_text(self, msg: email.message.Message) -> str:
        """Return the decoded text/calendar (VCALENDAR) part, or '' if none."""
        if msg.is_multipart():
            for part in msg.walk():
                if part.is_multipart():
                    continue
                if part.get_content_type() == "text/calendar":
                    return self._decode_part_payload(part)
        elif msg.get_content_type() == "text/calendar":
            return self._decode_part_payload(msg)
        return ""

    @staticmethod
    def _extract_bottom_reply(lines: list[str]) -> str:
        """Extract the reply content from a bottom-posted email.

        Bottom-posting places the new reply *below* the quoted block:

            > Alice wrote: original text
            > more quoted text

            New reply at the bottom here.

        This method scans backwards from the last line, collecting content
        until the first '>' quoted line is reached.  The quoted prefix is
        discarded; only the trailing reply is returned.

        Falls back to the full body if no separator boundary is found during
        the scan (safety net — callers should only invoke this after confirming
        that a leading '>' line exists).
        """
        result: list[str] = []
        found_separator = False
        for line in reversed(lines):
            if _REPLY_SEPARATOR_RE.match(line.strip()):
                found_separator = True
                break
            result.append(line)

        if not found_separator:
            # Safety fallback — no boundary found, return full body.
            return "\n".join(lines).strip()

        result.reverse()
        # Strip leading blank lines left after removing the quoted prefix.
        while result and not result[0].strip():
            result.pop(0)
        # Strip trailing blank lines.
        while result and not result[-1].strip():
            result.pop()

        # If nothing is left (e.g. blank reply above the quoted block),
        # fall back to the full body.
        return "\n".join(result) if result else "\n".join(lines).strip()

    @staticmethod
    def _strip_reply_chain(body: str) -> str:
        """Truncate the body at the first quoted-reply or forwarded-message marker.

        Handles the most common patterns found in work email clients:
        - Lines starting with ">" (standard quoting)
        - Outlook "-----Original Message-----" and "___..." separators
        - "On <date> ... wrote:" reply attribution (spans up to 3 lines)
        - "-----Forwarded message-----" blocks

        Supports both top-posting (reply above quoted history) and
        bottom-posting (reply below quoted history, common in technical
        mailing lists).  Only the new reply content is kept.

        NOTE — known gap: inline/interleaved posting (reply text interspersed
        between individual quoted paragraphs) is not supported.  See
        docs/EMAIL_PREPROCESSING.md for details.
        """
        lines = body.splitlines()
        result: list[str] = []
        has_real_content = False  # True once we've accumulated a non-blank line

        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()

            # Only strip reply markers once we have real content above them.
            # Emails that ARE entirely a forwarded/quoted message (nothing written
            # above the separator) must be preserved in full.
            if has_real_content:
                # Check hard separator patterns first.
                if _REPLY_SEPARATOR_RE.match(stripped):
                    break

                # "On ... wrote:" attribution line — may wrap across up to 3 lines.
                # Check each line individually: joining them would include quoted
                # text from the line AFTER "wrote:", breaking the end-of-line match.
                if _ON_WROTE_RE.match(stripped):
                    lookahead_lines = [
                        lines[j].strip() for j in range(i, min(i + 3, len(lines)))
                    ]
                    if any(_WROTE_END_RE.search(ln) for ln in lookahead_lines):
                        break

            # Outlook inline reply header: "From: Name\nDate: ...\nTo/Subject: ..."
            # Only trigger mid-body (result non-empty) to avoid false positives.
            if result and _OUTLOOK_FROM_RE.match(stripped):
                lookahead_lines = [
                    lines[j].strip() for j in range(i + 1, min(i + 5, len(lines)))
                ]
                matches = sum(
                    1 for ln in lookahead_lines if _OUTLOOK_CONTINUATION_RE.match(ln)
                )
                if matches >= 2:
                    break

            result.append(line)
            if stripped:
                has_real_content = True
            i += 1

        # Strip trailing blank lines left behind by the removed separator.
        while result and not result[-1].strip():
            result.pop()

        top_result = "\n".join(result)

        # Bottom-posting fallback: if the top-posting path produced nothing but
        # '>' quoted lines (i.e. it stopped at the second quoted line and kept
        # only the first), the real reply is at the bottom.  Re-extract from
        # the bottom and use whichever result contains actual non-quoted content.
        #
        # We do NOT trigger this for emails whose first non-blank line is a hard
        # separator (e.g. "-----Forwarded-----") — those are forwarded-only
        # emails correctly preserved in full by the has_real_content guard above.
        top_nonquoted = [l for l in top_result.splitlines() if l.strip() and not l.strip().startswith(">")]
        first_real = next((l.strip() for l in lines if l.strip()), "")
        if not top_nonquoted and first_real.startswith(">"):
            bottom_result = MailArchiveXLoader._extract_bottom_reply(lines)
            if bottom_result:
                return bottom_result

        return top_result

    def _normalize_mail_archive_x_file(
        self, parsed_data: Dict[str, Any], file_path: str
    ) -> NormalizedEmail:
        """Convert parsed Mail Archive X file data to NormalizedEmail."""
        return NormalizedEmail(
            sender=parsed_data.get("sender", "unknown"),
            subject=parsed_data.get("subject", "(no subject)"),
            date=parsed_data.get("date"),
            body=parsed_data.get("body", ""),
            source="mail_archive_x",
            source_id=file_path,
            recipients=parsed_data.get("recipients"),
            cc=parsed_data.get("cc"),
            message_id=parsed_data.get("message_id"),
            in_reply_to=parsed_data.get("in_reply_to"),
            references=parsed_data.get("references"),
            is_bulk=parsed_data.get("is_bulk", False),
        )
