"""Detect calendar invites/notifications and collapse them to one line.

Calendar mail (meeting invites, accept/decline/cancel notices) often carries a
huge text/calendar (VCALENDAR) part or an HTML rendering of recurring-event
details. Left intact it explodes into hundreds of near-useless chunks, so we
replace the body with a compact one-line summary while keeping the meeting's
essence (title, organizer, time, location, attendee count) searchable.

Pure / stdlib-only so it is host-testable; the loader calls it during parsing.
"""

import re

# Subject prefixes emitted by calendar clients (Outlook/Google/Apple), any locale
# we support here in English. Match at the very start, case-insensitively.
_CAL_SUBJECT = re.compile(
    r"^\s*(Canceled|Cancelled|Accepted|Declined|Tentative|Invitation|Updated invitation):",
    re.IGNORECASE,
)

# An iCalendar property line: NAME(;params)?:value
_PROP = re.compile(r"^(?P<name>[A-Za-z0-9\-]+)(;[^:]*)?:(?P<val>.*)$")

# iCalendar timestamp: YYYYMMDDThhmmss(Z)? — keep date + hh:mm.
_DT = re.compile(r"^(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})")


def is_calendar_subject(subject: str) -> bool:
    """True if the subject starts with a calendar-client prefix."""
    return bool(_CAL_SUBJECT.match(subject or ""))


def is_calendar(subject: str = "", body: str = "", calendar_text: str = "") -> bool:
    """Detect calendar mail by content (a calendar part / VCALENDAR) or subject."""
    if calendar_text:
        return True
    if "BEGIN:VCALENDAR" in (body or ""):
        return True
    return is_calendar_subject(subject)


def _fmt_dt(value: str) -> str:
    m = _DT.match(value.strip())
    if not m:
        return value.strip()
    y, mo, d, h, mi = m.groups()
    return f"{y}-{mo}-{d} {h}:{mi}"


def extract_vcalendar_fields(text: str) -> dict:
    """Pull event fields from a VCALENDAR/VEVENT block (best-effort)."""
    fields: dict = {}
    attendees = 0
    for raw in text.splitlines():
        m = _PROP.match(raw.strip())
        if not m:
            continue
        name = m.group("name").upper()
        val = m.group("val").strip()
        if name == "SUMMARY" and "summary" not in fields:
            fields["summary"] = val
        elif name == "ORGANIZER" and "organizer" not in fields:
            fields["organizer"] = val.replace("mailto:", "").strip()
        elif name == "DTSTART" and "dtstart" not in fields:
            fields["dtstart"] = _fmt_dt(val)
        elif name == "DTEND" and "dtend" not in fields:
            fields["dtend"] = _fmt_dt(val)
        elif name == "LOCATION" and "location" not in fields:
            fields["location"] = val
        elif name == "ATTENDEE":
            attendees += 1
    if attendees:
        fields["attendee_count"] = attendees
    return fields


def summarize_calendar(subject: str, source_text: str = "") -> str:
    """Return a single-line summary of a calendar message.

    When *source_text* contains a VCALENDAR block its fields are used; otherwise
    the summary falls back to the subject.
    """
    fields = (
        extract_vcalendar_fields(source_text)
        if "BEGIN:VCALENDAR" in (source_text or "")
        else {}
    )
    title = fields.get("summary") or subject
    parts = [f"[Calendar] {title}"]
    if fields.get("organizer"):
        parts.append(f"organizer: {fields['organizer']}")
    if fields.get("dtstart"):
        when = fields["dtstart"]
        if fields.get("dtend"):
            when += f"–{fields['dtend']}"
        parts.append(f"when: {when}")
    if fields.get("location"):
        parts.append(f"location: {fields['location']}")
    if fields.get("attendee_count"):
        parts.append(f"{fields['attendee_count']} attendee(s)")
    return " | ".join(parts)
