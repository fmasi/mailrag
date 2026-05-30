"""Pure helpers for the LLM Pass-2 summarize+judge prompt and its response.

No network here — :func:`build_prompt` formats the prompt and
:func:`parse_response` turns the model's reply into a normalized record.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional

_BODY_CHARS = 4000  # truncate body before sending; tune via the eval harness

_PROMPT_TEMPLATE = """You are cleaning a corporate email archive for a retrieval system.
Read the email and return STRICT JSON only.

Sender: {sender}
Date: {date}
Subject: {subject}
Body:
{body}

Decide:
- "is_noise": true if this is NOT a real human business conversation
  (newsletter, automated notification, marketing, social-media alert, system
  digest, calendar spam); false if it is genuine human business correspondence.
- "confidence": your confidence in the is_noise judgment, 0.0 to 1.0.
- "summary": one or two sentences capturing who is talking to whom and about
  what, suitable for search and reranking. Empty string if is_noise is true.
- "reason": one short sentence justifying the judgment.

Return ONLY this JSON (no markdown, no extra text):
{{"is_noise": <bool>, "confidence": <float>, "summary": "<text>", "reason": "<text>"}}"""


def build_prompt(email: Dict[str, Any], body_chars: int = _BODY_CHARS) -> str:
    """Format the per-email prompt; body is truncated to *body_chars*."""
    body = (email.get("body") or "").strip()
    if len(body) > body_chars:
        body = body[:body_chars] + " […truncated]"
    return _PROMPT_TEMPLATE.format(
        sender=(email.get("sender") or "")[:200],
        date=email.get("date") or "unknown",
        subject=(email.get("subject") or "")[:300],
        body=body,
    )


_THREAD_PROMPT_TEMPLATE = """You are cleaning a corporate email archive for a retrieval system.
You are given the EARLIER messages in this email thread (context only), then the TARGET
email. Read them and return STRICT JSON only about the TARGET.

{preceding_block}TARGET email:
Sender: {sender}
Date: {date}
Subject: {subject}
Body:
{body}

Decide:
- "is_noise": true if the TARGET is NOT a real human business conversation
  (newsletter, automated notification, marketing, social-media alert, system
  digest, calendar spam); false if it is genuine human business correspondence.
- "confidence": your confidence in the is_noise judgment, 0.0 to 1.0.
- "summary": one or two sentences capturing the key entities, decisions, and dates
  and what the TARGET says or asks, RESOLVING references from the earlier-message
  context (a terse reply should name what it is replying to), suitable for search.
  Keep it tight. Empty string if is_noise is true.
- "reason": one short sentence justifying the judgment.

Return ONLY this JSON (no markdown, no extra text):
{{"is_noise": <bool>, "confidence": <float>, "summary": "<text>", "reason": "<text>"}}"""


def _format_preceding(preceding, per_email_chars: int) -> str:
    """Render the earlier-thread context block (empty string if no preceding)."""
    if not preceding:
        return ""
    lines = ["EARLIER messages in this thread (context only, oldest first):"]
    for e in preceding:
        body = (e.get("body") or "").strip()
        if len(body) > per_email_chars:
            body = body[:per_email_chars] + " […]"
        sender = (e.get("sender") or "")[:120]
        date = e.get("date") or "unknown"
        lines.append(f"- From {sender} ({date}): {body}")
    return "\n".join(lines) + "\n\n"


def build_thread_aware_prompt(email: Dict[str, Any], preceding,
                              body_chars: int = _BODY_CHARS,
                              per_email_chars: int = 800,
                              max_preceding: int = 6) -> str:
    """Per-email summarize+judge prompt that resolves references using the *preceding*
    emails in the same thread (the request a terse reply answers). Append-only by
    design (no future context); emits the same schema as :func:`build_prompt`.
    Only the last ``max_preceding`` messages are shown, to bound densification.
    """
    pre = list(preceding)[-max_preceding:] if preceding else []
    body = (email.get("body") or "").strip()
    if len(body) > body_chars:
        body = body[:body_chars] + " […truncated]"
    return _THREAD_PROMPT_TEMPLATE.format(
        preceding_block=_format_preceding(pre, per_email_chars),
        sender=(email.get("sender") or "")[:200],
        date=email.get("date") or "unknown",
        subject=(email.get("subject") or "")[:300],
        body=body,
    )


def _first_json_object(text: str) -> Optional[str]:
    """Return the first balanced ``{...}`` substring (brace-matched), or None."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def _loads_lenient(text: str) -> Dict[str, Any]:
    """json.loads, falling back to the first balanced {...} object in *text*."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        obj = _first_json_object(text)
        if obj is None:
            raise ValueError("no JSON object found in response")
        return json.loads(obj)


def parse_response(text: str) -> Dict[str, Any]:
    """Parse the model's JSON reply into a normalized record.

    Tolerates fenced and prose-wrapped output by extracting the first balanced
    JSON object. Raises ``ValueError`` on no/invalid JSON or a missing
    ``is_noise`` field.
    """
    cleaned = text.strip()
    cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned)
    cleaned = re.sub(r"\n?```$", "", cleaned).strip()
    data = _loads_lenient(cleaned)
    if "is_noise" not in data:
        raise ValueError("response missing 'is_noise'")
    is_noise = bool(data["is_noise"])
    try:
        confidence = float(data.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))
    summary = "" if is_noise else str(data.get("summary", "")).strip()
    reason = str(data.get("reason", "")).strip()
    return {"is_noise": is_noise, "confidence": confidence,
            "summary": summary, "reason": reason}
