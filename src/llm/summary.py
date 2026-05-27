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
