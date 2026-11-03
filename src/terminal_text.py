"""Make email-derived text safe to print to a terminal.

Senders, subjects and anything an LLM wrote about a message are attacker-chosen.
An RFC 2047 encoded-word decodes to any character, ESC included, so printing
one raw lets an email clear the screen, retitle the window or worse. Rich and
Textual markup escaping does not help: it handles ``[`` and ``]``, not ESC.
"""

from __future__ import annotations


def printable(text: str) -> str:
    """``text`` with every non-printable character (control codes included)
    replaced by ``?``. Printable Unicode is kept as it is."""
    return "".join(ch if ch.isprintable() else "?" for ch in text)
