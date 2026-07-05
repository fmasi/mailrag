"""Pass-1: cheap, zero-loss noise pass. Tags noise_candidate on any rule match;
drops nothing. The single home for noise tagging (was inlined in build_local_eml_rag.py)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Pass1Stats:
    total: int
    kept: int
    dropped: int
    tagged: int


def run(emails, noise_filter):
    """Mutate each email's noise_candidate flag; return (kept_emails, Pass1Stats)."""
    tagged = 0
    for e in emails:
        if noise_filter.matched_category(e) is not None:
            e.noise_candidate = True
            tagged += 1
    return emails, Pass1Stats(total=len(emails), kept=len(emails), dropped=0, tagged=tagged)
