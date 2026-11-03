"""Pure error-bucket heuristics for the ``calibrate`` gate.

Given the per-email judgments a rubric produced on a sample, surface the two
buckets a human should eyeball before committing to a full Pass-2:

- ``false_noise``: record-ish mail the rubric flagged as noise (over-aggressive).
- ``false_keep``:  promo-ish mail the rubric kept (under-aggressive).

These are regex heuristics, not ground truth — they focus the human's attention,
they do not auto-decide. Records are flat dicts:
``{"sender", "subject", "is_noise", "confidence", "summary", "reason"}``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List

Record = Dict[str, Any]

# Record-ish vocabulary: receipts, statements, bookings, account/tax events, bills.
REC = re.compile(
    r"receipt|invoice|order|booking|reservation|statement|payment|delivery|shippe|"
    r"itinerary|ticket|boarding|refund|transaction|account|policy|\btax\b|enrol|"
    r"appointment|\bbill",
    re.I,
)
# Promo-ish vocabulary: marketing, newsletters, sales, digests.
PROMO = re.compile(
    r"market|promot|newsletter|\bsale\b|discount|offer|advertis|\bdeal\b|digest|"
    r"recommend",
    re.I,
)


def _blob(rec: Record) -> str:
    return f"{rec.get('subject') or ''} {rec.get('reason') or ''} {rec.get('summary') or ''}"


def noise_rate(results: List[Record]) -> float:
    """Fraction of *results* judged noise (0.0 for an empty list)."""
    if not results:
        return 0.0
    return sum(1 for r in results if r.get("is_noise")) / len(results)


def false_noise(results: List[Record]) -> List[Record]:
    """Record-ish emails the rubric flagged as noise (suspected over-aggressive)."""
    return [r for r in results if r.get("is_noise") and REC.search(_blob(r))]


def false_keep(results: List[Record]) -> List[Record]:
    """Promo-ish emails the rubric kept (suspected under-aggressive).

    Excludes anything whose subject/summary also reads record-ish (a "sale
    receipt" is a record, not a missed promo).
    """
    return [
        r
        for r in results
        if not r.get("is_noise")
        and PROMO.search(_blob(r))
        and not REC.search(f"{r.get('subject', '')} {r.get('summary', '')}")
    ]


@dataclass
class CalibrationReport:
    rubric: str
    sample: int
    noise_rate: float
    false_noise: List[Record] = field(default_factory=list)
    false_keep: List[Record] = field(default_factory=list)


#: Distinct sender+subject rows shown per bucket before "... more".
_SHOWN = 16

#: The two bucket headings. Worded as questions a first-time reader can answer
#: by looking, not in the pipeline's own terms. CLI and wizard share them.
FALSE_NOISE_HEADING = "Marked junk, but might be a real record"
FALSE_KEEP_HEADING = "Kept, but might be junk"


def _line(rec: Record, tail_key: str, count: int) -> str:
    sender = (rec.get("sender") or "")[:36]
    subject = (rec.get("subject") or "")[:44]
    tail = (rec.get(tail_key) or "")[:58]
    times = f"  (x{count})" if count > 1 else ""
    return f"  - {sender} | {subject}{times} :: {tail}"


def _bucket(records: List[Record], tail_key: str) -> List[str]:
    """One line per distinct sender+subject, with a count.

    A sample often holds the same notification many times over (eleven copies
    of one password-manager nag in one run), and listing each copy buried the
    rows worth reading."""
    groups: Dict[tuple, List[Record]] = {}
    for rec in records:
        key = ((rec.get("sender") or "").strip(), (rec.get("subject") or "").strip())
        groups.setdefault(key, []).append(rec)
    lines = [_line(recs[0], tail_key, len(recs)) for recs in list(groups.values())[:_SHOWN]]
    if len(groups) > _SHOWN:
        lines.append(f"  ... ({len(groups) - _SHOWN} more not shown)")
    return lines


def format_report(report: CalibrationReport) -> str:
    """Human-readable calibration summary (rate + the two suspect buckets)."""
    pct = round(100 * report.noise_rate)
    lines = [
        f"junk rules '{report.rubric}', tried on {report.sample} sample emails",
        f"junk: {pct}%  keep: {100 - pct}%",
        "",
        f"[{FALSE_NOISE_HEADING}] {len(report.false_noise)}",
    ]
    lines += _bucket(report.false_noise, "reason")
    lines += ["", f"[{FALSE_KEEP_HEADING}] {len(report.false_keep)}"]
    lines += _bucket(report.false_keep, "summary")
    return "\n".join(lines)
