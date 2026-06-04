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
    r"appointment|\bbill", re.I)
# Promo-ish vocabulary: marketing, newsletters, sales, digests.
PROMO = re.compile(
    r"market|promot|newsletter|\bsale\b|discount|offer|advertis|\bdeal\b|digest|"
    r"recommend", re.I)


def _blob(rec: Record) -> str:
    return (f"{rec.get('subject') or ''} "
            f"{rec.get('reason') or ''} "
            f"{rec.get('summary') or ''}")


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
    return [r for r in results
            if not r.get("is_noise") and PROMO.search(_blob(r))
            and not REC.search(f"{r.get('subject', '')} {r.get('summary', '')}")]


@dataclass
class CalibrationReport:
    rubric: str
    sample: int
    noise_rate: float
    false_noise: List[Record] = field(default_factory=list)
    false_keep: List[Record] = field(default_factory=list)


def _line(rec: Record, tail_key: str) -> str:
    sender = (rec.get("sender") or "")[:36]
    subject = (rec.get("subject") or "")[:44]
    tail = (rec.get(tail_key) or "")[:58]
    return f"  - {sender!r} | {subject!r} :: {tail}"


def format_report(report: CalibrationReport) -> str:
    """Human-readable calibration summary (rate + the two suspect buckets)."""
    pct = round(100 * report.noise_rate)
    lines = [
        f"rubric '{report.rubric}' — sample n={report.sample}",
        f"noise: {pct}%  keep: {100 - pct}%",
        "",
        f"[FALSE-NOISE suspects: record-ish but flagged noise] {len(report.false_noise)}",
    ]
    lines += [_line(r, "reason") for r in report.false_noise[:16]]
    if len(report.false_noise) > 16:
        lines.append(f"  ... ({len(report.false_noise) - 16} more not shown)")
    lines += ["", f"[FALSE-KEEP suspects: promo-ish but kept] {len(report.false_keep)}"]
    lines += [_line(r, "summary") for r in report.false_keep[:16]]
    if len(report.false_keep) > 16:
        lines.append(f"  ... ({len(report.false_keep) - 16} more not shown)")
    return "\n".join(lines)
