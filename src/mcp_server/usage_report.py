"""Read the MCP usage log back: what agents actually reach for, and what it cost.

The log exists to be acted on, not merely written. Questions it answers that no
amount of reading the code will: which tools are never called (a dead tool is
usually a badly *described* tool), which arguments callers bother to pass, which
calls are slow enough to be abandoned, and which scans ran out of budget rather
than finding nothing.

Two findings from its first week, as a sense of what to look for. Attachment
tools sat at zero calls until their descriptions changed — the tools worked, the
descriptions never said when to reach for them. And grep, which "felt" heavily
used, was 9% of calls against search's 53%: what made it *memorable* was being
decisive, not frequent. Impressions about tool use are unreliable; this file is
not.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from typing import Any, Dict, List

SLOW_MS = 5_000


def load(path: str) -> List[Dict[str, Any]]:
    """Parse the JSONL log, skipping unreadable lines rather than failing."""
    rows: List[Dict[str, Any]] = []
    if not os.path.exists(path):
        return rows
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def summarise(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate the log into the numbers worth looking at."""
    total = len(rows)
    tools = Counter(r.get("tool", "?") for r in rows)
    failures = Counter(r.get("tool", "?") for r in rows if not r.get("ok", True))
    slow = [r for r in rows if (r.get("duration_ms") or 0) > SLOW_MS]
    truncated = [r for r in rows if r.get("complete") is False]
    empty = [r for r in rows if r.get("result_count") == 0]
    # Which arguments callers actually choose to pass, per tool: an argument
    # nobody ever sets is either undiscoverable or unnecessary.
    args: Dict[str, Counter] = {}
    for r in rows:
        args.setdefault(r.get("tool", "?"), Counter()).update(r.get("args", {}).keys())
    return {
        "total": total,
        "tools": tools,
        "failures": failures,
        "slow": slow,
        "truncated": truncated,
        "empty": empty,
        "args": args,
    }


def render(summary: Dict[str, Any], *, limit: int = 5) -> str:
    """Human-readable report. Silence on an empty log, not a wall of zeroes."""
    total = summary["total"]
    if not total:
        return "no MCP usage logged yet (set MAILRAG_MCP_USAGE_LOG, or make some calls)"

    out = [f"{total} MCP calls logged", "", "by tool:"]
    for tool, n in summary["tools"].most_common():
        fails = summary["failures"].get(tool, 0)
        tail = f"   ({fails} failed)" if fails else ""
        out.append(f"  {tool:20s} {n:5d}  {100 * n / total:4.0f}%{tail}")

    out += ["", "arguments callers actually pass:"]
    for tool, counter in sorted(summary["args"].items()):
        used = ", ".join(f"{k}×{v}" for k, v in counter.most_common(6)) or "(none)"
        out.append(f"  {tool:20s} {used}")

    if summary["slow"]:
        out += ["", f"slow calls (>{SLOW_MS // 1000}s):"]
        for r in sorted(summary["slow"], key=lambda x: -(x.get("duration_ms") or 0))[:limit]:
            out.append(
                f"  {(r.get('duration_ms') or 0) / 1000:7.1f}s  {r.get('tool')}"
                f"  stop={r.get('stop_reason', '-')}  args={_brief(r.get('args'))}"
            )

    if summary["truncated"]:
        out += [
            "",
            f"scans that ran out of budget ({len(summary['truncated'])}) — these returned"
            " partial results, so an empty answer from one is not evidence of absence:",
        ]
        for r in summary["truncated"][:limit]:
            out.append(
                f"  {r.get('tool')}  scanned={r.get('scanned')}  stop={r.get('stop_reason')}"
                f"  args={_brief(r.get('args'))}"
            )

    if summary["empty"]:
        out += ["", f"calls returning nothing: {len(summary['empty'])}"]
    return "\n".join(out)


def _brief(args: Any, width: int = 60) -> str:
    if not isinstance(args, dict):
        return "-"
    text = ", ".join(f"{k}={v!r}" for k, v in args.items())
    return text if len(text) <= width else text[:width] + "…"
