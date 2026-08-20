"""How much of a corpus a profile actually claims — and what it silently leaves out.

Selection rules are a snapshot of an interactive choice: the wizard offers each
folder and the answers are recorded as path prefixes. Nothing afterwards reports
the consequence, so a deliberate skip and an oversight look identical once the
profile is saved.

That is not hypothetical. On a real corpus, 11,832 of 73,336 messages (16%)
belonged to no profile at all — most of an account's sent mail among them — and
it was found by accident while investigating something else. Mail in no profile
is not indexed, has no attachments ingested, and (since collection-scoped grep)
is not searchable either. This module makes that number reportable on demand.
"""

from __future__ import annotations

import os
from collections import Counter
from typing import Dict, List, Sequence

from src.ingest.local_source import resolve_index_files
from src.ingest.selection import list_eml_relpaths


def coverage(profiles: Sequence, root: str) -> Dict:
    """Compare what ``profiles`` select against every ``.eml`` under ``root``.

    Returns ``{total, claimed, unclaimed, per_profile, unclaimed_folders}``.
    Profiles are compared as a set because corpora share a root here: a message
    is "unclaimed" only when *no* profile selects it.
    """
    all_rel = set(list_eml_relpaths(root))
    all_abs = {os.path.join(root, r) for r in all_rel}

    per_profile: Dict[str, int] = {}
    claimed: set = set()
    for prof in profiles:
        kept, _ = resolve_index_files(
            prof.resolved_root(), prof.selection_rules, getattr(prof, "blacklist", None)
        )
        per_profile[getattr(prof, "collection", "?")] = len(kept)
        claimed |= set(kept)

    unclaimed = all_abs - claimed
    folders: Counter = Counter()
    for path in unclaimed:
        rel = os.path.relpath(path, root)
        parts = rel.split(os.sep)
        folders["/".join(parts[:2]) if len(parts) > 1 else "(root)"] += 1

    return {
        "total": len(all_abs),
        "claimed": len(claimed),
        "unclaimed": len(unclaimed),
        "per_profile": per_profile,
        "unclaimed_folders": folders,
    }


def render(result: Dict, *, limit: int = 12) -> str:
    """Report coverage, leading with the number that matters."""
    total, unclaimed = result["total"], result["unclaimed"]
    pct = (100 * unclaimed / total) if total else 0.0
    lines = [
        f"{total} .eml under the corpus root",
        f"  claimed by a profile : {result['claimed']}",
        f"  claimed by NONE      : {unclaimed}  ({pct:.0f}%)",
        "",
        "per profile:",
    ]
    for coll, n in sorted(result["per_profile"].items(), key=lambda kv: -kv[1]):
        lines.append(f"  {coll:34s} {n:7d}")
    if unclaimed:
        lines += [
            "",
            "folders no profile selects (these are indexed nowhere and, since",
            "grep is collection-scoped, searchable nowhere either):",
        ]
        for folder, n in result["unclaimed_folders"].most_common(limit):
            lines.append(f"  {n:7d}  {folder}")
        lines += [
            "",
            "If a folder here is deliberately excluded, that is fine — but it should",
            "be a decision someone made knowing the number, not a leftover from a",
            "prompt that never showed one.",
        ]
    return "\n".join(lines)


def load_profiles(paths: List[str]):
    """Load each profile path, skipping unreadable ones."""
    from src.profile import CorpusProfile

    out = []
    for p in paths:
        try:
            out.append(CorpusProfile.load(p))
        except Exception:
            continue
    return out
