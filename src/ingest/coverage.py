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


def _real(path: str) -> str:
    return os.path.realpath(os.path.expanduser(path))


def _folder_label(dirs: List[str]) -> str:
    """The report row a message belongs to, from its directory components.

    A two-level row counts everything beneath it. A one-level row holds only the
    files sitting directly in that folder, and says so: printed bare, ``5 Sent``
    beside ``4000 Sent/2019`` reads as "Sent holds five messages".
    """
    if not dirs:
        return "(root)"
    if len(dirs) == 1:
        return f"{dirs[0]}/ (direct files)"
    return "/".join(dirs[:2])


def coverage(profiles: Sequence, root: str) -> Dict:
    """Compare what ``profiles`` select against every ``.eml`` under ``root``.

    Returns ``{total, claimed, unclaimed, per_profile, unclaimed_folders}``.
    Profiles are compared as a set because corpora share a root here: a message
    is "unclaimed" only when *no* profile selects it.
    """
    # Both sides are compared as paths under the REAL root. A root taken as typed
    # (relative, ``~/mail``, or through a symlink) never matched the absolute
    # paths a profile resolves to, so every claimed file also counted as
    # unclaimed and the totals stopped adding up.
    root = _real(root)
    all_rel = set(list_eml_relpaths(root))
    all_abs = {os.path.join(root, r) for r in all_rel}

    per_profile: Dict[str, int] = {}
    claimed: set = set()
    for prof in profiles:
        prof_root = prof.resolved_root()
        kept, _ = resolve_index_files(
            prof_root, prof.selection_rules, getattr(prof, "blacklist", None)
        )
        real_prof = _real(prof_root)
        mine = {os.path.join(real_prof, os.path.relpath(k, prof_root)) for k in kept}
        # Counted within the report root, like every other number in the report.
        per_profile[getattr(prof, "collection", "?")] = len(mine & all_abs)
        claimed |= mine

    unclaimed = all_abs - claimed
    folders: Counter = Counter()
    for path in unclaimed:
        rel = os.path.relpath(path, root)
        # Group on the DIRECTORY, at most two levels deep. Taking the first two
        # path components instead named a message sitting straight in a top-level
        # folder after its own file, which is a subject line.
        folders[_folder_label(rel.split(os.sep)[:-1])] += 1

    return {
        "total": len(all_abs),
        # Only what lies under the report root: a profile may select mail outside
        # it, and counting that made "claimed" exceed the total it sits under.
        "claimed": len(claimed & all_abs),
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
