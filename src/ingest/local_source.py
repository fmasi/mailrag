"""Resolve which local .eml files to index.

Bridges the guided folder selection (src/ingest/selection.py) and the
content-addressed blacklist (src/data/blacklist.py): apply the saved selection
rules to the export tree, then drop any blacklisted files. The resulting list
is handed to MailArchiveXLoader(eml_files=...) so only the chosen, non-skipped
emails are parsed.
"""

from typing import List, Optional, Tuple

from src.data.blacklist import filter_blacklisted
from src.ingest.selection import select_eml_paths


def resolve_index_files(
    root: str,
    selection_rules,
    blacklist_path: Optional[str] = None,
) -> Tuple[List[str], List[str]]:
    """Return (files_to_index, skipped) for *root* given selection + blacklist.

    Files matching the selection rules are kept, minus any whose
    sha256 appears in *blacklist_path* (when provided).
    """
    selected = select_eml_paths(root, selection_rules)
    if blacklist_path:
        return filter_blacklisted(selected, blacklist_path)
    return selected, []
