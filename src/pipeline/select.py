"""Interactive folder-selection stage. Pure rule logic is in ingest/selection.py;
this is the thin glue that runs the picker and writes rules onto the profile."""

from __future__ import annotations

from src.ingest.selection import (
    discover_structure,
    list_eml_relpaths,
    prompt_guided_selection,
)


def run(profile, *, questionary=None):
    rels = list_eml_relpaths(profile.resolved_root())
    folder_tree, has_root = discover_structure(rels)
    rules = prompt_guided_selection(folder_tree, has_root, questionary=questionary)
    profile.selection_rules = rules
    return rules
