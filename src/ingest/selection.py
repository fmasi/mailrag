"""Source-agnostic folder selection for choosing which .eml files to index.

Everything here operates on ``/``-separated path *strings*, so the identical
logic serves both Azure blob names and a local directory tree. This is the
canonical implementation; the Azure batch script (``scripts/
batch_index_to_vector_store.py``) currently has its own copies of the
``prefix`` / ``direct-root-files`` / ``container-root`` rule semantics and will
migrate to import from here (see follow-up).

A *selection rule* is a dict of one of three shapes:

    {"type": "prefix", "value": "Inbox/Acme Corp/"}   # folder + all subfolders
    {"type": "direct-root-files", "root": "Inbox/"}     # files directly in Inbox/, not subfolders
    {"type": "container-root"}                          # files at the very top level
"""

import os
from typing import Dict, List, Tuple


def normalize_prefix(prefix: str) -> str:
    """Return *prefix* stripped, with a single trailing slash; empty stays empty."""
    prefix = prefix.strip()
    if not prefix:
        return ""
    return prefix if prefix.endswith("/") else prefix + "/"


def matches_rule(name: str, rule: dict) -> bool:
    """Return True if a ``/``-separated path *name* matches *rule*."""
    rule_type = rule["type"]
    if rule_type == "prefix":
        return name.startswith(rule["value"])
    if rule_type == "direct-root-files":
        root = rule["root"]
        if not name.startswith(root):
            return False
        remainder = name[len(root) :]
        return bool(remainder) and "/" not in remainder
    if rule_type == "container-root":
        return "/" not in name
    raise ValueError(f"Unknown selection rule type: {rule_type}")


def discover_structure(names) -> Tuple[Dict[str, dict], bool]:
    """Build a top-level + level-2 folder tree from ``/``-separated *names*.

    Returns ``(folder_tree, has_container_root_files)`` where each tree entry is
    ``{"children": set[str], "has_direct_files": bool}``.
    """
    folder_tree: Dict[str, dict] = {}
    has_container_root_files = False
    for name in names:
        parts = [p for p in name.split("/") if p]
        if len(parts) <= 1:
            has_container_root_files = True
            continue
        root = f"{parts[0]}/"
        entry = folder_tree.setdefault(root, {"children": set(), "has_direct_files": False})
        if len(parts) == 2:
            entry["has_direct_files"] = True
            continue
        entry["children"].add(f"{parts[0]}/{parts[1]}/")
    return folder_tree, has_container_root_files


def list_eml_relpaths(root: str) -> List[str]:
    """Sorted, ``/``-separated relative paths of every ``.eml`` under *root*."""
    rels: List[str] = []
    for dirpath, _dirs, files in os.walk(root):
        for fn in files:
            if fn.lower().endswith(".eml"):
                rel = os.path.relpath(os.path.join(dirpath, fn), root)
                rels.append(rel.replace(os.sep, "/"))
    return sorted(rels)


def filter_names_by_selection(names, selection_rules) -> List[str]:
    """Keep names matching any rule (preserves input order)."""
    return [n for n in names if any(matches_rule(n, r) for r in selection_rules)]


def select_eml_paths(root: str, selection_rules) -> List[str]:
    """Absolute paths of local ``.eml`` files whose relpath matches any rule."""
    return [
        os.path.join(root, *rel.split("/"))
        for rel in filter_names_by_selection(list_eml_relpaths(root), selection_rules)
    ]


def _require_questionary():
    """Lazily import ``questionary`` so the pure helpers stay dependency-free."""
    try:
        import questionary
    except ImportError as exc:  # pragma: no cover - exercised via injection in tests
        raise RuntimeError(
            "Guided selection requires the 'questionary' package. "
            "Install project dependencies and re-run."
        ) from exc
    return questionary


def prompt_guided_selection(folder_tree, has_container_root_files, questionary=None):
    """Interactively collect selection rules from a terminal UI.

    *questionary* is injected for testing; in production it defaults to the real
    ``questionary`` package. Returns a list of selection rules (see module docs).
    A ``None`` answer (Ctrl-C / ESC) raises ``KeyboardInterrupt``.
    """
    q = questionary or _require_questionary()
    selection_rules: List[dict] = []

    if has_container_root_files:
        include_root = q.confirm(
            "Include .eml files stored directly at the container root?",
            default=False,
        ).ask()
        if include_root is None:
            raise KeyboardInterrupt
        if include_root:
            selection_rules.append({"type": "container-root"})

    for root in sorted(folder_tree):
        entry = folder_tree[root]
        child_prefixes = sorted(entry["children"])

        if child_prefixes:
            action = q.select(
                f"{root} - choose indexing scope",
                choices=[
                    {"name": "Include this folder and all subfolders", "value": "all"},
                    {"name": "Skip this folder", "value": "skip"},
                    {"name": "Choose specific level-2 folders", "value": "children"},
                ],
            ).ask()
        else:
            action = q.select(
                f"{root} - choose indexing scope",
                choices=[
                    {"name": "Include this folder", "value": "all"},
                    {"name": "Skip this folder", "value": "skip"},
                ],
            ).ask()

        if action is None:
            raise KeyboardInterrupt
        if action == "all":
            selection_rules.append({"type": "prefix", "value": root})
            continue
        if action != "children":
            continue

        if entry["has_direct_files"]:
            include_direct = q.select(
                f"{root} [direct files only] - choose indexing scope",
                choices=[
                    {"name": "Include these direct files", "value": True},
                    {"name": "Skip these direct files", "value": False},
                ],
            ).ask()
            if include_direct is None:
                raise KeyboardInterrupt
            if include_direct:
                selection_rules.append({"type": "direct-root-files", "root": root})

        for child_prefix in child_prefixes:
            include_child = q.select(
                f"{child_prefix} - choose indexing scope",
                choices=[
                    {"name": "Include this folder", "value": True},
                    {"name": "Skip this folder", "value": False},
                ],
            ).ask()
            if include_child is None:
                raise KeyboardInterrupt
            if include_child:
                selection_rules.append({"type": "prefix", "value": child_prefix})

    return selection_rules
