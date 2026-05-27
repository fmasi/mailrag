#!/usr/bin/env python3
"""Interactive guided selector for a LOCAL .eml export (Mail Archiver X EML).

Walks a local export directory, shows the folder tree, and lets you pick which
folders to index using the same ``prefix`` / level-2 / direct-files semantics as
the Azure blob picker. Writes the chosen selection rules to a JSON file that the
indexer consumes, so selection and indexing are decoupled.

Usage:
    python scripts/select_local_eml.py --root ~/rag_eml --out ~/rag_eml.selection.json

The pure logic lives in ``src/ingest/selection.py`` (unit-tested); this script is
thin interactive glue and requires the ``questionary`` package at runtime.
"""
import argparse
import json
import os
import sys
import time

# Allow running from the repo root without installation.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.ingest import selection


def _print_tree(rels, folder_tree, has_root):
    print(f"\nDiscovered {len(rels)} .eml files. Folder structure:")
    if has_root:
        n = sum(1 for r in rels if "/" not in r)
        print(f"  (container root)         {n:7d}")
    for top in sorted(folder_tree):
        n = sum(1 for r in rels if r.startswith(top))
        print(f"  {top:24} {n:7d}")
        for child in sorted(folder_tree[top]["children"]):
            cn = sum(1 for r in rels if r.startswith(child))
            print(f"      {child:24} {cn:7d}")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Guided selection over a local .eml export."
    )
    parser.add_argument(
        "--root",
        default=os.path.expanduser("~/rag_eml"),
        help="Root directory of the .eml export (default: ~/rag_eml)",
    )
    parser.add_argument(
        "--out",
        default=os.path.expanduser("~/rag_eml.selection.json"),
        help="Where to write the chosen selection (default: ~/rag_eml.selection.json)",
    )
    args = parser.parse_args(argv)

    root = os.path.abspath(os.path.expanduser(args.root))
    if not os.path.isdir(root):
        parser.error(f"root not found: {root}")

    rels = selection.list_eml_relpaths(root)
    folder_tree, has_root = selection.discover_structure(rels)
    _print_tree(rels, folder_tree, has_root)

    try:
        rules = selection.prompt_guided_selection(folder_tree, has_root)
    except KeyboardInterrupt:
        print("\nAborted; nothing written.")
        return 1

    selected = selection.select_eml_paths(root, rules)
    payload = {
        "root": root,
        "selection_rules": rules,
        "n_selected": len(selected),
        "n_total": len(rels),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(args.out, "w") as fh:
        json.dump(payload, fh, indent=2)

    print(f"\nSelected {len(selected)} of {len(rels)} .eml files.")
    print("Rules:")
    for rule in rules:
        print(f"  {rule}")
    print(f"\nSaved selection -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
