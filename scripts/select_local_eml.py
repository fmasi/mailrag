#!/usr/bin/env python3
"""Thin shim — orchestration now lives in src/pipeline. Old flags preserved.

Interactive guided selector for a LOCAL .eml export (Mail Archiver X EML).

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
import os
import sys

# Allow running from the repo root without installation.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.pipeline import select as select_stage
from src.profile import CorpusProfile


def main(argv=None):
    parser = argparse.ArgumentParser(description="Guided selection over a local .eml export.")
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

    try:
        prof = CorpusProfile(root=root)
        select_stage.run(prof)
    except KeyboardInterrupt:
        print("\nAborted; nothing written.")
        return 1

    prof.save(args.out)
    print(f"\nSaved selection -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
