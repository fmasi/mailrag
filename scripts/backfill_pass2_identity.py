#!/usr/bin/env python3
"""Backfill stable identity (message_id, content_sha256) onto existing Pass-2
cache rows.

The cache is primarily keyed by the raw-file sha256, which a mailbox re-export
can change for an unchanged email.  This one-off (idempotent) pass re-parses the
selected ``.eml`` files with the SAME loader the build uses, derives the stable
fallback identifiers, and writes them onto matching rows — no LLM calls, so the
expensive sweep output is never recomputed.

    conda run -n rag python scripts/backfill_pass2_identity.py \
        --cache ~/rag_pass2/pass2.db --selection ~/rag_eml.selection.json

Safe to re-run: rows that already have both identifiers are skipped.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys

from src.data.blacklist import file_sha256
from src.data.identity import email_identity
from src.llm.cache import Pass2Cache


def _load_email_fields(path):
    """Parse one .eml with the production loader; return its NormalizedEmail."""
    from src.data.loaders.mail_archive_x import MailArchiveXLoader
    with contextlib.redirect_stdout(io.StringIO()):
        emails = list(MailArchiveXLoader(eml_files=[path]).load())
    if not emails:
        raise ValueError("no email parsed")
    return emails[0]


def backfill(cache_path: str, selection_path: str) -> dict:
    from src.ingest.local_source import resolve_index_files

    sel = json.load(open(selection_path))
    kept, _ = resolve_index_files(sel["root"], sel["selection_rules"], None)
    cache = Pass2Cache(cache_path)  # opening migrates the schema in place

    counts = {"updated": 0, "skipped_present": 0, "not_in_cache": 0, "error": 0}
    total = len(kept)
    for i, path in enumerate(kept, 1):
        try:
            sha = file_sha256(path)
            row = cache.get(sha)
            if row is None:
                counts["not_in_cache"] += 1
                continue
            if row["message_id"] is not None or row["content_sha256"] is not None:
                counts["skipped_present"] += 1
                continue
            e = _load_email_fields(path)
            mid, chash = email_identity(
                sender=e.sender or "", subject=e.subject or "", date=e.date,
                body=e.body or "", message_id=e.message_id or "",
            )
            cache.set_identity(sha, mid, chash)
            counts["updated"] += 1
        except Exception as exc:
            counts["error"] += 1
            print(f"  error on {path}: {exc}", file=sys.stderr)
        if i % 2000 == 0:
            print(f"  {i}/{total}  {counts}", flush=True)

    cache.close()
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--selection", required=True)
    args = ap.parse_args()
    counts = backfill(args.cache, args.selection)
    print(f"backfill: {counts}")


if __name__ == "__main__":
    main()
