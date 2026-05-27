#!/usr/bin/env python3
"""Debug the reply-chain stripping on the longest emails in the sample.

Downloads a sample of blobs, runs each body through _strip_reply_chain,
and shows before/after token counts for the worst offenders.

Usage:
    python scripts/debug_strip_reply_chain.py
    python scripts/debug_strip_reply_chain.py --sample 500 --show 10
"""

import argparse
import os
import random
import sys
import tempfile

from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.batch_index_to_vector_store import (
    _filter_blobs_by_selection,
    _read_checkpoint_state,
)
from src.data.loaders.mail_archive_x import MailArchiveXLoader


def _rough_token_count(text: str) -> int:
    return int(len(text.split()) / 0.75)


def _load_raw_bodies(loader: MailArchiveXLoader) -> list[dict]:
    """Load emails but capture body BEFORE and AFTER stripping."""
    import email
    from email import policy

    results = []
    eml_files = loader._discover_eml_files()

    for eml_path in eml_files:
        try:
            with open(eml_path, "rb") as f:
                raw = f.read()
            raw = loader._strip_mbox_preamble(raw)
            msg = email.message_from_bytes(raw, policy=policy.compat32)
            raw_body = loader._extract_email_body_from_message(msg)
            stripped_body = loader._strip_reply_chain(raw_body)
            results.append({
                "path": eml_path,
                "raw_tokens": _rough_token_count(raw_body),
                "stripped_tokens": _rough_token_count(stripped_body),
                "raw_body": raw_body,
                "stripped_body": stripped_body,
            })
        except Exception as e:
            print(f"  Error on {eml_path}: {e}")

    return results


def _first_line_of_quote(raw_body: str) -> str:
    """Return the first line that looks like a reply marker, or 'none found'."""
    from src.data.loaders.mail_archive_x import (
        _ON_WROTE_RE,
        _REPLY_SEPARATOR_RE,
        _WROTE_END_RE,
    )
    lines = raw_body.splitlines()
    for i, line in enumerate(lines):
        s = line.strip()
        if _REPLY_SEPARATOR_RE.match(s):
            return f"line {i+1}: {repr(s[:120])}"
        if _ON_WROTE_RE.match(s):
            lookahead = " ".join(lines[j].strip() for j in range(i, min(i + 3, len(lines))))
            if _WROTE_END_RE.search(lookahead):
                return f"line {i+1}: {repr(s[:120])}"
    return "none found — stripping had no effect"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sample", type=int, default=300,
                        help="Number of blobs to download (default: 300)")
    parser.add_argument("--show", type=int, default=5,
                        help="Number of worst-case emails to inspect (default: 5)")
    parser.add_argument("--preview", type=int, default=800,
                        help="Characters of raw body to show (default: 800)")
    args = parser.parse_args()

    load_dotenv()

    from azure.storage.blob import BlobServiceClient

    connection_string = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
    container_name = os.environ.get("AZURE_BLOB_CONTAINER", "eml-archive")
    service_client = BlobServiceClient.from_connection_string(connection_string)
    container_client = service_client.get_container_client(container_name)

    all_blobs = [b for b in container_client.list_blobs() if b.name.endswith(".eml")]
    checkpoint_state = _read_checkpoint_state()
    if checkpoint_state and checkpoint_state.get("version", 1) >= 2:
        all_blobs = _filter_blobs_by_selection(all_blobs, checkpoint_state["selection_rules"])

    sample_blobs = random.sample(all_blobs, min(args.sample, len(all_blobs)))
    print(f"Downloading {len(sample_blobs)} blobs...")

    from concurrent.futures import ThreadPoolExecutor

    def _download(blob):
        bc = container_client.get_blob_client(blob.name)
        return blob.name, bc.download_blob().readall()

    with tempfile.TemporaryDirectory() as tmp:
        with ThreadPoolExecutor(max_workers=8) as ex:
            for name, data in ex.map(_download, sample_blobs):
                local = os.path.join(tmp, os.path.basename(name))
                with open(local, "wb") as f:
                    f.write(data)

        loader = MailArchiveXLoader(tmp)
        print("Parsing and comparing before/after stripping...\n")
        results = _load_raw_bodies(loader)

    if not results:
        print("No emails parsed.")
        return

    # Sort by raw token count descending — worst offenders first
    results.sort(key=lambda r: r["raw_tokens"], reverse=True)

    # Summary
    total_raw = sum(r["raw_tokens"] for r in results)
    total_stripped = sum(r["stripped_tokens"] for r in results)
    unchanged = sum(1 for r in results if r["raw_tokens"] == r["stripped_tokens"])
    reduced = len(results) - unchanged
    pct_saved = 100.0 * (total_raw - total_stripped) / total_raw if total_raw else 0

    print(f"{'='*60}")
    print(f"  Stripping summary  (n={len(results)})")
    print(f"{'='*60}")
    print(f"  Emails where stripping changed body : {reduced} ({100*reduced//len(results)}%)")
    print(f"  Emails unchanged (no pattern found) : {unchanged} ({100*unchanged//len(results)}%)")
    print(f"  Total tokens before : {total_raw:,}")
    print(f"  Total tokens after  : {total_stripped:,}")
    print(f"  Tokens removed      : {total_raw - total_stripped:,}  ({pct_saved:.1f}% reduction)")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------ #
    # Invariant validation                                                #
    # ------------------------------------------------------------------ #
    def _first_nonblank(text: str) -> str:
        for line in text.splitlines():
            if line.strip():
                return line.strip()
        return ""

    suspicious = []
    for r in results:
        if r["raw_tokens"] == r["stripped_tokens"]:
            continue  # unchanged — nothing to check

        issues = []

        # Invariant 1: first non-blank line must be identical before/after
        raw_first = _first_nonblank(r["raw_body"])
        stripped_first = _first_nonblank(r["stripped_body"])
        if raw_first != stripped_first:
            issues.append(
                f"FIRST LINE CHANGED\n"
                f"        raw:     {raw_first[:120]!r}\n"
                f"        stripped:{stripped_first[:120]!r}"
            )

        # Invariant 2: stripped body must not be longer than raw
        if r["stripped_tokens"] > r["raw_tokens"]:
            issues.append("STRIPPED BODY IS LONGER THAN RAW (bug in logic)")

        # Invariant 3: suspicious if result is nearly empty relative to input size.
        # Short "Thanks" / "OK" replies (< 5 tokens) from long emails are the
        # most likely indicator of a false positive.  We use a ratio threshold so
        # that genuinely terse replies to short emails don't trigger the check.
        if r["stripped_tokens"] > 0:
            retention_pct = 100.0 * r["stripped_tokens"] / r["raw_tokens"]
        else:
            retention_pct = 0.0
        if r["raw_tokens"] > 100 and r["stripped_tokens"] < 5:
            issues.append(
                f"NEAR-EMPTY OUTPUT: {r['stripped_tokens']} tokens left "
                f"from {r['raw_tokens']} ({retention_pct:.1f}% retained) — "
                f"verify this is a genuine short reply"
            )

        if issues:
            suspicious.append({**r, "issues": issues})

    if suspicious:
        print(f"{'!'*60}")
        print(f"  VALIDATION FAILURES — {len(suspicious)} suspicious email(s)")
        print(f"{'!'*60}\n")
        for r in suspicious:
            print(f"  {os.path.basename(r['path'])}")
            for issue in r["issues"]:
                print(f"    [FAIL] {issue}")
            print()
    else:
        print(
            f"  Invariant check PASSED — all {len([r for r in results if r['raw_tokens'] != r['stripped_tokens']])} "
            f"stripped emails preserved their first line and produced non-empty output.\n"
        )

    print(f"Top {args.show} longest emails (by raw body) — before vs after:\n")
    for i, r in enumerate(results[: args.show], 1):
        reduction = r["raw_tokens"] - r["stripped_tokens"]
        pct = 100 * reduction / r["raw_tokens"] if r["raw_tokens"] else 0
        trigger = _first_line_of_quote(r["raw_body"])

        print(f"  [{i}] {os.path.basename(r['path'])}")
        print(f"      raw={r['raw_tokens']:,} tokens  stripped={r['stripped_tokens']:,} tokens  "
              f"removed={reduction:,} ({pct:.0f}%)")
        print(f"      First quote marker → {trigger}")
        print(f"\n      --- raw body (first {args.preview} chars) ---")
        print(r["raw_body"][: args.preview].replace("\n", "\n      "))
        if r["stripped_body"] != r["raw_body"]:
            print(f"\n      --- stripped body (first {args.preview} chars) ---")
            print(r["stripped_body"][: args.preview].replace("\n", "\n      "))
        print()


if __name__ == "__main__":
    main()
