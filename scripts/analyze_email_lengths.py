#!/usr/bin/env python3
"""Analyze email body length statistics from a sample of blobs.

Uses the saved checkpoint selection (if any) to sample the same emails
that batch_index_to_vector_store.py would process.

Usage:
    python scripts/analyze_email_lengths.py
    python scripts/analyze_email_lengths.py --sample 500
    python scripts/analyze_email_lengths.py --chunk-size 512
"""

import argparse
import os
import random
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Reuse helpers from the batch script
from scripts.batch_index_to_vector_store import (  # noqa: E402
    _filter_blobs_by_selection,
    _read_checkpoint_state,
)
from src.data.loaders.mail_archive_x import MailArchiveXLoader  # noqa: E402

_DEFAULT_SAMPLE = 200
_DEFAULT_CHUNK_SIZE = 512


def _rough_token_count(text: str) -> int:
    """Approximate token count: ~0.75 words per token (GPT-style rule of thumb)."""
    words = len(text.split())
    return int(words / 0.75)


def _percentile(sorted_values: list[int], pct: float) -> int:
    if not sorted_values:
        return 0
    idx = int(len(sorted_values) * pct / 100)
    return sorted_values[min(idx, len(sorted_values) - 1)]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--sample",
        metavar="N",
        type=int,
        default=_DEFAULT_SAMPLE,
        help=f"Number of emails to sample (default: {_DEFAULT_SAMPLE})",
    )
    parser.add_argument(
        "--chunk-size",
        metavar="TOKENS",
        type=int,
        default=_DEFAULT_CHUNK_SIZE,
        help=f"Chunk size to simulate (default: {_DEFAULT_CHUNK_SIZE})",
    )
    parser.add_argument(
        "--workers",
        metavar="N",
        type=int,
        default=8,
        help="Parallel download workers (default: 8)",
    )
    args = parser.parse_args()

    load_dotenv()

    from azure.storage.blob import BlobServiceClient  # noqa: E402

    connection_string = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
    if not connection_string:
        raise ValueError("AZURE_STORAGE_CONNECTION_STRING is not set")
    container_name = os.environ.get("AZURE_BLOB_CONTAINER", "eml-archive")

    service_client = BlobServiceClient.from_connection_string(connection_string)
    container_client = service_client.get_container_client(container_name)

    all_blobs = [b for b in container_client.list_blobs() if b.name.endswith(".eml")]
    print(f"Total .eml blobs in container: {len(all_blobs)}")

    # Apply checkpoint selection if one exists
    checkpoint_state = _read_checkpoint_state()
    if checkpoint_state and checkpoint_state.get("version", 1) >= 2:
        selection_rules = checkpoint_state["selection_rules"]
        all_blobs = _filter_blobs_by_selection(all_blobs, selection_rules)
        print(f"After checkpoint selection filter: {len(all_blobs)} blobs")
    else:
        prefix = os.environ.get("AZURE_BLOB_PREFIX", "").strip()
        if prefix:
            all_blobs = [b for b in all_blobs if b.name.startswith(prefix)]
            print(f"After prefix filter '{prefix}': {len(all_blobs)} blobs")

    sample_size = min(args.sample, len(all_blobs))
    sample_blobs = random.sample(all_blobs, sample_size)
    print(f"Sampling {sample_size} emails for analysis...\n")

    def _download(blob):
        bc = container_client.get_blob_client(blob.name)
        return blob.name, bc.download_blob().readall()

    raw_emails: list[tuple[str, bytes]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for result in ex.map(_download, sample_blobs):
            raw_emails.append(result)

    # Parse emails using the same loader as the indexer
    token_counts: list[int] = []
    with tempfile.TemporaryDirectory() as tmp:
        for name, data in raw_emails:
            local_path = os.path.join(tmp, os.path.basename(name))
            with open(local_path, "wb") as f:
                f.write(data)

        loader = MailArchiveXLoader(tmp)
        emails = loader.load()
        for em in emails:
            token_counts.append(_rough_token_count(em.body))

    if not token_counts:
        print("No emails could be parsed.")
        return

    token_counts.sort()
    n = len(token_counts)
    total = sum(token_counts)
    chunk_size = args.chunk_size

    will_split = sum(1 for t in token_counts if t > chunk_size)
    fits_in_one = n - will_split

    print("=" * 55)
    print(f"  Email body length analysis  (n={n})")
    print("=" * 55)
    print(f"  Min          : {token_counts[0]:>6} tokens")
    print(f"  Median (p50) : {_percentile(token_counts, 50):>6} tokens")
    print(f"  Mean         : {total // n:>6} tokens")
    print(f"  p75          : {_percentile(token_counts, 75):>6} tokens")
    print(f"  p90          : {_percentile(token_counts, 90):>6} tokens")
    print(f"  p95          : {_percentile(token_counts, 95):>6} tokens")
    print(f"  p99          : {_percentile(token_counts, 99):>6} tokens")
    print(f"  Max          : {token_counts[-1]:>6} tokens")
    print("-" * 55)
    print(f"  Chunk size simulated : {chunk_size} tokens")
    print(f"  Fit in 1 chunk       : {fits_in_one:>5} ({100 * fits_in_one // n}%)")
    print(f"  Would be split       : {will_split:>5} ({100 * will_split // n}%)")
    print("=" * 55)

    # Suggest a good chunk size
    p90 = _percentile(token_counts, 90)
    suggested = max(256, min(1024, (p90 // 64 + 1) * 64))  # round up to nearest 64
    print(f"\n  Suggested chunk_size to cover 90% of emails whole: {suggested}")
    print(f"  (Set RAG_CHUNK_SIZE={suggested} in .env)")


if __name__ == "__main__":
    main()
