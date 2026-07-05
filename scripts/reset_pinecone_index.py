#!/usr/bin/env python3
"""Reset Pinecone index and checkpoint file for a fresh batch re-indexing.

This script:
1. Deletes all vectors from the Pinecone index
2. Removes the batch checkpoint file
3. Follows the same environment variable precedence as batch_index_to_vector_store.py

Environment variable precedence:
  1. Codespace secrets / System environment variables
  2. Variables from .env file
  3. Default values (if defined)

Usage:
    python scripts/reset_pinecone_index.py
    python scripts/reset_pinecone_index.py --yes

This prepares your Pinecone index for a fresh batch index run.
"""

import argparse
import os
import sys

from dotenv import load_dotenv

# Ensure the project root is on sys.path so src is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config.settings import RAGConfig  # noqa: E402

CHECKPOINT_FILE = os.path.join(os.path.dirname(__file__), ".vector_batch_checkpoint.txt")


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Delete all vectors from Pinecone index and clear checkpoint."
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip interactive confirmation prompt.",
    )
    return parser.parse_args()


def _confirm_destructive_action(index_name: str) -> None:
    """Require explicit user confirmation before deleting vectors."""
    expected = f"DELETE {index_name}"
    print("WARNING: This action is destructive.")
    print(f"It will permanently delete ALL vectors from Pinecone index '{index_name}'.")
    print("To continue, type the exact confirmation string below:")
    print(f"  {expected}")
    response = input("Confirmation: ").strip()

    if response != expected:
        print("Aborted. No changes were made.")
        sys.exit(0)


def main() -> None:
    """Reset Pinecone index and checkpoint."""
    args = _parse_args()

    # Load .env variables (codespace secrets have higher precedence)
    load_dotenv()
    RAGConfig.initialize_settings()

    # Get Pinecone credentials with fallbacks
    api_key = os.environ.get("PINECONE_API_KEY")
    index_name = os.environ.get("PINECONE_INDEX_NAME", "email-rag")

    if not api_key:
        raise ValueError(
            "PINECONE_API_KEY is not set. Either set it as a codespace secret or in .env file."
        )

    if not args.yes:
        _confirm_destructive_action(index_name)

    print(f"Resetting Pinecone index: {index_name}")
    print("=" * 60)

    # Delete all vectors from the index
    try:
        from pinecone import Pinecone

        pc = Pinecone(api_key=api_key)
        index = pc.Index(index_name)

        # Get current stats before deletion
        stats_before = index.describe_index_stats()
        vector_count = stats_before.total_vector_count

        if vector_count == 0:
            print(f"✓ Index '{index_name}' is already empty (0 vectors)")
        else:
            print(f"Deleting {vector_count} vectors from index '{index_name}'...")
            index.delete(delete_all=True)
            print(f"✓ All vectors deleted from '{index_name}'")

    except Exception as e:
        print(f"✗ Error deleting vectors from Pinecone: {e}", file=sys.stderr)
        sys.exit(1)

    # Remove checkpoint file
    if os.path.exists(CHECKPOINT_FILE):
        try:
            os.remove(CHECKPOINT_FILE)
            print(f"✓ Removed checkpoint file: {CHECKPOINT_FILE}")
        except Exception as e:
            print(f"✗ Error removing checkpoint file: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        print("✓ No checkpoint file found (already clean)")

    print("=" * 60)
    print("✓ Reset complete. Ready to run:")
    print("  poetry run python scripts/batch_index_to_vector_store.py")


if __name__ == "__main__":
    main()
