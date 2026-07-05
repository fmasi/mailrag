#!/usr/bin/env python3
"""Reset Qdrant data and batch checkpoint for a fresh re-index run.

This script:
1. Deletes all points while preserving collection schema/indexes (optional),
   or deletes the full collection
2. Removes the batch checkpoint file
3. Uses the same environment precedence as batch_index_to_vector_store.py

Environment variable precedence:
  1. System env / Codespace secrets
  2. .env values
  3. Defaults in RAGConfig

Usage:
    python scripts/reset_qdrant_index.py
    python scripts/reset_qdrant_index.py --yes
    python scripts/reset_qdrant_index.py --drop-schema --yes
"""

import argparse
import ipaddress
import os
import sys
from urllib.parse import urlparse

from dotenv import load_dotenv

# Ensure project root is importable as `src`
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config.settings import RAGConfig  # noqa: E402

CHECKPOINT_FILE = os.path.join(os.path.dirname(__file__), ".vector_batch_checkpoint.txt")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reset Qdrant vectors and clear batch checkpoint.")
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip interactive confirmation prompt.",
    )
    parser.add_argument(
        "--drop-schema",
        action="store_true",
        help=(
            "Delete the entire collection (including schema/payload indexes). "
            "By default, the script only clears points and preserves schema."
        ),
    )
    return parser.parse_args()


def _confirm_destructive_action(collection_name: str, drop_schema: bool) -> None:
    action = "DELETE" if drop_schema else "CLEAR"
    expected = f"{action} {collection_name}"
    print("WARNING: This action is destructive.")
    if drop_schema:
        print(
            f"It will permanently delete collection '{collection_name}' "
            "and all stored vectors/indexes."
        )
    else:
        print(
            f"It will permanently delete all points in '{collection_name}', "
            "but keep collection schema and payload indexes."
        )
    print("To continue, type the exact confirmation string below:")
    print(f"  {expected}")

    response = input("Confirmation: ").strip()
    if response != expected:
        print("Aborted. No changes were made.")
        sys.exit(0)


def _read_qdrant_connection() -> tuple[str, str, str | None, bool]:
    url = (os.environ.get("QDRANT_URL") or RAGConfig.QDRANT_URL).strip()
    collection_name = (
        os.environ.get("QDRANT_COLLECTION_NAME") or RAGConfig.QDRANT_COLLECTION_NAME
    ).strip()
    api_key = (os.environ.get("QDRANT_API_KEY") or RAGConfig.QDRANT_API_KEY).strip() or None

    prefer_grpc_raw = os.environ.get("QDRANT_PREFER_GRPC")
    if prefer_grpc_raw is None:
        prefer_grpc = bool(RAGConfig.QDRANT_PREFER_GRPC)
    else:
        prefer_grpc = prefer_grpc_raw.strip().lower() in {"1", "true", "yes", "on"}

    if not url:
        raise ValueError("QDRANT_URL is not set")
    if not collection_name:
        raise ValueError("QDRANT_COLLECTION_NAME is not set")

    return url, collection_name, api_key, prefer_grpc


def _classify_qdrant_target(url: str, api_key: str | None) -> str:
    """Return a human-readable target classification for the configured Qdrant URL."""
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()

    local_hosts = {
        "localhost",
        "127.0.0.1",
        "::1",
        "host.docker.internal",
        "gateway.docker.internal",
    }
    if host in local_hosts:
        return "local"

    try:
        ip_obj = ipaddress.ip_address(host)
        if ip_obj.is_private or ip_obj.is_loopback:
            return "local/private-network"
    except ValueError:
        pass

    if host.endswith(".qdrant.io"):
        return "qdrant-cloud"

    if parsed.scheme == "https" or api_key:
        return "remote/cloud-like"

    return "remote"


def _delete_all_points_keep_schema(client, collection_name: str, batch_size: int = 1000) -> int:
    """Delete all points while preserving collection config and payload indexes."""
    from qdrant_client import models

    deleted_total = 0
    while True:
        points, _ = client.scroll(
            collection_name=collection_name,
            limit=batch_size,
            with_payload=False,
            with_vectors=False,
        )
        if not points:
            break

        point_ids = [p.id for p in points]
        client.delete(
            collection_name=collection_name,
            points_selector=models.PointIdsList(points=point_ids),
            wait=True,
        )
        deleted_total += len(point_ids)
        print(f"  Deleted {deleted_total} points...")

    return deleted_total


def main() -> None:
    args = _parse_args()

    # Load env and initialize non-LLM settings (avoids requiring provider LLM keys).
    load_dotenv()
    RAGConfig.initialize_settings(include_llm=False)

    url, collection_name, api_key, prefer_grpc = _read_qdrant_connection()
    target_kind = _classify_qdrant_target(url, api_key)

    print("Qdrant target configuration:")
    print(f"  URL         : {url}")
    print(f"  Collection  : {collection_name}")
    print(f"  Target type : {target_kind}")
    print(f"  API key     : {'set' if api_key else 'not set'}")
    print(f"  prefer_grpc : {prefer_grpc}")
    print("=" * 60)

    if not args.yes:
        _confirm_destructive_action(collection_name, args.drop_schema)

    print(f"Resetting Qdrant data: {collection_name}")
    print("=" * 60)

    try:
        from qdrant_client import QdrantClient

        client = QdrantClient(
            url=url,
            api_key=api_key,
            prefer_grpc=prefer_grpc,
            check_compatibility=False,
        )

        exists = client.collection_exists(collection_name=collection_name)
        if not exists:
            if not args.drop_schema:
                print(f"✓ Collection '{collection_name}' does not exist. Nothing to clear.")
            else:
                print(f"✓ Collection '{collection_name}' does not exist (already clean)")
        else:
            info = client.get_collection(collection_name=collection_name)
            points_count = getattr(info, "points_count", None)
            if points_count is None and hasattr(info, "result"):
                points_count = getattr(info.result, "points_count", None)
            points_count = int(points_count or 0)

            if not args.drop_schema:
                print(
                    f"Deleting all points from '{collection_name}' "
                    f"({points_count} points), preserving schema/indexes..."
                )
                deleted_total = _delete_all_points_keep_schema(client, collection_name)
                print(
                    f"✓ Cleared {deleted_total} points from '{collection_name}' "
                    "(schema/indexes preserved)"
                )
            else:
                print(f"Deleting collection '{collection_name}' ({points_count} points)...")
                client.delete_collection(collection_name=collection_name)
                print(f"✓ Collection '{collection_name}' deleted")

    except Exception as exc:
        print(f"✗ Error deleting Qdrant collection: {exc}", file=sys.stderr)
        sys.exit(1)

    if os.path.exists(CHECKPOINT_FILE):
        try:
            os.remove(CHECKPOINT_FILE)
            print(f"✓ Removed checkpoint file: {CHECKPOINT_FILE}")
        except Exception as exc:
            print(f"✗ Error removing checkpoint file: {exc}", file=sys.stderr)
            sys.exit(1)
    else:
        print("✓ No checkpoint file found (already clean)")

    print("=" * 60)
    print("✓ Reset complete. Ready to run:")
    print("  poetry run python scripts/batch_index_to_vector_store.py")


if __name__ == "__main__":
    main()
