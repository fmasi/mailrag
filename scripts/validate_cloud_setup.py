#!/usr/bin/env python3
"""Comprehensive validation suite for Azure Blob + cloud vector store setup.

Run after deploying Azure Blob plus your configured vector store provider
(`qdrant` or `pinecone`) to confirm connectivity, data integrity,
index completeness, and end-to-end retrieval capability.

Usage:
    python scripts/validate_cloud_setup.py           # run all checks
    python scripts/validate_cloud_setup.py --quick   # skip retrieval query test
"""

import argparse
import os
import sys
import time
from typing import Optional

from dotenv import load_dotenv

# Ensure the project root is on ``sys.path`` so ``src`` is importable.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PASS = "✅"
FAIL = "❌"
WARN = "⚠️"
INFO = "ℹ️"


def _header(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def _result(ok: bool, message: str) -> bool:
    print(f"  {PASS if ok else FAIL}  {message}")
    return ok


def _get_vector_provider() -> str:
    provider = os.environ.get("VECTOR_STORE_PROVIDER", "qdrant").strip().lower()
    if provider not in {"qdrant", "pinecone"}:
        raise ValueError("VECTOR_STORE_PROVIDER must be 'qdrant' or 'pinecone' for this validator")
    return provider


def _get_embedding_provider() -> str:
    provider = os.environ.get("RAG_EMBEDDING_PROVIDER", "openai").strip().lower()
    if provider not in {"openai", "lmstudio"}:
        raise ValueError("RAG_EMBEDDING_PROVIDER must be 'openai' or 'lmstudio'")
    return provider


def check_embedding_smoke(embedding_provider: str) -> Optional[bool]:
    """Run a live embedding smoke check when LM Studio is configured."""
    _header("Test 0: Embedding Provider Smoke Test")

    if embedding_provider != "lmstudio":
        print(
            f"  {INFO}  Embedding provider is '{embedding_provider}', skipping LM Studio smoke test"
        )
        return None

    try:
        from scripts.smoke_lmstudio_embedding import run_smoke

        ok, message = run_smoke("validator smoke probe")
        _result(ok, message)
        return ok
    except Exception as exc:
        _result(False, f"Smoke test error: {exc}")
        return False


# ---------------------------------------------------------------------------
# Test 1 — Environment variables
# ---------------------------------------------------------------------------


def check_env_vars(vector_provider: str, embedding_provider: str) -> bool:
    _header("Test 1: Environment Variables")
    required_common = {
        "AZURE_STORAGE_CONNECTION_STRING": "Azure Blob connection",
        "AZURE_BLOB_CONTAINER": "Azure Blob container name",
        "VECTOR_STORE_PROVIDER": "Vector store provider (qdrant or pinecone)",
        "RAG_EMBEDDING_PROVIDER": "Embedding provider (openai or lmstudio)",
    }

    required_provider: dict[str, str] = {}
    if vector_provider == "qdrant":
        required_provider["QDRANT_URL"] = "Qdrant cluster URL"
        required_provider["QDRANT_COLLECTION_NAME"] = "Qdrant collection name"
    else:
        required_provider["PINECONE_API_KEY"] = "Pinecone API key"
        required_provider["PINECONE_INDEX_NAME"] = "Pinecone index name"

    required_embedding: dict[str, str] = {}
    if embedding_provider == "openai":
        required_embedding["OPENAI_API_KEY"] = "OpenAI API key for embeddings"
    else:
        required_embedding["RAG_EMBEDDING_API_BASE"] = "LM Studio/OpenAI-compatible API base"

    all_ok = True

    print(f"  {INFO}  VECTOR_STORE_PROVIDER={vector_provider}")
    print(f"  {INFO}  RAG_EMBEDDING_PROVIDER={embedding_provider}")

    for var, desc in {**required_common, **required_provider, **required_embedding}.items():
        present = bool(os.environ.get(var, "").strip())
        if not _result(present, f"{var} — {desc}"):
            all_ok = False

    # Optional values that are often useful to show for debugging.
    if vector_provider == "qdrant":
        has_qdrant_key = bool(os.environ.get("QDRANT_API_KEY", "").strip())
        print(f"  {INFO}  QDRANT_API_KEY present: {has_qdrant_key}")

    return all_ok


# ---------------------------------------------------------------------------
# Test 2 — Azure Blob connectivity & blob count
# ---------------------------------------------------------------------------


def check_azure_blob() -> tuple[bool, int]:
    """Return (success, blob_count)."""
    _header("Test 2: Azure Blob Storage Connectivity")
    try:
        from azure.storage.blob import BlobServiceClient

        conn = os.environ["AZURE_STORAGE_CONNECTION_STRING"]
        container = os.environ.get("AZURE_BLOB_CONTAINER", "eml-archive")
        prefix = os.environ.get("AZURE_BLOB_PREFIX", "")

        service = BlobServiceClient.from_connection_string(conn)
        container_client = service.get_container_client(container)

        # List blobs
        blobs = [
            b
            for b in container_client.list_blobs(name_starts_with=prefix or None)
            if b.name.endswith(".eml")
        ]
        blob_count = len(blobs)

        _result(True, f"Connected to container '{container}'")
        _result(blob_count > 0, f"Found {blob_count:,} .eml blobs")

        # Show a few sample blob names
        if blobs:
            print(f"  {INFO}  Sample blobs:")
            for b in blobs[:3]:
                print(f"       • {b.name}")
            if blob_count > 3:
                print(f"       … and {blob_count - 3:,} more")

        return True, blob_count

    except Exception as exc:
        _result(False, f"Azure Blob error: {exc}")
        return False, 0


# ---------------------------------------------------------------------------
# Test 3 — Vector store connectivity & vector count
# ---------------------------------------------------------------------------


def check_pinecone() -> tuple[bool, int, Optional[int]]:
    """Return (success, vector_count, dimension)."""
    _header("Test 3: Pinecone Connectivity")
    try:
        from pinecone import Pinecone

        api_key = os.environ["PINECONE_API_KEY"]
        index_name = os.environ["PINECONE_INDEX_NAME"]

        print(f"  {INFO}  Resolved Pinecone index: {index_name}")

        pc = Pinecone(api_key=api_key)
        index = pc.Index(index_name)
        stats = index.describe_index_stats()

        vector_count = stats.total_vector_count
        dimension = stats.dimension

        _result(True, f"Connected to index '{index_name}'")
        _result(vector_count > 0, f"Index contains {vector_count:,} vectors")
        _result(
            dimension == 1536, f"Dimension = {dimension} (expected 1536 for text-embedding-3-small)"
        )

        # Per-namespace breakdown if present
        if hasattr(stats, "namespaces") and stats.namespaces:
            print(f"  {INFO}  Namespaces:")
            for ns, ns_stats in stats.namespaces.items():
                ns_label = ns if ns else "(default)"
                print(f"       • {ns_label}: {ns_stats.vector_count:,} vectors")

        return True, vector_count, dimension

    except Exception as exc:
        _result(False, f"Pinecone error: {exc}")
        return False, 0, None


def check_qdrant() -> tuple[bool, int, Optional[int]]:
    """Return (success, vector_count, dimension)."""
    _header("Test 3: Qdrant Connectivity")
    try:
        from qdrant_client import QdrantClient

        url = os.environ["QDRANT_URL"]
        collection = os.environ.get("QDRANT_COLLECTION_NAME", "email-rag")
        api_key = os.environ.get("QDRANT_API_KEY", "").strip() or None
        prefer_grpc = os.environ.get("QDRANT_PREFER_GRPC", "false").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

        print(f"  {INFO}  Resolved Qdrant collection: {collection}")
        print(f"  {INFO}  QDRANT_PREFER_GRPC={prefer_grpc}")
        print(f"  {INFO}  QDRANT_API_KEY present: {bool(api_key)}")

        client = QdrantClient(url=url, api_key=api_key, prefer_grpc=prefer_grpc)

        collection_exists = client.collection_exists(collection_name=collection)
        _result(collection_exists, f"Collection '{collection}' exists")
        if not collection_exists:
            return False, 0, None

        info = client.get_collection(collection_name=collection)
        points_count = getattr(info, "points_count", None)
        if points_count is None and hasattr(info, "result"):
            points_count = getattr(info.result, "points_count", None)
        points_count = int(points_count or 0)

        # Newer qdrant-client usually exposes this under config.params.vectors.size.
        dimension = None
        config = getattr(info, "config", None)
        params = getattr(config, "params", None)
        vectors = getattr(params, "vectors", None)
        if hasattr(vectors, "size"):
            dimension = vectors.size

        _result(True, f"Connected to Qdrant collection '{collection}'")
        _result(points_count > 0, f"Collection contains {points_count:,} vectors")
        if dimension is not None:
            print(f"  {INFO}  Vector dimension: {dimension}")

        return True, points_count, dimension

    except Exception as exc:
        _result(False, f"Qdrant error: {exc}")
        return False, 0, None


# ---------------------------------------------------------------------------
# Test 4 — Index completeness (blob count vs vector count)
# ---------------------------------------------------------------------------


def check_completeness(blob_count: int, vector_count: int, provider_label: str) -> bool:
    _header("Test 4: Index Completeness")

    if blob_count == 0:
        _result(False, "No blobs found — cannot assess completeness")
        return False

    # Vectors may exceed blob count due to chunking, or be fewer if batch
    # indexing is still running.
    ratio = vector_count / blob_count if blob_count else 0

    print(f"  {INFO}  Azure blobs  : {blob_count:,}")
    print(f"  {INFO}  {provider_label} vecs: {vector_count:,}")
    print(f"  {INFO}  Ratio (vecs/blobs): {ratio:.2f}")

    if ratio >= 1.0:
        _result(
            True,
            "Vector count ≥ blob count — index looks complete (chunking may explain ratio > 1)",
        )
        return True
    elif ratio >= 0.95:
        print(f"  {WARN}  Ratio ≥ 0.95 — nearly complete. Batch job may still be running.")
        return True
    else:
        _result(
            False,
            f"Only {ratio:.0%} of blobs appear indexed. Re-run batch_index_to_vector_store.py to resume.",
        )
        return False


# ---------------------------------------------------------------------------
# Test 5 — Load a small sample from Azure Blob via the loader
# ---------------------------------------------------------------------------


def check_azure_loader() -> bool:
    _header("Test 5: AzureBlobEmailLoader (load 5 emails)")
    try:
        from src.data.loader import load_emails

        start = time.time()
        docs = load_emails(source="azure_blob", num_samples=5)
        elapsed = time.time() - start

        _result(len(docs) > 0, f"Loaded {len(docs)} documents in {elapsed:.1f}s")

        for i, doc in enumerate(docs[:3]):
            sender = doc.metadata.get("sender") or "?"
            subject = (doc.metadata.get("subject") or "?")[:50]
            has_text = bool(doc.text and doc.text.strip())
            _result(has_text, f"Doc {i}: sender={sender}, subject={subject}")

        return len(docs) > 0

    except Exception as exc:
        _result(False, f"Loader error: {exc}")
        return False


# ---------------------------------------------------------------------------
# Test 6 — Query against Pinecone-backed index
# ---------------------------------------------------------------------------


def check_vector_store_query(vector_provider: str) -> bool:
    provider_title = "Qdrant" if vector_provider == "qdrant" else "Pinecone"
    _header(f"Test 6: End-to-End Query via {provider_title}")
    try:
        from src.config.settings import RAGConfig

        # Force the configured provider for retrieval validation.
        RAGConfig.VECTOR_STORE_PROVIDER = vector_provider
        RAGConfig.initialize_settings(include_llm=False)

        from src.storage.persist import StorageManager

        print(f"  Loading index from {provider_title}...")
        index = StorageManager.load_index()

        # Pure retrieval (no LLM call)
        retriever = index.as_retriever(similarity_top_k=3)
        results = retriever.retrieve("meeting schedule")

        _result(
            len(results) > 0, f"Retrieval returned {len(results)} results for 'meeting schedule'"
        )

        for i, r in enumerate(results):
            score = f"{r.score:.3f}" if hasattr(r, "score") and r.score else "n/a"
            snippet = r.text[:80].replace("\n", " ") if r.text else ""
            print(f"       [{i + 1}] score={score}  {snippet}…")

        return len(results) > 0

    except Exception as exc:
        _result(False, f"Query error: {exc}")
        return False


# ---------------------------------------------------------------------------
# Test 7 — Unit tests (existing mocked tests)
# ---------------------------------------------------------------------------


def check_unit_tests() -> bool:
    _header("Test 7: Existing Unit Tests (mocked)")
    import subprocess

    result = subprocess.run(
        ["poetry", "run", "pytest", "tests/", "-v", "--tb=short"],
        capture_output=True,
        text=True,
        cwd=os.path.join(os.path.dirname(__file__), ".."),
    )
    passed = result.returncode == 0

    # Print last 20 lines of output for summary
    lines = result.stdout.strip().split("\n")
    for line in lines[-20:]:
        print(f"  {line}")

    _result(passed, "All unit tests passed" if passed else "Some tests failed — see output above")
    return passed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate cloud storage setup")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Skip the live retrieval query test",
    )
    args = parser.parse_args()

    load_dotenv()

    try:
        vector_provider = _get_vector_provider()
        embedding_provider = _get_embedding_provider()
    except ValueError as exc:
        print(f"\n{FAIL}  Invalid configuration: {exc}\n")
        sys.exit(1)

    print("\n🔍  Cloud Storage Validation Suite")
    print("=" * 60)
    print(f"Using vector provider: {vector_provider}")
    print(f"Using embedding provider: {embedding_provider}")
    if vector_provider == "qdrant":
        print(
            f"Resolved target collection: {os.environ.get('QDRANT_COLLECTION_NAME', 'email-rag')}"
        )
    else:
        print(
            "Resolved target index: "
            f"{os.environ.get('PINECONE_INDEX_NAME', '').strip() or '<missing>'}"
        )

    results: dict[str, Optional[bool]] = {}

    # 0. Embedding smoke test (LM Studio only)
    results["embedding_smoke"] = check_embedding_smoke(embedding_provider)

    # 1. Env vars
    results["env_vars"] = check_env_vars(vector_provider, embedding_provider)

    # 2. Azure Blob
    azure_ok, blob_count = check_azure_blob()
    results["azure_blob"] = azure_ok

    # 3. Vector store
    if vector_provider == "qdrant":
        vector_ok, vector_count, _ = check_qdrant()
        results["qdrant"] = vector_ok
    else:
        vector_ok, vector_count, _ = check_pinecone()
        results["pinecone"] = vector_ok

    # 4. Completeness
    provider_label = "Qdrant" if vector_provider == "qdrant" else "Pinecone"
    if azure_ok and vector_ok:
        results["completeness"] = check_completeness(blob_count, vector_count, provider_label)
    else:
        results["completeness"] = False
        print(f"\n  {WARN}  Skipping completeness check (Azure or {provider_label} unreachable)")

    # 5. Azure loader
    if azure_ok:
        results["azure_loader"] = check_azure_loader()
    else:
        results["azure_loader"] = False

    # 6. End-to-end query
    if not args.quick and vector_ok:
        results[f"{vector_provider}_query"] = check_vector_store_query(vector_provider)
    else:
        reason = "--quick flag" if args.quick else f"{provider_label} unreachable"
        print(f"\n  {WARN}  Skipping query test ({reason})")
        results[f"{vector_provider}_query"] = None  # skipped

    # 7. Unit tests
    results["unit_tests"] = check_unit_tests()

    # Summary
    _header("Summary")
    if vector_provider == "qdrant":
        print("  Target provider: qdrant")
        print(f"  Target collection: {os.environ.get('QDRANT_COLLECTION_NAME', 'email-rag')}")
        print(f"  Target url: {os.environ.get('QDRANT_URL', '').strip() or '<missing>'}")
    else:
        print("  Target provider: pinecone")
        print(f"  Target index: {os.environ.get('PINECONE_INDEX_NAME', '').strip() or '<missing>'}")

    total = 0
    passed = 0
    for name, ok in results.items():
        if ok is None:
            print(f"  ⏭️  {name}: skipped")
            continue
        total += 1
        if ok:
            passed += 1
        print(f"  {PASS if ok else FAIL}  {name}")

    print(f"\n  {passed}/{total} checks passed")

    if passed == total:
        print("\n🎉  All checks passed — your cloud setup is fully operational!\n")
    else:
        print("\n⚠️  Some checks failed — review the output above for details.\n")

    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
