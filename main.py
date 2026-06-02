"""
Main entry point for the Email RAG system.

This script demonstrates how to use all the modules together:
1. Initialize configuration
2. Build/load the index
3. Query the indexed emails

Run this file to test the full pipeline.
"""

import os
from dotenv import load_dotenv

from src.indexing.contextual_index import build_contextual_index
from src.llm.thread_summaries import generate_thread_summaries
from src.query.hybrid import build_hybrid_searcher


def _init_settings():
    """Ensure environment variables are loaded for the demo.

    The demo uses the project LLM client (``src.llm.client``) and bge-m3
    (FlagEmbedding) directly, so no LlamaIndex LLM or embedding model is
    needed.  Calling ``RAGConfig.initialize_settings`` is intentionally
    skipped because it imports ``llama_index.llms.*`` which is not installed
    in the ``rag`` conda environment.
    """
    from dotenv import load_dotenv
    load_dotenv()


def _load_demo_emails(num_samples):
    """Load NormalizedEmail objects from the Enron dataset."""
    from src.data.loaders.enron import EnronDatasetLoader
    return EnronDatasetLoader().load(num_samples=num_samples)


def _make_embedder():
    """Construct the BGE-M3 embedder (lazy import: avoids FlagEmbedding at import time)."""
    from src.ingest.embedder import BgeM3Embedder
    return BgeM3Embedder(use_fp16=True)


def _require_qdrant(url="http://localhost:6333"):
    """Raise SystemExit if Qdrant is not reachable."""
    from src.ingest import hybrid_qdrant as hq
    try:
        hq.get_client(url).get_collections()
    except Exception as e:
        raise SystemExit(
            f"Qdrant not reachable at {url}. Start it with `make demo` or "
            f"`docker compose up -d qdrant`. ({e})"
        )



def run_demo(
    num_samples=100,
    collection="mailrag-demo",
    queries=(
        "What meeting times were proposed?",
        "What did people decide about the schedule?",
    ),
):
    """Build the contextual index and answer queries via thread-aware retrieval."""
    _init_settings()
    _require_qdrant()
    emails = _load_demo_emails(num_samples)
    from src.data.threading import assign_subject_fallback_thread_ids
    assign_subject_fallback_thread_ids(emails)
    print(f"Loaded {len(emails)} Enron emails; generating thread-aware summaries (LLM)...")
    summaries = generate_thread_summaries(emails)
    res = build_contextual_index(
        emails,
        collection=collection,
        embedder=_make_embedder(),
        summaries=summaries,
        embed_summary=True,
        recreate=True,
    )
    print(f"Built '{res.collection}': {res.kept_emails} emails -> {res.chunks} chunks")
    searcher = build_hybrid_searcher(collection, mode="hybrid")
    for q in queries:
        print(f"\nQ: {q}")
        contexts = searcher.search_threads(q)
        print(f"  retrieved {len(contexts)} thread(s); answering...")
        from src.llm.answer import answer_from_threads
        print("  A:", answer_from_threads(q, contexts))


def main():
    load_dotenv()
    run_demo()


if __name__ == "__main__":
    main()
