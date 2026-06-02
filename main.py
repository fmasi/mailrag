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

from src.config.settings import RAGConfig
from src.indexing.contextual_index import build_contextual_index
from src.llm.thread_summaries import generate_thread_summaries
from src.query.hybrid import build_hybrid_searcher


def _init_settings():
    """Initialise LLM + LlamaIndex Settings (requires an API key in the env)."""
    RAGConfig.initialize_settings()


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


def _answer(query, contexts):
    """Generate a grounded answer using the LLM and the retrieved thread contexts."""
    from llama_index.core import Settings
    if not contexts:
        return "No relevant threads retrieved."
    joined = "\n\n---\n\n".join(c.text for c in contexts[:3])
    prompt = (
        f"Answer the question using only these email threads.\n\n"
        f"Threads:\n{joined}\n\nQuestion: {query}\nAnswer:"
    )
    return str(Settings.llm.complete(prompt))


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
        print("  A:", _answer(q, contexts))


def main():
    load_dotenv()
    run_demo()


if __name__ == "__main__":
    main()
