"""
Advanced example demonstrating additional features of the Email RAG system.

This script shows:
1. Custom query types
2. Metadata filtering
3. Batch processing
4. Index statistics
5. Performance monitoring
"""

import time

from dotenv import load_dotenv
from src.indexing.indexer import EmailIndexer
from src.query.engine import EmailQueryEngine

from src.config.settings import RAGConfig


def print_section(title: str):
    """Pretty print a section header."""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}\n")


def example_1_index_statistics():
    """Example 1: Check index statistics and storage info."""
    print_section("Example 1: Index Statistics")

    info = EmailIndexer.get_index_info()

    print("📊 Index Information:")
    print(f"   Storage Location: {info['storage_location']}")
    print(f"   Index Exists: {'✓' if info['index_exists'] else '✗'}")

    if info["index_exists"]:
        print(f"   Storage Size: {info['storage_size_mb']} MB")
        print(f"   Documents: {info.get('document_count', 'unknown')}")

    return info


def example_2_batch_queries(query_engine: EmailQueryEngine):
    """Example 2: Process multiple queries efficiently."""
    print_section("Example 2: Batch Query Processing")

    queries = [
        "What was discussed about scheduling?",
        "Any mentions of deadlines?",
        "Who were the main participants?",
    ]

    print(f"Running {len(queries)} queries...\n")

    start_time = time.time()
    results = []

    for i, query in enumerate(queries, 1):
        print(f"[{i}/{len(queries)}] {query}")
        response = query_engine.query(query)
        results.append(
            {
                "query": query,
                "response": response,
            }
        )
        print("✓ Complete\n")

    elapsed = time.time() - start_time

    print("Summary:")
    print(f"  Total queries: {len(queries)}")
    print(f"  Total time: {elapsed:.2f} seconds")
    print(f"  Average time per query: {elapsed / len(queries):.2f} seconds")


def example_3_retrieval_analysis(query_engine: EmailQueryEngine):
    """Example 3: Analyze what documents are being retrieved."""
    print_section("Example 3: Retrieval Analysis")

    queries = [
        "project management",
        "company policies",
        "financial reports",
    ]

    print("Analyzing retrieval for different query types:\n")

    for query in queries:
        print(f"Query: '{query}'")
        results = query_engine.retrieval_query(query, top_k=3)

        print(f"  📨 Found {len(results)} relevant emails")

        for i, result in enumerate(results, 1):
            sender = result["metadata"].get("sender", "Unknown")
            subject = result["metadata"].get("subject", "No subject")
            score = result.get("score", 0)

            print(f"    {i}. From: {sender[:40]:<40} | Score: {score:.3f}")
            print(f"       Subject: {subject[:60]}")

        print()


def example_4_metadata_exploration(query_engine: EmailQueryEngine):
    """Example 4: Explore metadata without heavy computation."""
    print_section("Example 4: Metadata Exploration")

    print("Senders found in retrieval results:\n")

    results = query_engine.retrieval_query("important", top_k=10)

    senders = {}
    for result in results:
        sender = result["metadata"].get("sender", "Unknown")
        senders[sender] = senders.get(sender, 0) + 1

    print("Sender frequency in results:")
    for sender, count in sorted(senders.items(), key=lambda x: x[1], reverse=True):
        print(f"  • {sender}: {count} emails")


def example_5_custom_filter_query(query_engine: EmailQueryEngine):
    """Example 5: Complex filtering with metadata."""
    print_section("Example 5: Custom Filtering")

    # This example shows how you could implement custom filtering
    # In this case, looking for emails with specific characteristics

    print("Finding varied email types:\n")

    categories = {
        "Meeting related": "meeting schedule agenda",
        "Urgent matters": "urgent important critical",
        "Proposals": "proposal suggest recommend",
    }

    for category, keywords in categories.items():
        results = query_engine.retrieval_query(keywords, top_k=2)
        print(f"{category}:")

        for result in results:
            subject = result["metadata"].get("subject", "No subject")
            print(f"  • {subject[:60]}")
        print()


def main():
    """Run all advanced examples."""
    print("\n" + "=" * 70)
    print("  Email RAG System - Advanced Examples")
    print("=" * 70)

    # Initialize
    print("\n🔧 Initializing system...")
    load_dotenv()

    try:
        RAGConfig.initialize_settings()
        print("✓ Configuration initialized")
    except ValueError as e:
        print(f"✗ Error: {e}")
        return

    # Build/load index
    print("\n📚 Building/loading index...")
    index = EmailIndexer.build_index(
        num_samples=50,  # Smaller dataset for quick examples
        force_rebuild=False,
    )

    # Initialize query engine
    query_engine = EmailQueryEngine(index)
    print("✓ Query engine ready\n")

    # Run examples
    print("\n🚀 Running advanced examples...\n")

    try:
        # Example 1: Index statistics
        example_1_index_statistics()

        # Example 2: Batch queries
        example_2_batch_queries(query_engine)

        # Example 3: Retrieval analysis
        example_3_retrieval_analysis(query_engine)

        # Example 4: Metadata exploration
        example_4_metadata_exploration(query_engine)

        # Example 5: Custom filtering
        example_5_custom_filter_query(query_engine)

        print_section("✅ All examples completed!")

    except Exception as e:
        print(f"\n❌ Error during examples: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
