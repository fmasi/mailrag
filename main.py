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
from src.data.loader import load_emails, load_enron_dataset
from src.indexing.indexer import EmailIndexer
from src.query.engine import EmailQueryEngine


def demonstrate_multi_source_loading():
    """
    Demonstrate how to load emails from different sources.
    
    This is a teaching example - uncomment the code you want to try.
    """
    print("\n📧 Multi-Source Email Loading Examples:\n")
    
    # Example 1: Load from Enron (backward compatible)
    print("1. Loading from Enron (backward compatible):")
    print("   docs = load_enron_dataset(num_samples=10)")
    try:
        docs = load_enron_dataset(num_samples=10)
        print(f"   ✓ Loaded {len(docs)} documents from Enron")
        if docs:
            print(f"   - Sample: {docs[0].metadata.get('sender')} - {docs[0].metadata.get('subject')}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # Example 2: Load from Enron using new source-agnostic function
    print("\n2. Loading from Enron (new source-agnostic API):")
    print("   docs = load_emails(source='enron', num_samples=10)")
    try:
        docs = load_emails(source="enron", num_samples=10)
        print(f"   ✓ Loaded {len(docs)} documents from Enron")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # Example 3: Load from Mail Archive X (requires actual backup)
    print("\n3. Loading from Mail Archive X (requires backup directory):")
    print("   docs = load_emails(")
    print("       source='mail_archive_x',")
    print("       backup_dir='/path/to/mail_archive_backup',")
    print("       num_samples=10")
    print("   )")
    print("   → Uncomment this and provide a real backup path to test")
    # Uncomment to test with actual Mail Archive X backup:
    # try:
    #     docs = load_emails(
    #         source="mail_archive_x",
    #         backup_dir="/path/to/mail_archive_backup",
    #         num_samples=10
    #     )
    #     print(f"   ✓ Loaded {len(docs)} documents from Mail Archive X")
    # except Exception as e:
    #     print(f"   ✗ Error: {e}")
    
    # Example 4: Combine both sources
    print("\n4. Combining both sources:")
    print("   enron_docs = load_emails(source='enron', num_samples=5)")
    print("   max_docs = load_emails(source='mail_archive_x', backup_dir='...', num_samples=5)")
    print("   all_docs = enron_docs + max_docs")
    print("   → This gives you a mixed dataset from both sources")
    
    print("\n" + "=" * 60 + "\n")


def main():
    """
    Main function demonstrating the Email RAG system workflow.
    """
    # Load environment variables from .env file
    load_dotenv()
    
    print("\n🚀 Email RAG System - Multi-Source Email Support\n")
    print("=" * 60)
    
    # ===== Data Loading Examples =====
    # The system now supports multiple email sources!
    #
    # Option A: Load from Enron (as before - backward compatible)
    #   docs = load_enron_dataset(num_samples=100)
    #
    # Option B: Load from Mail Archive X backup
    #   docs = load_emails(
    #       source="mail_archive_x",
    #       backup_dir="/path/to/mail_archive_backup",
    #       num_samples=100
    #   )
    #
    # Option C: Combine both sources
    #   enron_docs = load_emails(source="enron", num_samples=50)
    #   max_docs = load_emails(
    #       source="mail_archive_x",
    #       backup_dir="/path/to/backup",
    #       num_samples=50
    #   )
    #   all_docs = enron_docs + max_docs  # Mix both sources
    #
    # For now, we use Enron (default)
    
    # Initialize configuration
    # This sets up the LLM, embeddings, and other global settings
    print("Step 1: Initializing configuration...")
    try:
        RAGConfig.initialize_settings()
        print("✓ Configuration initialized")
    except ValueError as e:
        print(f"✗ Configuration error: {e}")
        print("\nPlease ensure your .env file has:")
        print("  OPENAI_API_KEY=sk-...")
        return
    
    # Demonstrate multi-source loading capabilities
    print("\nStep 1.5: Multi-Source Email Loading")
    demonstrate_multi_source_loading()
    
    # Build or load the index
    print("\nStep 2: Building/loading index...")
    print("(First time: creates and saves index. Subsequent runs: loads from disk)")
    
    # For demonstration, using smaller sample
    # In production, remove num_samples to use full dataset
    index = EmailIndexer.build_index(
        num_samples=100,  # Using 100 samples for quick demo
        force_rebuild=False,  # Set to True to recreate index
    )
    
    # Print index information
    EmailIndexer.print_index_info()
    
    # Initialize query engine
    print("\nStep 3: Initializing query engine...")
    query_engine = EmailQueryEngine(index)
    print("✓ Query engine ready")
    
    # Example queries
    print("\n" + "=" * 60)
    print("Running example queries...")
    print("=" * 60)
    
    # Query 1: Pure retrieval (similarity search)
    print("\n1️⃣  Pure Retrieval Query:")
    print("   Finding emails related to: 'meeting schedule'")
    retrieval_results = query_engine.retrieval_query("meeting schedule", top_k=3)
    query_engine.print_query_results(retrieval_results, title="Retrieval Results: 'meeting schedule'")
    
    # Query 2: RAG query (with LLM)
    print("\n2️⃣  RAG Query (with LLM):")
    print("   Question: 'What are the main topics discussed in the emails?'")
    rag_response = query_engine.query("What are the main topics discussed in the emails?")
    print(f"\n{rag_response}\n")
    
    # Query 3: Metadata-filtered query
    print("\n3️⃣  Metadata-Filtered Query:")
    print("   Searching for emails about 'meetings' from specific people")
    filtered_results = query_engine.query_with_metadata_filter(
        "meetings", top_k=3
    )
    query_engine.print_query_results(filtered_results, title="Filtered Results: 'meetings'")
    
    print("\n" + "=" * 60)
    print("✅ Demo complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("  - Modify queries in main() to test different questions")
    print("  - Change num_samples to process full dataset")
    print("  - Update RAGConfig to use different LLM or embeddings")
    print("  - Add your own custom query methods to EmailQueryEngine")


if __name__ == "__main__":
    main()
