"""
Data loading module for the Email RAG system.

This module provides a source-agnostic entry point for loading emails
from multiple sources while keeping backward compatibility.
"""

from typing import List, Optional

from llama_index.core import Document

# AzureBlobEmailLoader is imported lazily in the azure_blob branch below: it
# pulls the optional `azure-storage-blob` dependency, so a top-level import here
# would make `from src.data.loader import load_emails` fail on a minimal install
# (the enron / mail_archive_x paths need no Azure). See #44.
from src.data.loaders import EnronDatasetLoader, MailArchiveXLoader


def load_emails(
    source: str = "enron",
    backup_dir: Optional[str] = None,
    num_samples: Optional[int] = None,
) -> List[Document]:
    """
    Load emails from a specified source and return Document objects.

    Args:
        source: "enron", "mail_archive_x", or "azure_blob".
        backup_dir: Required when source="mail_archive_x".
        num_samples: Maximum number of emails to load. None means all.

    Returns:
        List of LlamaIndex Document objects ready for indexing.
    """
    if source == "enron":
        loader = EnronDatasetLoader()
    elif source == "mail_archive_x":
        if not backup_dir:
            raise ValueError("backup_dir required for mail_archive_x source")
        loader = MailArchiveXLoader(backup_dir)
    elif source == "azure_blob":
        from src.data.loaders import AzureBlobEmailLoader  # noqa: PLC0415 (optional dep)

        loader = AzureBlobEmailLoader()
    else:
        raise ValueError(
            "Unknown source: {source}. Must be 'enron', 'mail_archive_x', or 'azure_blob'".format(
                source=source
            )
        )

    normalized_emails = loader.load(num_samples=num_samples)
    documents = [
        email.to_document(doc_id=f"{email.source}_{i}") for i, email in enumerate(normalized_emails)
    ]
    return documents


def load_enron_dataset(num_samples: Optional[int] = None) -> List[Document]:
    """Backward-compatible wrapper for existing Enron-only calls."""
    return load_emails(source="enron", num_samples=num_samples)


def validate_documents(documents: List[Document]) -> None:
    """
    Validate that documents were loaded correctly.

    Args:
        documents: List of Document objects to validate

    Why this approach:
        - Good practice to validate data before indexing
        - Helps catch issues early (e.g., empty documents)
        - Provides feedback on data quality
    """
    print("\nValidating documents...")

    # Check for empty documents
    empty_docs = [d for d in documents if not d.text or len(d.text.strip()) == 0]
    if empty_docs:
        print(f"  Warning: {len(empty_docs)} empty documents found")

    # Check metadata completeness
    docs_with_metadata = sum(1 for d in documents if d.metadata)
    print(f"  ✓ {docs_with_metadata}/{len(documents)} documents have metadata")

    # Show sample
    if documents:
        print("\n  Sample document:")
        print(f"    From: {documents[0].metadata.get('sender', 'N/A')}")
        print(f"    Subject: {documents[0].metadata.get('subject', 'N/A')}")
        print(f"    Date: {documents[0].metadata.get('date', 'N/A')}")
        print(f"    Content preview: {documents[0].text[:100]}...")
