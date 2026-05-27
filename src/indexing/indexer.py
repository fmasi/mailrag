"""
Indexing module for the Email RAG system.

This module orchestrates the full pipeline of creating an index:
1. Loading raw data
2. Validating it
3. Creating and persisting the index

This separation allows us to keep indexing logic separate from querying logic,
making it easier to update indexing strategies without touching the query engine.
"""

import os
from typing import List, Optional

from llama_index.core import VectorStoreIndex, Document

from src.data.loader import load_enron_dataset, validate_documents
from src.storage.persist import StorageManager
from src.config.settings import RAGConfig


class EmailIndexer:
    """
    Orchestrates the indexing pipeline for email data.
    
    This class manages:
    1. Loading data from the dataset
    2. Creating and managing indexes
    3. Handling persistence
    """
    
    @staticmethod
    def build_index(
        num_samples: Optional[int] = None,
        force_rebuild: bool = False,
        validate: bool = True,
    ) -> VectorStoreIndex:
        """
        Build or load an email index.
        
        This is the main entry point for getting a ready-to-query index.
        It handles both creating a new index and loading an existing one.
        
        Args:
            num_samples: Number of emails to index. If None, uses all available.
                        Useful for testing with smaller datasets.
            force_rebuild: If True, always rebuild the index (ignores existing).
                          Useful for testing index creation or updating embeddings.
            validate: If True, validates documents before indexing.
                     Helps catch data issues early.
        
        Returns:
            VectorStoreIndex ready for querying
            
        Example usage:
            # First time (creates and persists):
            index = EmailIndexer.build_index()
            
            # Second time (loads from disk quickly):
            index = EmailIndexer.build_index()
            
            # Testing with small dataset:
            index = EmailIndexer.build_index(num_samples=100)
            
            # Force rebuild (ignores cache):
            index = EmailIndexer.build_index(force_rebuild=True)
        """
        # Check if we should use existing index
        if not force_rebuild and StorageManager.index_exists():
            print("Found existing index, loading from disk...")
            return StorageManager.load_index()
        
        # Load raw data from dataset
        print(f"Building fresh index (num_samples={num_samples})...")
        documents = load_enron_dataset(num_samples=num_samples)
        
        # Validate data quality
        if validate:
            validate_documents(documents)
        
        # Create and save index
        index = StorageManager.create_and_save_index(documents)
        
        return index
    
    @staticmethod
    def get_index_info() -> dict:
        """
        Get information about the current index.
        
        Returns:
            Dictionary with index statistics (size, storage location, etc.)
            
        Why this method:
            - Useful for debugging and monitoring
            - Helps understand index state without loading it
            - Can be called before/after indexing for comparison
        """
        storage_dir = RAGConfig.get_storage_dir()
        
        info = {
            "storage_location": storage_dir,
            "index_exists": StorageManager.index_exists(),
            "storage_size_mb": 0,
        }
        
        if StorageManager.index_exists():
            # Calculate storage size
                        
            total_size = 0
            for dirpath, dirnames, filenames in os.walk(storage_dir):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    total_size += os.path.getsize(filepath)
            
            info["storage_size_mb"] = round(total_size / (1024 * 1024), 2)
            
            # Try to load index and get document count
            try:
                index = StorageManager.load_index()
                info["document_count"] = len(index.docstore.docs)
            except:
                info["document_count"] = "unknown"
        
        return info
    
    @staticmethod
    def print_index_info() -> None:
        """Pretty print index information."""
        info = EmailIndexer.get_index_info()
        
        print("\n=== Index Information ===")
        print(f"Storage location: {info['storage_location']}")
        print(f"Index exists: {info['index_exists']}")
        
        if info['index_exists']:
            print(f"Storage size: {info['storage_size_mb']} MB")
            print(f"Documents indexed: {info.get('document_count', 'unknown')}")
        else:
            print("No index found. Run build_index() to create one.")
        print("=" * 24)
