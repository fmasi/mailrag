"""
Storage and persistence module for the Email RAG system.

This module handles the core requirement of avoiding re-downloading and re-indexing
the dataset every time the script runs. It manages:

1. Checking if a valid index already exists on disk
2. Loading the index from disk if it exists
3. Creating a new index and saving it if it doesn't exist

This approach significantly improves efficiency for development and deployment.
"""

import os
import shutil
import time
from typing import Optional

from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    Settings,
    load_index_from_storage,
)
from llama_index.core.vector_stores import SimpleVectorStore

from src.config.settings import RAGConfig


def _get_pinecone_vector_store():
    """Lazy import to avoid requiring pinecone when using SimpleVectorStore."""
    from pinecone import Pinecone
    from llama_index.vector_stores.pinecone import PineconeVectorStore

    api_key = os.environ.get("PINECONE_API_KEY")
    index_name = os.environ.get("PINECONE_INDEX_NAME")
    if not api_key:
        raise ValueError("PINECONE_API_KEY environment variable is not set")
    if not index_name:
        raise ValueError("PINECONE_INDEX_NAME environment variable is not set")

    pc = Pinecone(api_key=api_key)
    index = pc.Index(index_name)
    return PineconeVectorStore(pinecone_index=index)


def _get_qdrant_client_and_collection():
    """Build a Qdrant client and return it with the configured collection name."""
    from qdrant_client import QdrantClient

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
        raise ValueError("QDRANT_URL environment variable is not set")
    if not collection_name:
        raise ValueError("QDRANT_COLLECTION_NAME environment variable is not set")

    client = QdrantClient(url=url, api_key=api_key, prefer_grpc=prefer_grpc)
    return client, collection_name


def _get_qdrant_vector_store():
    """Lazy import to avoid requiring qdrant packages for other providers."""
    from llama_index.vector_stores.qdrant import QdrantVectorStore

    client, collection_name = _get_qdrant_client_and_collection()
    return QdrantVectorStore(client=client, collection_name=collection_name)


def _ingest_to_qdrant(documents, vector_store, verbose: bool) -> dict[str, float]:
    """Chunk, embed, and upload documents to Qdrant in explicit sequential batches.

    Splits the work into two visible stages:
    1. Chunking  — SentenceSplitter with its own progress bar; prints chunk count.
    2. Embedding + upload — one HTTP request to LM Studio and one vector-store
       add call per batch, shown as a single updating Rich progress bar.

    Using a manual loop (rather than IngestionPipeline) means exactly one embedding
    request is in-flight at any moment, preventing LM Studio's queue from flooding.
    """
    import contextlib
    from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.core.schema import MetadataMode

    # Stage 1: chunk — iterate per-document so we can drive our own Rich bar.
    splitter = SentenceSplitter(
        chunk_size=RAGConfig.CHUNK_SIZE,
        chunk_overlap=RAGConfig.CHUNK_OVERLAP,
    )
    if verbose:
        parse_progress = Progress(
            TextColumn("  [cyan]Parsing[/cyan]"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
        )
        nodes = []
        with parse_progress:
            task_id = parse_progress.add_task("", total=len(documents))
            for doc in documents:
                nodes.extend(splitter.get_nodes_from_documents([doc], show_progress=False))
                parse_progress.update(task_id, advance=1)
        print(f"  → {len(nodes)} chunks from {len(documents)} documents")
    else:
        nodes = splitter.get_nodes_from_documents(documents, show_progress=False)

    # Drop exact-duplicate chunks (boilerplate signatures/disclaimers/footers)
    # before embedding so identical text isn't embedded repeatedly. Match on the
    # chunk TEXT only (MetadataMode.NONE) so the same disclaimer sent by
    # different people collapses to a single vector.
    from src.data.dedup import dedup_by_content

    _before = len(nodes)
    nodes = dedup_by_content(
        nodes, key=lambda n: n.get_content(metadata_mode=MetadataMode.NONE)
    )
    if verbose and len(nodes) != _before:
        print(f"  → deduped {_before - len(nodes)} duplicate chunk(s); {len(nodes)} remain")

    if not nodes:
        return {
            "embed_secs": 0.0,
            "upload_secs": 0.0,
            "combined_secs": 0.0,
        }

    # Stage 2: embed + upsert one sub-batch at a time, tracked by a single bar.
    batch_size = RAGConfig.EMBEDDING_BATCH_SIZE
    batches = [nodes[i : i + batch_size] for i in range(0, len(nodes), batch_size)]
    total_embed_secs = 0.0
    total_upload_secs = 0.0

    progress: Progress | None = (
        Progress(
            TextColumn("  [cyan]Processing[/cyan]"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("·"),
            TextColumn("[dim]{task.fields[timing]}[/dim]"),
            TimeElapsedColumn(),
        )
        if verbose
        else None
    )

    with progress or contextlib.nullcontext():
        task_id = progress.add_task("", total=len(batches), timing="") if progress else None

        for i, batch in enumerate(batches):
            texts = [n.get_content(metadata_mode=MetadataMode.EMBED) for n in batch]

            t_embed = time.monotonic()
            embeddings = Settings.embed_model.get_text_embedding_batch(
                texts, show_progress=False
            )
            embed_secs = time.monotonic() - t_embed
            total_embed_secs += embed_secs

            for node, emb in zip(batch, embeddings):
                node.embedding = emb

            t_upload = time.monotonic()
            vector_store.add(batch)
            upload_secs = time.monotonic() - t_upload
            total_upload_secs += upload_secs

            if progress is not None:
                avg_embed = total_embed_secs / (i + 1)
                progress.update(
                    task_id,
                    advance=1,
                    timing=f"avg embed {avg_embed:.1f}s/batch",
                )

    if verbose:
        total_stage_secs = total_embed_secs + total_upload_secs
        embed_pct = 100.0 * total_embed_secs / total_stage_secs if total_stage_secs > 0 else 0.0
        upload_pct = 100.0 * total_upload_secs / total_stage_secs if total_stage_secs > 0 else 0.0
        print(
            "  Stage totals: "
            f"embed {total_embed_secs:.1f}s ({embed_pct:.1f}%), "
            f"upload {total_upload_secs:.1f}s ({upload_pct:.1f}%), "
            f"combined {total_stage_secs:.1f}s"
        )

    return {
        "embed_secs": total_embed_secs,
        "upload_secs": total_upload_secs,
        "combined_secs": total_embed_secs + total_upload_secs,
    }


class StorageManager:
    """
    Manages persistence of the LlamaIndex VectorStoreIndex.
    
    This class encapsulates all storage/loading logic, making it easy to:
    - Swap storage backends (e.g., from SimpleVectorStore to Pinecone)
    - Manage storage directories
    - Handle index versioning if needed in the future
    """
    
    @staticmethod
    def index_exists() -> bool:
        """
        Check if a valid index already exists in storage.
        
        Returns:
            True if index exists and is valid, False otherwise.
            
        Why this check:
            - We need to know if we should load an existing index or create a new one
            - Checking for the storage directory isn't enough; we verify structure
            - This prevents confusion if the directory exists but is empty
        """
        if RAGConfig.VECTOR_STORE_PROVIDER == "pinecone":
            from pinecone import Pinecone

            api_key = os.environ.get("PINECONE_API_KEY")
            index_name = os.environ.get("PINECONE_INDEX_NAME")
            if not api_key:
                raise ValueError("PINECONE_API_KEY environment variable is not set")
            if not index_name:
                raise ValueError("PINECONE_INDEX_NAME environment variable is not set")

            pc = Pinecone(api_key=api_key)
            index = pc.Index(index_name)
            stats = index.describe_index_stats()
            return stats.total_vector_count > 0

        if RAGConfig.VECTOR_STORE_PROVIDER == "qdrant":
            client, collection_name = _get_qdrant_client_and_collection()
            if not client.collection_exists(collection_name=collection_name):
                return False

            collection_info = client.get_collection(collection_name=collection_name)
            points_count = getattr(collection_info, "points_count", None)
            if points_count is None and hasattr(collection_info, "result"):
                points_count = getattr(collection_info.result, "points_count", None)
            return (points_count or 0) > 0

        storage_dir = RAGConfig.get_storage_dir()
        
        # Check if storage directory exists
        if not os.path.exists(storage_dir):
            return False
        
        # Check for required index files
        # SimpleVectorStore saves to 'default__vector_store.json'
        index_file = os.path.join(storage_dir, "default__vector_store.json")
        
        # Also check for the metadata file
        has_vector_store = os.path.exists(index_file)
        
        return has_vector_store
    
    @staticmethod
    def load_index() -> VectorStoreIndex:
        """
        Load an existing index from disk (or Pinecone when configured).
        
        Returns:
            VectorStoreIndex loaded from storage.
            
        Raises:
            ValueError: If no index exists on disk.
            
        Why this approach:
            - We use StorageContext to manage all stored data
            - SimpleVectorStore loads vector embeddings from disk
            - This is much faster than re-embedding all documents
        """
        if RAGConfig.VECTOR_STORE_PROVIDER == "pinecone":
            vector_store = _get_pinecone_vector_store()
            index = VectorStoreIndex.from_vector_store(vector_store)
            print("✓ Index loaded from Pinecone")
            return index

        if RAGConfig.VECTOR_STORE_PROVIDER == "qdrant":
            vector_store = _get_qdrant_vector_store()
            index = VectorStoreIndex.from_vector_store(vector_store)
            print("✓ Index loaded from Qdrant")
            return index

        storage_dir = RAGConfig.get_storage_dir()
        
        if not StorageManager.index_exists():
            raise ValueError(
                f"No index found in {storage_dir}. "
                "Please create an index first using StorageManager.create_and_save_index()"
            )
        
        print(f"Loading existing index from {storage_dir}...")
        
        # Create a StorageContext that loads from disk
        # SimpleVectorStore will load all embeddings from the saved JSON file
        storage_context = StorageContext.from_defaults(
            persist_dir=storage_dir,
            vector_store=SimpleVectorStore(),
        )
        
        # Load the index using the storage context
        index = load_index_from_storage(storage_context)
        
        print(f"✓ Index loaded successfully. Contains {len(index.docstore.docs)} documents")
        
        return index
    
    @staticmethod
    def create_and_save_index(
        documents,
        verbose: bool = True,
        return_stats: bool = False,
    ) -> VectorStoreIndex | tuple[VectorStoreIndex, dict[str, float] | None]:
        """
        Create a new index from documents and save it to disk (or Pinecone).
        
        Args:
            documents: List of LlamaIndex Document objects
            verbose: Whether to print progress information
        
        Returns:
            The created VectorStoreIndex
            
        Why this approach:
            - VectorStoreIndex automatically embeds all documents
            - We use SimpleVectorStore (local, no dependencies)
            - persist_dir automatically saves to disk
            - This is one-time cost; subsequent runs load from disk
        """
        if RAGConfig.VECTOR_STORE_PROVIDER == "pinecone":
            if verbose:
                print(f"\nCreating Pinecone index from {len(documents)} documents...")
            vector_store = _get_pinecone_vector_store()
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            index = VectorStoreIndex.from_documents(
                documents,
                storage_context=storage_context,
                show_progress=verbose,
            )
            if verbose:
                print("✓ Index created and stored in Pinecone")
            if return_stats:
                return index, None
            return index

        if RAGConfig.VECTOR_STORE_PROVIDER == "qdrant":
            if verbose:
                print(f"\nIndexing {len(documents)} documents to Qdrant...")
            vector_store = _get_qdrant_vector_store()
            ingest_stats = _ingest_to_qdrant(documents, vector_store, verbose)
            # Return a lightweight index handle pointing at the populated collection
            index = VectorStoreIndex.from_vector_store(vector_store)
            if verbose:
                print("✓ Index created and stored in Qdrant")
            if return_stats:
                return index, ingest_stats
            return index

        storage_dir = RAGConfig.get_storage_dir()
        
        if verbose:
            print(f"\nCreating index from {len(documents)} documents...")
            print(f"This may take a few minutes as we embed all documents...")
        
        # Initialize storage context with SimpleVectorStore
        # This will try to load existing if available, otherwise creates new
        storage_context = StorageContext.from_defaults(
            vector_store=SimpleVectorStore(),
        )
        
        # Create the index
        # This step:
        # 1. Uses the global Settings.embed_model to embed all documents
        # 2. Stores embeddings and metadata in the vector store
        # 3. Prepares the index for querying
        print("Embedding documents (this uses the configured embedding model)...")
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            show_progress=verbose,
        )
        
        if verbose:
            print(f"✓ Index created with {len(index.docstore.docs)} documents")
            print(f"Saving index to {storage_dir}...")
        
        # Save the index to disk
        # This persists:
        # - Vector embeddings (in default__vector_store.json)
        # - Document metadata
        # - Index structure
        storage_context.persist(persist_dir=storage_dir)
        
        if verbose:
            print(f"✓ Index saved to {storage_dir}")
            print(f"  Future runs will load from disk (much faster!)")

        if return_stats:
            return index, None
        
        return index
    
    @staticmethod
    def get_or_create_index(documents=None, force_recreate: bool = False) -> VectorStoreIndex:
        """
        Get existing index from disk, or create a new one if it doesn't exist.
        
        This is the main entry point for loading/creating indexes.
        
        Args:
            documents: List of Document objects (required if index doesn't exist)
            force_recreate: If True, always create a fresh index (ignores existing)
        
        Returns:
            VectorStoreIndex ready for querying
            
        Why this approach:
            - Encapsulates the common workflow: "load if exists, else create"
            - Provides a single point for this decision
            - Makes it easy to add caching strategies in the future
            
        Example usage:
            # First time: creates and saves
            index = StorageManager.get_or_create_index(documents=docs)
            
            # Second time: loads from disk (fast!)
            index = StorageManager.get_or_create_index()
        """
        if force_recreate:
            print("Force recreating index (removing existing storage)...")
            storage_dir = RAGConfig.get_storage_dir()
            if os.path.exists(storage_dir):
                shutil.rmtree(storage_dir)
                os.makedirs(storage_dir)
        
        # Try to load existing index
        if StorageManager.index_exists():
            return StorageManager.load_index()
        
        # If no index exists, create one
        if documents is None:
            raise ValueError(
                "No existing index found, and documents were not provided. "
                "Please provide documents to create a new index."
            )
        
        return StorageManager.create_and_save_index(documents)
    
    @staticmethod
    def clear_storage() -> None:
        """
        Clear all stored indexes and data.
        
        Useful for testing or if you want a fresh start.
        
        Why this method:
            - Safe way to reset without manual file deletion
            - Useful for development/testing
            - Prevents accidental data loss (requires explicit call)
        """
        storage_dir = RAGConfig.get_storage_dir()
        
        if os.path.exists(storage_dir):
            print(f"Clearing storage at {storage_dir}...")
            shutil.rmtree(storage_dir)
            os.makedirs(storage_dir)
            print("✓ Storage cleared")
        else:
            print(f"No storage directory found at {storage_dir}")
