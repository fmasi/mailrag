"""
Configuration module for the Email RAG system.

This module centralizes all settings and uses the LlamaIndex Settings object
to configure the LLM, embedding model, and other system-wide parameters.
This makes it easy to swap components (e.g., different LLMs or embeddings)
without changing other parts of the codebase.
"""

import os
from typing import Optional

from llama_index.core import Settings


class RAGConfig:
    """
    Centralized configuration for the Email RAG system.
    
    This class manages all configuration parameters and provides methods
    to initialize the LlamaIndex Settings object with the appropriate
    LLM and embedding models.
    """
    
    # Storage and data paths
    STORAGE_DIR: str = "./storage"
    DATA_CACHE_DIR: str = "./data_cache"
    
    # LLM Configuration
    LLM_PROVIDER: str = "openai"  # Options: "openai", "perplexity", or "lmstudio"
    LLM_MODEL: str = "gpt-3.5-turbo"
    LLM_TEMPERATURE: float = 0.7
    LLM_API_BASE: str = "https://api.openai.com/v1"
    LLM_API_KEY: str = ""
    
    # Embedding Configuration
    EMBEDDING_PROVIDER: str = "openai"  # Options: "openai" or "lmstudio"
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    EMBEDDING_API_BASE: str = "https://api.openai.com/v1"
    EMBEDDING_API_KEY: str = ""
    EMBEDDING_BATCH_SIZE: int = 100
    EMBEDDING_NUM_WORKERS: int = 1
    EMBEDDING_TIMEOUT: int = 1800  # seconds; covers LM Studio cold-start (model load)
    
    # Dataset Configuration
    DATASET_NAME: str = "MichaelR207/enron_qa_0922"
    DATASET_SPLIT: str = "train"
    
    # Chunking and indexing parameters
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 64
    
    # Azure Blob Storage (Phase 1)
    AZURE_STORAGE_CONNECTION_STRING: str = ""
    AZURE_BLOB_CONTAINER: str = "eml-archive"
    AZURE_BLOB_PREFIX: str = ""
    
    # Vector Store Configuration (Phase 2)
    VECTOR_STORE_PROVIDER: str = "simple"  # "simple", "pinecone", or "qdrant"
    PINECONE_API_KEY: str = ""
    PINECONE_INDEX_NAME: str = "email-rag"
    QDRANT_URL: str = ""
    QDRANT_API_KEY: str = ""
    QDRANT_COLLECTION_NAME: str = "mailrag-demo"
    QDRANT_PREFER_GRPC: bool = False
    
    @staticmethod
    def load_from_env() -> None:
        """
        Load configuration from environment variables.

        This allows overriding defaults via a `.env` file or environment variables
        without changing code.

        Supported environment variables:
            - RAG_LLM_PROVIDER (str): LLM provider, one of {"openai", "perplexity", "lmstudio"}
            - RAG_LLM_MODEL (str): LLM model name
            - RAG_LLM_TEMPERATURE (float): LLM sampling temperature
            - RAG_LLM_API_BASE (str): LLM API base URL (for lmstudio: http://host.docker.internal:1234/v1)
            - RAG_LLM_API_KEY (str): LLM API key (optional for lmstudio)
            - RAG_EMBEDDING_PROVIDER (str): embedding provider, one of {"openai", "lmstudio"}
            - RAG_EMBEDDING_MODEL (str): embedding model name
            - RAG_EMBEDDING_API_BASE (str): embedding API base URL
            - RAG_EMBEDDING_API_KEY (str): embedding API key (optional for lmstudio)
            - RAG_EMBEDDING_BATCH_SIZE (int): number of texts per embedding request batch
            - RAG_EMBEDDING_NUM_WORKERS (int): embedding worker count for supported providers
            - RAG_CHUNK_SIZE (int): document chunk size for indexing
            - RAG_CHUNK_OVERLAP (int): overlap between adjacent chunks
            - STORAGE_DIR (str): path to persist index/storage artifacts
        """

        def _env_str(var_name: str, current_value: str) -> str:
            value = os.getenv(var_name)
            if value is None:
                return current_value

            value = value.strip()
            if not value:
                print(f"Warning: {var_name} is empty. Using default '{current_value}'.")
                return current_value

            return value

        def _env_int(var_name: str, current_value: int) -> int:
            value = os.getenv(var_name)
            if value is None:
                return current_value

            try:
                return int(value)
            except ValueError:
                print(f"Warning: Invalid {var_name}='{value}'. Using default {current_value}.")
                return current_value

        def _env_float(var_name: str, current_value: float) -> float:
            value = os.getenv(var_name)
            if value is None:
                return current_value

            try:
                return float(value)
            except ValueError:
                print(f"Warning: Invalid {var_name}='{value}'. Using default {current_value}.")
                return current_value

        def _env_bool(var_name: str, current_value: bool) -> bool:
            value = os.getenv(var_name)
            if value is None:
                return current_value

            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "on"}:
                return True
            if normalized in {"0", "false", "no", "off"}:
                return False

            print(f"Warning: Invalid {var_name}='{value}'. Using default {current_value}.")
            return current_value

        llm_provider = _env_str("RAG_LLM_PROVIDER", RAGConfig.LLM_PROVIDER).lower()
        if llm_provider in {"openai", "perplexity", "lmstudio"}:
            RAGConfig.LLM_PROVIDER = llm_provider
        else:
            print(
                f"Warning: Invalid RAG_LLM_PROVIDER='{llm_provider}'. "
                f"Using default '{RAGConfig.LLM_PROVIDER}'."
            )

        RAGConfig.LLM_MODEL = _env_str("RAG_LLM_MODEL", RAGConfig.LLM_MODEL)
        RAGConfig.LLM_API_BASE = _env_str("RAG_LLM_API_BASE", RAGConfig.LLM_API_BASE)
        RAGConfig.LLM_API_KEY = _env_str("RAG_LLM_API_KEY", RAGConfig.LLM_API_KEY)
        RAGConfig.STORAGE_DIR = _env_str("STORAGE_DIR", RAGConfig.STORAGE_DIR)

        embedding_provider = _env_str(
            "RAG_EMBEDDING_PROVIDER", RAGConfig.EMBEDDING_PROVIDER
        ).lower()
        if embedding_provider in {"openai", "lmstudio"}:
            RAGConfig.EMBEDDING_PROVIDER = embedding_provider
        else:
            print(
                f"Warning: Invalid RAG_EMBEDDING_PROVIDER='{embedding_provider}'. "
                f"Using default '{RAGConfig.EMBEDDING_PROVIDER}'."
            )

        RAGConfig.EMBEDDING_MODEL = _env_str("RAG_EMBEDDING_MODEL", RAGConfig.EMBEDDING_MODEL)
        RAGConfig.EMBEDDING_API_BASE = _env_str(
            "RAG_EMBEDDING_API_BASE", RAGConfig.EMBEDDING_API_BASE
        )
        RAGConfig.EMBEDDING_API_KEY = _env_str(
            "RAG_EMBEDDING_API_KEY", RAGConfig.EMBEDDING_API_KEY
        )
        RAGConfig.EMBEDDING_BATCH_SIZE = _env_int(
            "RAG_EMBEDDING_BATCH_SIZE", RAGConfig.EMBEDDING_BATCH_SIZE
        )
        RAGConfig.EMBEDDING_NUM_WORKERS = _env_int(
            "RAG_EMBEDDING_NUM_WORKERS", RAGConfig.EMBEDDING_NUM_WORKERS
        )
        RAGConfig.EMBEDDING_TIMEOUT = _env_int(
            "RAG_EMBEDDING_TIMEOUT", RAGConfig.EMBEDDING_TIMEOUT
        )
        RAGConfig.CHUNK_SIZE = _env_int("RAG_CHUNK_SIZE", RAGConfig.CHUNK_SIZE)
        RAGConfig.CHUNK_OVERLAP = _env_int("RAG_CHUNK_OVERLAP", RAGConfig.CHUNK_OVERLAP)
        RAGConfig.LLM_TEMPERATURE = _env_float("RAG_LLM_TEMPERATURE", RAGConfig.LLM_TEMPERATURE)

        # Azure Blob Storage
        RAGConfig.AZURE_STORAGE_CONNECTION_STRING = _env_str(
            "AZURE_STORAGE_CONNECTION_STRING", RAGConfig.AZURE_STORAGE_CONNECTION_STRING
        )
        RAGConfig.AZURE_BLOB_CONTAINER = _env_str(
            "AZURE_BLOB_CONTAINER", RAGConfig.AZURE_BLOB_CONTAINER
        )
        RAGConfig.AZURE_BLOB_PREFIX = _env_str(
            "AZURE_BLOB_PREFIX", RAGConfig.AZURE_BLOB_PREFIX
        )

        # Vector Store Provider
        vector_provider = _env_str(
            "VECTOR_STORE_PROVIDER", RAGConfig.VECTOR_STORE_PROVIDER
        ).lower()
        if vector_provider in {"simple", "pinecone", "qdrant"}:
            RAGConfig.VECTOR_STORE_PROVIDER = vector_provider
        else:
            print(
                f"Warning: Invalid VECTOR_STORE_PROVIDER='{vector_provider}'. "
                f"Using default '{RAGConfig.VECTOR_STORE_PROVIDER}'."
            )
        RAGConfig.PINECONE_API_KEY = _env_str(
            "PINECONE_API_KEY", RAGConfig.PINECONE_API_KEY
        )
        RAGConfig.PINECONE_INDEX_NAME = _env_str(
            "PINECONE_INDEX_NAME", RAGConfig.PINECONE_INDEX_NAME
        )
        RAGConfig.QDRANT_URL = _env_str("QDRANT_URL", RAGConfig.QDRANT_URL)
        RAGConfig.QDRANT_API_KEY = _env_str("QDRANT_API_KEY", RAGConfig.QDRANT_API_KEY)
        RAGConfig.QDRANT_COLLECTION_NAME = _env_str(
            "QDRANT_COLLECTION_NAME", RAGConfig.QDRANT_COLLECTION_NAME
        )
        RAGConfig.QDRANT_PREFER_GRPC = _env_bool(
            "QDRANT_PREFER_GRPC", RAGConfig.QDRANT_PREFER_GRPC
        )
    
    @staticmethod
    def initialize_settings(
        include_llm: bool = True,
        include_embeddings: bool = True,
    ) -> None:
        """
        Initialize the LlamaIndex Settings object with configured LLM and embeddings.

        This is where we set up the global LlamaIndex settings. By centralizing this,
        we avoid having to pass these objects around throughout the codebase.
        Instead, LlamaIndex components will use these global settings automatically.

        Args:
            include_llm: When True (default), build and set Settings.llm.  Pass
                False to skip LLM construction (e.g. indexing-only pipelines).
            include_embeddings: When True (default), build and set
                Settings.embed_model.  Pass False when the caller manages its own
                embedding model directly (e.g. the novel demo uses bge-m3 via
                FlagEmbedding and does not need the LlamaIndex embed model).
                Skipping also avoids importing ``llama_index.embeddings.openai``,
                which is not available in all environments.

        Both the LLM and embedding imports are lazy (deferred to call time) so
        that ``import src.config.settings`` is safe in environments that have
        ``llama-index-core`` but not the optional LLM / embedding integrations.
        """
        # Load config from environment variables
        RAGConfig.load_from_env()

        llm = None
        if include_llm:
            # Lazy imports: only require llama_index.llms.* when we actually
            # build the LLM.  This keeps ``import src.config.settings`` safe in
            # environments (e.g. the novel demo's ``rag`` conda env) that have
            # llama-index-core but NOT llama-index-llms-openai/perplexity.
            if RAGConfig.LLM_PROVIDER == "openai":
                from llama_index.llms.openai import OpenAI  # noqa: PLC0415
                openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()
                if not openai_api_key:
                    raise ValueError("OPENAI_API_KEY environment variable is not set")
                llm = OpenAI(
                    model=RAGConfig.LLM_MODEL,
                    temperature=RAGConfig.LLM_TEMPERATURE,
                    api_key=openai_api_key,
                )
            elif RAGConfig.LLM_PROVIDER == "perplexity":
                from llama_index.llms.perplexity import Perplexity  # noqa: PLC0415
                perplexity_api_key = os.getenv("PERPLEXITY_API_KEY", "").strip()
                if not perplexity_api_key:
                    raise ValueError("PERPLEXITY_API_KEY environment variable is not set")
                llm = Perplexity(
                    model=RAGConfig.LLM_MODEL,
                    temperature=RAGConfig.LLM_TEMPERATURE,
                    api_key=perplexity_api_key,
                )
            elif RAGConfig.LLM_PROVIDER == "lmstudio":
                from llama_index.llms.openai import OpenAI  # noqa: PLC0415
                llm = OpenAI(
                    model=RAGConfig.LLM_MODEL,
                    temperature=RAGConfig.LLM_TEMPERATURE,
                    api_base=RAGConfig.LLM_API_BASE,
                    # LM Studio does not require auth; pass a placeholder so the
                    # OpenAI client doesn't complain about a missing api_key.
                    api_key=RAGConfig.LLM_API_KEY or "lm-studio",
                )
            else:
                raise ValueError(f"Unknown LLM provider: {RAGConfig.LLM_PROVIDER}")

        if include_embeddings:
            # Lazy import: only require llama_index.embeddings.openai when we
            # actually need to build the embed model.  This keeps
            # ``import src.config.settings`` safe in environments (e.g. the
            # novel demo's ``rag`` conda env) that have FlagEmbedding/bge-m3
            # but NOT llama_index.embeddings.openai.
            from llama_index.embeddings.openai import OpenAIEmbedding  # noqa: PLC0415

            if RAGConfig.EMBEDDING_PROVIDER == "openai":
                openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()
                if not openai_api_key:
                    raise ValueError("OPENAI_API_KEY environment variable is not set")
                embed_model = OpenAIEmbedding(
                    model=RAGConfig.EMBEDDING_MODEL,
                    api_key=openai_api_key,
                    embed_batch_size=RAGConfig.EMBEDDING_BATCH_SIZE,
                    num_workers=RAGConfig.EMBEDDING_NUM_WORKERS,
                )
            elif RAGConfig.EMBEDDING_PROVIDER == "lmstudio":
                embedding_kwargs = {
                    # LlamaIndex OpenAIEmbedding validates `model` against OpenAI enums.
                    # Use `model_name` to support arbitrary OpenAI-compatible model IDs
                    # served by LM Studio (e.g., text-embedding-bge-m3).
                    "model_name": RAGConfig.EMBEDDING_MODEL,
                    "api_base": RAGConfig.EMBEDDING_API_BASE,
                    "embed_batch_size": RAGConfig.EMBEDDING_BATCH_SIZE,
                    "num_workers": RAGConfig.EMBEDDING_NUM_WORKERS,
                    # LM Studio can take several minutes on a cold start (model load).
                    # The OpenAI client default is 600s, which is not enough when the
                    # GPU has to pull a large model into VRAM before the first inference.
                    "timeout": RAGConfig.EMBEDDING_TIMEOUT,
                }
                # LM Studio OpenAI-compatible endpoints usually do not require auth.
                if RAGConfig.EMBEDDING_API_KEY:
                    embedding_kwargs["api_key"] = RAGConfig.EMBEDDING_API_KEY
                embed_model = OpenAIEmbedding(**embedding_kwargs)
            else:
                raise ValueError(
                    f"Unknown embedding provider: {RAGConfig.EMBEDDING_PROVIDER}. "
                    "Expected one of: openai, lmstudio."
                )

            Settings.embed_model = embed_model

        # Set the global LlamaIndex settings
        # Now all LlamaIndex components will use these settings by default
        if include_llm:
            Settings.llm = llm
        Settings.chunk_size = RAGConfig.CHUNK_SIZE
        Settings.chunk_overlap = RAGConfig.CHUNK_OVERLAP
    
    @staticmethod
    def get_storage_dir() -> str:
        """Get the storage directory path, creating it if it doesn't exist."""
        os.makedirs(RAGConfig.STORAGE_DIR, exist_ok=True)
        return RAGConfig.STORAGE_DIR
    
    @staticmethod
    def get_data_cache_dir() -> str:
        """Get the data cache directory path, creating it if it doesn't exist."""
        os.makedirs(RAGConfig.DATA_CACHE_DIR, exist_ok=True)
        return RAGConfig.DATA_CACHE_DIR
