"""Unit tests for embedding provider behavior in RAGConfig settings."""

import unittest
from unittest.mock import patch

import pytest

# These tests patch llama_index.embeddings.openai.OpenAIEmbedding; the optional
# embeddings integration must be importable to patch it. Skip cleanly when it's
# absent so the suite passes out of the box on a minimal env (#44).
pytest.importorskip("llama_index.embeddings.openai")

from src.config.settings import RAGConfig


class TestEmbeddingProviderSettings(unittest.TestCase):
    """Validate provider-specific embedding initialization and key checks."""

    def setUp(self):
        self._snapshot = {
            "LLM_PROVIDER": RAGConfig.LLM_PROVIDER,
            "LLM_MODEL": RAGConfig.LLM_MODEL,
            "LLM_TEMPERATURE": RAGConfig.LLM_TEMPERATURE,
            "LLM_API_BASE": RAGConfig.LLM_API_BASE,
            "LLM_API_KEY": RAGConfig.LLM_API_KEY,
            "EMBEDDING_PROVIDER": RAGConfig.EMBEDDING_PROVIDER,
            "EMBEDDING_MODEL": RAGConfig.EMBEDDING_MODEL,
            "EMBEDDING_API_BASE": RAGConfig.EMBEDDING_API_BASE,
            "EMBEDDING_API_KEY": RAGConfig.EMBEDDING_API_KEY,
            "CHUNK_SIZE": RAGConfig.CHUNK_SIZE,
            "CHUNK_OVERLAP": RAGConfig.CHUNK_OVERLAP,
        }

    def tearDown(self):
        for key, value in self._snapshot.items():
            setattr(RAGConfig, key, value)

    @patch.dict(
        "os.environ",
        {
            "RAG_EMBEDDING_PROVIDER": "lmstudio",
            "RAG_EMBEDDING_MODEL": "text-embedding-nomic-embed-text-v1.5",
            "RAG_EMBEDDING_API_BASE": "http://host.docker.internal:1234/v1",
        },
        clear=True,
    )
    @patch("src.config.settings.Settings")
    @patch("llama_index.embeddings.openai.OpenAIEmbedding")
    def test_lmstudio_embedding_does_not_require_openai_key(self, mock_embedding, _mock_settings):
        """LM Studio mode should initialize embeddings without OPENAI_API_KEY."""
        RAGConfig.initialize_settings(include_llm=False)

        mock_embedding.assert_called_once()
        kwargs = mock_embedding.call_args.kwargs
        model_value = kwargs.get("model_name") or kwargs.get("model")
        self.assertEqual(model_value, "text-embedding-nomic-embed-text-v1.5")
        self.assertEqual(kwargs["api_base"], "http://host.docker.internal:1234/v1")
        self.assertNotIn("api_key", kwargs)

    @patch.dict(
        "os.environ",
        {
            "RAG_EMBEDDING_PROVIDER": "openai",
            "RAG_EMBEDDING_MODEL": "text-embedding-3-small",
        },
        clear=True,
    )
    def test_openai_embedding_requires_openai_api_key(self):
        """OpenAI embedding mode must fail when OPENAI_API_KEY is absent."""
        with self.assertRaisesRegex(ValueError, "OPENAI_API_KEY environment variable is not set"):
            RAGConfig.initialize_settings(include_llm=False)

    @patch.dict(
        "os.environ",
        {
            "RAG_EMBEDDING_PROVIDER": "invalid-provider",
            "OPENAI_API_KEY": "test-key",
        },
        clear=True,
    )
    @patch("src.config.settings.Settings")
    @patch("llama_index.embeddings.openai.OpenAIEmbedding")
    @patch("builtins.print")
    def test_invalid_embedding_provider_falls_back_to_default(
        self, mock_print, mock_embedding, _mock_settings
    ):
        """Invalid embedding provider should emit warning and use default provider."""
        RAGConfig.EMBEDDING_PROVIDER = "openai"

        RAGConfig.initialize_settings(include_llm=False)

        self.assertEqual(RAGConfig.EMBEDDING_PROVIDER, "openai")
        mock_embedding.assert_called_once()
        self.assertTrue(
            any("Invalid RAG_EMBEDDING_PROVIDER" in str(call) for call in mock_print.call_args_list)
        )

    @patch.dict(
        "os.environ",
        {
            "RAG_LLM_PROVIDER": "lmstudio",
            "RAG_LLM_MODEL": "mistralai/magistral-small-2509",
            "RAG_LLM_API_BASE": "http://host.docker.internal:1234/v1",
            # embeddings also via LM Studio so no OPENAI_API_KEY is needed at all
            "RAG_EMBEDDING_PROVIDER": "lmstudio",
            "RAG_EMBEDDING_MODEL": "text-embedding-nomic-embed-text-v1.5",
            "RAG_EMBEDDING_API_BASE": "http://host.docker.internal:1234/v1",
        },
        clear=True,
    )
    @patch("src.config.settings.Settings")
    @patch("llama_index.embeddings.openai.OpenAIEmbedding")
    @patch("llama_index.llms.openai_like.OpenAILike")
    def test_lmstudio_llm_does_not_require_openai_key(
        self, mock_llm, _mock_embedding, _mock_settings
    ):
        """LM Studio LLM mode should initialize without OPENAI_API_KEY, using the
        configured api_base and a placeholder api_key. Post P2 Step-3 the answer
        side builds an OpenAILike (unified with the cleanup client)."""
        RAGConfig.initialize_settings(include_llm=True)

        mock_llm.assert_called_once()
        kwargs = mock_llm.call_args.kwargs
        self.assertEqual(kwargs["model"], "mistralai/magistral-small-2509")
        self.assertEqual(kwargs["api_base"], "http://host.docker.internal:1234/v1")
        # No real key supplied -> placeholder so the OpenAI client doesn't complain.
        self.assertEqual(kwargs["api_key"], "lm-studio")


if __name__ == "__main__":
    unittest.main()
