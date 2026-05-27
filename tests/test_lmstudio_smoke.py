"""Optional integration smoke test for LM Studio embeddings."""

import os

import pytest
from dotenv import load_dotenv
from llama_index.core import Settings

from src.config.settings import RAGConfig


@pytest.mark.integration
def test_lmstudio_embedding_smoke() -> None:
    """Verify one live LM Studio embedding call when explicitly enabled."""
    if os.getenv("RUN_LMSTUDIO_SMOKE", "0") != "1":
        pytest.skip("Set RUN_LMSTUDIO_SMOKE=1 to run LM Studio integration smoke test")

    load_dotenv(".env")
    RAGConfig.load_from_env()

    if RAGConfig.EMBEDDING_PROVIDER != "lmstudio":
        pytest.skip("LM Studio provider not configured")

    RAGConfig.initialize_settings(include_llm=False)
    vector = Settings.embed_model.get_text_embedding("smoke test")

    assert vector is not None
    assert len(vector) > 0
