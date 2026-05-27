#!/usr/bin/env python3
"""Smoke-test LM Studio embedding path used by this repository."""

import os
import sys

from dotenv import load_dotenv
from llama_index.core import Settings

# Ensure project root is importable as `src`.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config.settings import RAGConfig  # noqa: E402


def run_smoke(text: str = "LM Studio embedding smoke test") -> tuple[bool, str]:
    """Return (ok, message) after making one embedding call."""
    try:
        RAGConfig.initialize_settings(include_llm=False)
        vector = Settings.embed_model.get_text_embedding(text)
        dimension = len(vector) if vector else 0
        if dimension <= 0:
            return False, "Embedding returned an empty vector"
        return (
            True,
            f"Embedding OK (dimension={dimension}, model={RAGConfig.EMBEDDING_MODEL})",
        )
    except Exception as exc:
        return False, f"Embedding smoke failed: {exc}"


def main() -> int:
    """Run smoke check from CLI."""
    load_dotenv()
    RAGConfig.load_from_env()

    if RAGConfig.EMBEDDING_PROVIDER != "lmstudio":
        print("SKIP: RAG_EMBEDDING_PROVIDER is not lmstudio")
        return 0

    ok, message = run_smoke()
    print(("OK: " if ok else "ERROR: ") + message)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
