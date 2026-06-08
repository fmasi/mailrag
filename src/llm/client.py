# src/llm/client.py
"""OpenAI-compatible chat client for the local LM Studio server (Gemma).

LM Studio exposes an OpenAI-shaped ``/v1/chat/completions`` endpoint, so we
reuse the ``openai`` SDK pointed at it. Production stays fully offline; the only
cloud caller is the dev-only ``eval`` harness (separate reference client).
"""
from __future__ import annotations

import os


def make_client():
    """Build an OpenAI SDK client pointed at the local LM Studio server."""
    from openai import OpenAI

    base_url = os.getenv("RAG_LLM_BASE_URL", "http://localhost:1234/v1").strip()
    api_key = os.getenv("RAG_LLM_API_KEY", "").strip() or "lm-studio"
    return OpenAI(base_url=base_url, api_key=api_key)


def default_model() -> str:
    """The configured Gemma model id (``RAG_LLM_MODEL``)."""
    return os.getenv("RAG_LLM_MODEL", "").strip()


def chat(client, model: str, prompt: str, temperature: float = 0.0) -> str:
    """Send a single-turn prompt and return the raw assistant text."""
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return resp.choices[0].message.content.strip()


def chat_vision(client, model: str, prompt: str, image_bytes: bytes, mime: str,
                temperature: float = 0.0) -> str:
    """Single-turn multimodal prompt: text + one inline (base64) image. Returns raw text."""
    import base64
    b64 = base64.b64encode(image_bytes).decode("ascii")
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
        ]}],
        temperature=temperature,
    )
    return resp.choices[0].message.content.strip()
