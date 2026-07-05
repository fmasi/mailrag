# src/llm/client.py
"""Unified LLM client for the cleanup pipeline.

Single-turn text completions are routed through a LlamaIndex ``OpenAILike`` LLM
— the *same* abstraction the answer side uses via ``Settings.llm`` — so there is
one LLM stack and one endpoint configuration (``RAG_LLM_API_BASE``) for the whole
system. Any OpenAI-compatible server works: LM Studio (local, the default),
NVIDIA NIM, Ollama, vLLM, OpenAI itself.

The inline-image vision path (:func:`chat_vision`) keeps a thin *raw-OpenAI* shim
because ``OpenAILike`` does not carry multimodal ``image_url`` content. Production
stays fully offline by default; the only cloud caller is the dev-only ``eval``
harness (separate reference client).
"""

from __future__ import annotations

import os
import threading

# Generous context window so LlamaIndex never silently truncates our cleanup
# prompts (the old raw-OpenAI path did no truncation). Cleanup prompts are small
# (~a few thousand chars); this just disables OpenAILike's defensive trimming.
_CONTEXT_WINDOW = 32768


def resolve_llm_api_base() -> str:
    """The configured chat endpoint, local-first by default.

    Reads the **single canonical** variable ``RAG_LLM_API_BASE`` — the same one
    ``RAGConfig`` / ``Settings.llm`` use — so configuring the LLM once points
    *both* the LlamaIndex answer side and this cleanup client at the same server.
    ``RAG_LLM_BASE_URL`` is kept as a legacy alias. A blank or whitespace-only
    value is treated as unset (so it falls through to the alias / default rather
    than yielding an empty URL).
    """
    for var in ("RAG_LLM_API_BASE", "RAG_LLM_BASE_URL"):  # canonical, then legacy alias
        value = (os.getenv(var) or "").strip()
        if value:
            return value
    return "http://localhost:1234/v1"


def _resolve_api_key() -> str:
    """Endpoint API key; ``lm-studio`` placeholder keeps the OpenAI client happy
    against auth-less local servers."""
    return os.getenv("RAG_LLM_API_KEY", "").strip() or "lm-studio"


class _LLMClient:
    """Holds endpoint config and lazily builds one ``OpenAILike`` per
    ``(model, temperature)``.

    Building the LLM once per model (rather than per call) matters for the bulk
    cleanup sweeps, which issue thousands of completions across worker threads
    over a single shared client. ``.base_url`` is exposed so callers (and tests)
    can read the configured endpoint, matching the old raw-OpenAI client.
    """

    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.api_key = api_key
        self._cache: dict[tuple[str, float], object] = {}
        self._lock = threading.Lock()

    def llm(self, model: str, temperature: float = 0.0):
        """The ``OpenAILike`` LLM for ``model`` at ``temperature`` (cached)."""
        key = (model, temperature)
        with self._lock:
            inst = self._cache.get(key)
            if inst is None:
                from llama_index.llms.openai_like import OpenAILike  # noqa: PLC0415

                inst = OpenAILike(
                    model=model,
                    api_base=self.base_url,
                    api_key=self.api_key,
                    # LM Studio / NIM / Ollama all serve chat models; route
                    # complete() through /chat/completions like the old client.
                    is_chat_model=True,
                    temperature=temperature,
                    context_window=_CONTEXT_WINDOW,
                )
                self._cache[key] = inst
            return inst

    def raw_openai(self):
        """A raw ``openai`` SDK client for the multimodal vision shim."""
        from openai import OpenAI  # noqa: PLC0415

        return OpenAI(base_url=self.base_url, api_key=self.api_key)


def make_client() -> _LLMClient:
    """Build the unified LLM client for the configured chat endpoint."""
    return _LLMClient(resolve_llm_api_base(), _resolve_api_key())


def default_model() -> str:
    """The configured chat model id (``RAG_LLM_MODEL``)."""
    return os.getenv("RAG_LLM_MODEL", "").strip()


def chat(client: _LLMClient, model: str, prompt: str, temperature: float = 0.0) -> str:
    """Send a single-turn prompt through the OpenAILike LLM; return stripped text."""
    return client.llm(model, temperature).complete(prompt).text.strip()


def chat_vision(
    client: _LLMClient,
    model: str,
    prompt: str,
    image_bytes: bytes,
    mime: str,
    temperature: float = 0.0,
) -> str:
    """Single-turn multimodal prompt: text + one inline (base64) image.

    Uses the raw-OpenAI shim because ``OpenAILike`` does not carry ``image_url``
    multimodal content. Returns the stripped assistant text.
    """
    import base64  # noqa: PLC0415

    b64 = base64.b64encode(image_bytes).decode("ascii")
    resp = client.raw_openai().chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                ],
            }
        ],
        temperature=temperature,
    )
    return resp.choices[0].message.content.strip()
