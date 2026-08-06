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

# The historical placeholder used against auth-less local servers (LM Studio,
# Ollama). It is a valid bearer *only* for servers that ignore auth. When a real
# key is required and this sentinel is sent, the server answers 401 (issue #83).
_PLACEHOLDER_KEY = "lm-studio"

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
    """Endpoint API key.

    Reads ``RAG_LLM_API_KEY`` and falls back to the ``lm-studio`` placeholder,
    which keeps the OpenAI client happy against **auth-less** local servers. If
    the endpoint actually enforces auth (some LM Studio / NIM / cloud configs
    do), the placeholder is rejected with a 401 — that is issue #83, and the
    remedy is to set ``RAG_LLM_API_KEY`` in the environment / MCP server config.
    ``healthcheck()`` surfaces that as a clear, actionable error at startup
    rather than an opaque 401 per query.
    """
    return os.getenv("RAG_LLM_API_KEY", "").strip() or _PLACEHOLDER_KEY


def using_placeholder_key() -> bool:
    """True when no real ``RAG_LLM_API_KEY`` is configured (sentinel in use)."""
    return _resolve_api_key() == _PLACEHOLDER_KEY


class LLMHealthcheckError(RuntimeError):
    """Raised when the configured LLM endpoint/key is not usable.

    Carries a human-actionable message (naming ``RAG_LLM_API_KEY`` when a 401 /
    auth failure is the likely cause) so the MCP server / answer path can fail
    loudly at init instead of returning a raw 401 on every query.
    """


def _request_timeout() -> float:
    """Per-request timeout in seconds (``RAG_LLM_TIMEOUT``, default 120)."""
    try:
        return float(os.getenv("RAG_LLM_TIMEOUT", "") or 120.0)
    except ValueError:
        return 120.0


def _max_retries() -> int:
    """Client-side retries per request (``RAG_LLM_MAX_RETRIES``, default 2)."""
    try:
        return max(0, int(os.getenv("RAG_LLM_MAX_RETRIES", "") or 2))
    except ValueError:
        return 2


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
                    # Bounded so an unreachable endpoint fails in seconds rather
                    # than minutes of exponential backoff. A sweep over
                    # thousands of emails against a dead endpoint would
                    # otherwise take hours to report what it knew immediately;
                    # the scheduled run simply retries on the next tick.
                    timeout=_request_timeout(),
                    max_retries=_max_retries(),
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


def _looks_like_auth_error(exc: BaseException) -> bool:
    """Heuristic: does ``exc`` look like a 401/403 auth failure?

    We match on the string form because the OpenAILike stack wraps the
    underlying ``openai`` error in a variety of exception types; a substring
    check on the status/keywords is robust across those wrappers.
    """
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(
        marker in text
        for marker in ("401", "403", "unauthorized", "malformed", "api key", "api token", "invalid")
    )


def _auth_hint() -> str:
    """The actionable remediation line appended to auth-failure messages."""
    if using_placeholder_key():
        return (
            "the endpoint requires authentication but no RAG_LLM_API_KEY is set "
            "(the 'lm-studio' placeholder was sent and rejected). Set RAG_LLM_API_KEY "
            "in the environment / MCP server config to a valid key for this endpoint."
        )
    return (
        "RAG_LLM_API_KEY is set but was rejected by the endpoint. Check the key value "
        "and that it is valid for this RAG_LLM_API_BASE."
    )


def chat(client: _LLMClient, model: str, prompt: str, temperature: float = 0.0) -> str:
    """Send a single-turn prompt through the OpenAILike LLM; return stripped text.

    On an authentication failure (e.g. the sentinel key rejected with 401) the
    raw error is re-raised as :class:`LLMHealthcheckError` with a clear message
    naming ``RAG_LLM_API_KEY``, rather than leaking an opaque provider 401.
    """
    try:
        return client.llm(model, temperature).complete(prompt).text.strip()
    except Exception as exc:  # noqa: BLE001 - re-raised with a clearer message
        if _looks_like_auth_error(exc):
            raise LLMHealthcheckError(
                f"LLM request rejected by {client.base_url}: {_auth_hint()}"
            ) from exc
        raise


def healthcheck(client: _LLMClient | None = None, *, model: str | None = None) -> None:
    """Verify the configured LLM endpoint + key are reachable and authorized.

    Sends one tiny completion. On success returns ``None``. On failure raises
    :class:`LLMHealthcheckError` with an actionable message — auth failures name
    ``RAG_LLM_API_KEY``; a missing model names ``RAG_LLM_MODEL``; anything else
    reports the endpoint that could not be reached. Callers (the MCP server's
    answer path) use this to fail loudly at startup instead of per-query.

    ``client`` / ``model`` are injectable for tests so no live endpoint is hit.
    """
    client = client or make_client()
    model = model if model is not None else default_model()
    if not model:
        raise LLMHealthcheckError(
            "no chat model configured: set RAG_LLM_MODEL to the model id served by "
            f"{client.base_url}"
        )
    try:
        client.llm(model, 0.0).complete("ping")
    except LLMHealthcheckError:
        raise
    except Exception as exc:  # noqa: BLE001 - classified into a clear message
        if _looks_like_auth_error(exc):
            raise LLMHealthcheckError(
                f"LLM endpoint {client.base_url} rejected the credentials: {_auth_hint()}"
            ) from exc
        raise LLMHealthcheckError(
            f"LLM endpoint {client.base_url} (model {model!r}) is unreachable: {exc}. "
            "Check RAG_LLM_API_BASE and that the server is running."
        ) from exc


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
