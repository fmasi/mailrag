# tests/test_llm_client.py
"""Tests for the unified LLM client.

After the P2 Step-3 migration ``make_client()`` returns a thin ``_LLMClient``
that routes single-turn text completions through a LlamaIndex ``OpenAILike``
LLM (the same abstraction ``Settings.llm`` uses), while keeping a raw-OpenAI
shim for the inline-image vision path. No network here — the OpenAILike /
OpenAI boundary is faked.
"""

import os
import unittest
from unittest import mock

from src.llm import client


class _FakeCompletion:
    """Mimics a LlamaIndex CompletionResponse (``.text``)."""

    def __init__(self, text):
        self.text = text


class _FakeLLM:
    def __init__(self, text):
        self._text = text
        self.last_prompt = None

    def complete(self, prompt):
        self.last_prompt = prompt
        return _FakeCompletion(self._text)


class _FakeLLMClient:
    """Stands in for ``_LLMClient`` — records how ``chat()`` drives it."""

    def __init__(self, text):
        self._llm = _FakeLLM(text)
        self.llm_calls = []
        self.base_url = "http://localhost:1234/v1"

    def llm(self, model, temperature=0.0):
        self.llm_calls.append((model, temperature))
        return self._llm


class TestChat(unittest.TestCase):
    """chat() must delegate to the OpenAILike LLM and return stripped text."""

    def test_returns_stripped_content(self):
        c = _FakeLLMClient("  hello  ")
        self.assertEqual(client.chat(c, "m", "prompt"), "hello")

    def test_routes_model_temperature_and_prompt_through_llm(self):
        c = _FakeLLMClient("ok")
        client.chat(c, "gemma-x", "the prompt", temperature=0.0)
        self.assertEqual(c.llm_calls, [("gemma-x", 0.0)])
        self.assertEqual(c._llm.last_prompt, "the prompt")


class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)


class _FakeResp:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]


class _FakeCompletions:
    def __init__(self, content):
        self._content = content
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return _FakeResp(self._content)


class _FakeChat:
    def __init__(self, content):
        self.completions = _FakeCompletions(content)


class _FakeRawOpenAI:
    def __init__(self, content):
        self.chat = _FakeChat(content)


class _VisionClient:
    """Stands in for ``_LLMClient`` for the vision shim path."""

    def __init__(self, content):
        self._raw = _FakeRawOpenAI(content)

    def raw_openai(self):
        return self._raw


class TestMakeClientBuildsOpenAILike(unittest.TestCase):
    """make_client() must return a client whose .llm(model) is a real
    OpenAILike wired to the configured endpoint (construction only, no network)."""

    def test_llm_is_openai_like_wired_to_endpoint(self):
        with mock.patch.dict(
            os.environ, {"RAG_LLM_API_BASE": "http://unified:9999/v1"}, clear=True
        ):
            c = client.make_client()
            llm = c.llm("some-model", 0.0)
        from llama_index.llms.openai_like import OpenAILike

        self.assertIsInstance(llm, OpenAILike)
        self.assertEqual(llm.model, "some-model")
        self.assertEqual(str(llm.api_base).rstrip("/"), "http://unified:9999/v1")
        self.assertEqual(llm.temperature, 0.0)

    def test_llm_is_cached_per_model_and_temperature(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            c = client.make_client()
            a = c.llm("m", 0.0)
            b = c.llm("m", 0.0)
            d = c.llm("m", 0.7)
        self.assertIs(a, b, "same (model, temperature) must reuse one OpenAILike")
        self.assertIsNot(a, d, "different temperature must build a distinct LLM")


class TestMakeClientConfig(unittest.TestCase):
    """The client must read the SAME endpoint env var as the LlamaIndex
    answer-side (RAGConfig/Settings.llm), so configuring one configures both.
    Canonical var is RAG_LLM_API_BASE; RAG_LLM_BASE_URL stays a legacy alias."""

    @staticmethod
    def _base(c):
        return str(c.base_url).rstrip("/")

    def test_uses_unified_rag_llm_api_base(self):
        with mock.patch.dict(
            os.environ, {"RAG_LLM_API_BASE": "http://unified:9999/v1"}, clear=True
        ):
            c = client.make_client()
        self.assertEqual(self._base(c), "http://unified:9999/v1")

    def test_api_base_takes_precedence_over_legacy_base_url(self):
        with mock.patch.dict(
            os.environ,
            {"RAG_LLM_API_BASE": "http://canonical:1/v1", "RAG_LLM_BASE_URL": "http://legacy:2/v1"},
            clear=True,
        ):
            c = client.make_client()
        self.assertEqual(self._base(c), "http://canonical:1/v1")

    def test_legacy_base_url_still_honored(self):
        # back-compat: existing setups that only set RAG_LLM_BASE_URL keep working
        with mock.patch.dict(os.environ, {"RAG_LLM_BASE_URL": "http://legacy:1234/v1"}, clear=True):
            c = client.make_client()
        self.assertEqual(self._base(c), "http://legacy:1234/v1")

    def test_defaults_to_local_first_lmstudio(self):
        # local-first default preserved when nothing is set
        with mock.patch.dict(os.environ, {}, clear=True):
            c = client.make_client()
        self.assertEqual(self._base(c), "http://localhost:1234/v1")


class TestResolveLlmApiBase(unittest.TestCase):
    """The public endpoint resolver (shared by make_client and the eval tools,
    e.g. scripts/eval/bench_models.py). Canonical RAG_LLM_API_BASE wins; legacy
    RAG_LLM_BASE_URL is honored as a fallback; default is local-first."""

    def test_prefers_canonical_over_legacy(self):
        with mock.patch.dict(
            os.environ,
            {"RAG_LLM_API_BASE": "http://canon:9/v1", "RAG_LLM_BASE_URL": "http://legacy:2/v1"},
            clear=True,
        ):
            self.assertEqual(client.resolve_llm_api_base(), "http://canon:9/v1")

    def test_legacy_alias_fallback(self):
        with mock.patch.dict(os.environ, {"RAG_LLM_BASE_URL": "http://legacy:2/v1"}, clear=True):
            self.assertEqual(client.resolve_llm_api_base(), "http://legacy:2/v1")

    def test_local_first_default(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(client.resolve_llm_api_base(), "http://localhost:1234/v1")

    # A whitespace-only value must be treated as unset (fall through to the
    # alias / default), never yielding an empty URL. One scenario per test.
    def test_blank_canonical_falls_through_to_legacy(self):
        with mock.patch.dict(
            os.environ,
            {"RAG_LLM_API_BASE": "   ", "RAG_LLM_BASE_URL": "http://legacy:2/v1"},
            clear=True,
        ):
            self.assertEqual(client.resolve_llm_api_base(), "http://legacy:2/v1")

    def test_blank_canonical_alone_falls_through_to_default(self):
        with mock.patch.dict(os.environ, {"RAG_LLM_API_BASE": "  "}, clear=True):
            self.assertEqual(client.resolve_llm_api_base(), "http://localhost:1234/v1")

    def test_blank_legacy_falls_through_to_default(self):
        with mock.patch.dict(os.environ, {"RAG_LLM_BASE_URL": "  "}, clear=True):
            self.assertEqual(client.resolve_llm_api_base(), "http://localhost:1234/v1")

    def test_both_blank_falls_through_to_default(self):
        with mock.patch.dict(
            os.environ, {"RAG_LLM_API_BASE": " ", "RAG_LLM_BASE_URL": "  "}, clear=True
        ):
            self.assertEqual(client.resolve_llm_api_base(), "http://localhost:1234/v1")


class _RaisingLLM:
    """An LLM whose ``complete`` raises a chosen exception (endpoint failures)."""

    def __init__(self, exc):
        self._exc = exc

    def complete(self, prompt):
        raise self._exc


class _RaisingClient:
    """Stands in for ``_LLMClient`` but its ``.llm`` raises on ``complete``."""

    def __init__(self, exc, base_url="http://localhost:1234/v1"):
        self.base_url = base_url
        self._llm = _RaisingLLM(exc)

    def llm(self, model, temperature=0.0):
        return self._llm


class TestApiKeyResolution(unittest.TestCase):
    """The sentinel key is used only when RAG_LLM_API_KEY is unset (issue #83)."""

    def test_placeholder_when_unset(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(client._resolve_api_key(), "lm-studio")
            self.assertTrue(client.using_placeholder_key())

    def test_real_key_used_when_set(self):
        with mock.patch.dict(os.environ, {"RAG_LLM_API_KEY": "sk-real"}, clear=True):
            self.assertEqual(client._resolve_api_key(), "sk-real")
            self.assertFalse(client.using_placeholder_key())

    def test_blank_key_falls_back_to_placeholder(self):
        with mock.patch.dict(os.environ, {"RAG_LLM_API_KEY": "   "}, clear=True):
            self.assertTrue(client.using_placeholder_key())


class TestChatAuthError(unittest.TestCase):
    """A 401/auth failure becomes a clear, actionable LLMHealthcheckError."""

    def test_401_reraised_as_healthcheck_error_naming_key(self):
        exc = Exception("Error code: 401 - Malformed LM Studio API token")
        c = _RaisingClient(exc)
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(client.LLMHealthcheckError) as ctx:
                client.chat(c, "m", "hi")
        msg = str(ctx.exception)
        self.assertIn("RAG_LLM_API_KEY", msg)
        self.assertNotEqual(msg.strip(), "401")  # not a raw status leak

    def test_non_auth_error_propagates_unchanged(self):
        exc = RuntimeError("connection refused")
        c = _RaisingClient(exc)
        with self.assertRaises(RuntimeError):
            client.chat(c, "m", "hi")


class TestHealthcheck(unittest.TestCase):
    """healthcheck() fails loudly at init instead of a per-query 401 (issue #83)."""

    def test_ok_when_endpoint_responds(self):
        c = _FakeLLMClient("pong")
        # No RAG_LLM_MODEL needed when we pass model explicitly.
        self.assertIsNone(client.healthcheck(c, model="m"))

    def test_missing_model_raises_naming_model_var(self):
        c = _FakeLLMClient("x")
        with self.assertRaises(client.LLMHealthcheckError) as ctx:
            client.healthcheck(c, model="")
        self.assertIn("RAG_LLM_MODEL", str(ctx.exception))

    def test_auth_failure_names_api_key_and_placeholder(self):
        c = _RaisingClient(Exception("HTTP 401 Unauthorized"))
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(client.LLMHealthcheckError) as ctx:
                client.healthcheck(c, model="m")
        self.assertIn("RAG_LLM_API_KEY", str(ctx.exception))

    def test_unreachable_endpoint_names_base(self):
        c = _RaisingClient(RuntimeError("connection refused"), base_url="http://x:9/v1")
        with self.assertRaises(client.LLMHealthcheckError) as ctx:
            client.healthcheck(c, model="m")
        self.assertIn("http://x:9/v1", str(ctx.exception))
        self.assertIn("RAG_LLM_API_BASE", str(ctx.exception))

    def test_docker_host_gotcha_reported_as_unreachable(self):
        # A base URL pointing at host.docker.internal from the host (issue #29
        # class) surfaces as a clear unreachable error naming RAG_LLM_API_BASE,
        # not an opaque connection traceback.
        c = _RaisingClient(
            ConnectionError("Failed to establish a new connection: [Errno 61]"),
            base_url="http://host.docker.internal:1234/v1",
        )
        with self.assertRaises(client.LLMHealthcheckError) as ctx:
            client.healthcheck(c, model="m")
        msg = str(ctx.exception)
        self.assertIn("host.docker.internal", msg)
        self.assertIn("RAG_LLM_API_BASE", msg)


if __name__ == "__main__":
    unittest.main()


class TestAuthHeuristicIsNarrow(unittest.TestCase):
    """Auth failures are classified ENDPOINT-level, so they never spend a
    message's retry budget. That makes a false positive costly: a per-message
    rejection misread as auth would be re-judged every tick forever, never
    abandoned, and — since indexing waits on judging — never indexed."""

    def _looks_auth(self, msg, cls="APIStatusError"):
        from src.llm.client import _looks_like_auth_error

        return _looks_like_auth_error(type(cls, (Exception,), {})(msg))

    def test_a_real_401_is_detected(self):
        self.assertTrue(
            self._looks_auth("Error code: 401 - {'error': {'code': 'invalid_api_key'}}")
        )

    def test_a_403_is_detected(self):
        self.assertTrue(self._looks_auth("Error code: 403 - forbidden"))

    def test_an_lm_studio_token_rejection_is_detected(self):
        self.assertTrue(
            self._looks_auth("Malformed LM Studio API token provided", cls="AuthenticationError")
        )

    def test_an_over_length_prompt_is_NOT_auth(self):
        """Every OpenAI-spec 4xx body contains 'invalid_request_error'. Matching
        a bare 'invalid' made every per-message rejection look like auth."""
        self.assertFalse(
            self._looks_auth(
                "Error code: 400 - {'error': {'message': \"This model's maximum context "
                "length is 8192 tokens\", 'type': 'invalid_request_error'}}"
            )
        )

    def test_a_bad_parameter_is_NOT_auth(self):
        self.assertFalse(
            self._looks_auth(
                "Error code: 400 - {'error': {'message': 'Invalid value for temperature', "
                "'type': 'invalid_request_error'}}"
            )
        )

    def test_a_malformed_json_response_is_NOT_auth(self):
        self.assertFalse(self._looks_auth("malformed JSON in response body", cls="ValueError"))

    def test_a_token_count_containing_401_is_NOT_auth(self):
        """Reproduced by review: bare substring matching made a 400 whose body
        says "resulted in 40123 tokens" read as an auth failure — and auth is
        endpoint-level, so that message would defer forever."""
        self.assertFalse(
            self._looks_auth(
                "Error code: 400 - {'error': {'message': 'your messages resulted in "
                "40123 tokens, however the model supports at most 8192'}}"
            )
        )

    def test_a_real_status_code_still_matches_at_a_word_boundary(self):
        self.assertTrue(self._looks_auth("Error code: 401 - unauthorized"))
        self.assertTrue(self._looks_auth("unexpected status 403: forbidden"))
