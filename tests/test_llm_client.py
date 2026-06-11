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
        with mock.patch.dict(os.environ, {"RAG_LLM_API_BASE": "http://unified:9999/v1"}, clear=True):
            c = client.make_client()
        self.assertEqual(self._base(c), "http://unified:9999/v1")

    def test_api_base_takes_precedence_over_legacy_base_url(self):
        with mock.patch.dict(os.environ,
                             {"RAG_LLM_API_BASE": "http://canonical:1/v1",
                              "RAG_LLM_BASE_URL": "http://legacy:2/v1"}, clear=True):
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


if __name__ == "__main__":
    unittest.main()
