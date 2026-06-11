# tests/test_llm_client.py
"""Tests for the LM Studio chat client (no network; fake client object)."""
import os
import unittest
from unittest import mock

from src.llm import client


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


class _FakeClient:
    def __init__(self, content):
        self.chat = _FakeChat(content)


class TestChat(unittest.TestCase):
    def test_returns_stripped_content(self):
        c = _FakeClient("  hello  ")
        self.assertEqual(client.chat(c, "m", "prompt"), "hello")

    def test_passes_model_and_prompt(self):
        c = _FakeClient("ok")
        client.chat(c, "gemma-x", "the prompt")
        kw = c.chat.completions.last_kwargs
        self.assertEqual(kw["model"], "gemma-x")
        self.assertEqual(kw["messages"], [{"role": "user", "content": "the prompt"}])
        self.assertEqual(kw["temperature"], 0.0)


class TestMakeClientConfig(unittest.TestCase):
    """The cleanup-pipeline client must read the SAME endpoint env var as the
    LlamaIndex answer-side (RAGConfig.Settings.llm), so configuring one configures
    both. Canonical var is RAG_LLM_API_BASE; RAG_LLM_BASE_URL stays a legacy alias."""

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
