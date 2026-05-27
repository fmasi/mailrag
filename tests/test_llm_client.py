# tests/test_llm_client.py
"""Tests for the LM Studio chat client (no network; fake client object)."""
import unittest

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


if __name__ == "__main__":
    unittest.main()
