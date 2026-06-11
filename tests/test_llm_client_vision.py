import unittest
from unittest import mock
from src.llm import client as C


class _RawHolder:
    """Stands in for ``_LLMClient`` — exposes the raw-OpenAI shim used by vision."""

    def __init__(self, raw):
        self._raw = raw

    def raw_openai(self):
        return self._raw


class TestChatVision(unittest.TestCase):
    """chat_vision() keeps a raw-OpenAI multimodal shim (OpenAILike can't carry
    inline images). It must build the text+image_url message and strip output."""

    def test_builds_multimodal_message_via_raw_openai(self):
        raw = mock.MagicMock()
        raw.chat.completions.create.return_value = mock.MagicMock(
            choices=[mock.MagicMock(message=mock.MagicMock(content="  hi  "))])
        client = _RawHolder(raw)

        out = C.chat_vision(client, "gemma", "describe", b"\x89PNG", "image/png")

        self.assertEqual(out, "hi")
        kwargs = raw.chat.completions.create.call_args.kwargs
        self.assertEqual(kwargs["model"], "gemma")
        content = kwargs["messages"][0]["content"]
        self.assertEqual(content[0]["type"], "text")
        self.assertEqual(content[0]["text"], "describe")
        self.assertEqual(content[1]["type"], "image_url")
        import base64
        self.assertEqual(
            content[1]["image_url"]["url"],
            "data:image/png;base64," + base64.b64encode(b"\x89PNG").decode("ascii"))


if __name__ == "__main__":
    unittest.main()
