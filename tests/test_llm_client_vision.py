import unittest
from unittest import mock
from src.llm import client as C


class TestChatVision(unittest.TestCase):
    def test_builds_multimodal_message(self):
        fake = mock.MagicMock()
        fake.chat.completions.create.return_value = mock.MagicMock(
            choices=[mock.MagicMock(message=mock.MagicMock(content="  hi  "))])
        out = C.chat_vision(fake, "gemma", "describe", b"\x89PNG", "image/png")
        self.assertEqual(out, "hi")
        msgs = fake.chat.completions.create.call_args.kwargs["messages"]
        content = msgs[0]["content"]
        self.assertEqual(content[0]["type"], "text")
        self.assertEqual(content[1]["type"], "image_url")
        import base64
        self.assertEqual(
            content[1]["image_url"]["url"],
            "data:image/png;base64," + base64.b64encode(b"\x89PNG").decode("ascii"))


if __name__ == "__main__":
    unittest.main()
