import unittest
from unittest import mock
from src.llm.answer import answer_from_threads


class _Ctx:
    def __init__(self, text):
        self.text = text


class TestAnswerFromThreads(unittest.TestCase):
    def test_empty_contexts_short_circuits(self):
        self.assertEqual(answer_from_threads("q", []), "No relevant threads retrieved.")

    def test_uses_top_k_and_calls_llm(self):
        ctxs = [_Ctx("A"), _Ctx("B"), _Ctx("C"), _Ctx("D")]
        with mock.patch("src.llm.answer.make_client", return_value="CLIENT"), \
             mock.patch("src.llm.answer.default_model", return_value="M"), \
             mock.patch("src.llm.answer.chat", return_value="ANSWER") as chat:
            out = answer_from_threads("q?", ctxs, k=2)
        self.assertEqual(out, "ANSWER")
        prompt = chat.call_args.args[2]
        self.assertIn("A", prompt)
        self.assertIn("B", prompt)
        self.assertNotIn("C", prompt)  # truncated to k=2


if __name__ == "__main__":
    unittest.main()
