import unittest
from unittest import mock

from src.llm import rubrics
from src.llm.rubrics import Rubric

_EMAIL = {"sender": "a@x.com", "subject": "Hi", "date": "2020", "body": "hello body"}


class TestBuildJudgePrompt(unittest.TestCase):
    def test_uses_judge_template_when_present(self):
        r = Rubric(
            name="t",
            template="FULL summarize {sender} {body}",
            judge_template="JUDGE verdict-only {sender} {subject} {body}",
        )
        with mock.patch("src.llm.rubrics.load_rubric", return_value=r):
            prompt = rubrics.build_judge_prompt("t", _EMAIL)
        self.assertIn("JUDGE verdict-only", prompt)
        self.assertNotIn("summarize", prompt)
        self.assertIn("a@x.com", prompt)
        self.assertIn("hello body", prompt)

    def test_falls_back_to_full_template_and_warns(self):
        r = Rubric(name="t", template="FULL summarize {sender} {body}", judge_template="")
        with mock.patch("src.llm.rubrics.load_rubric", return_value=r):
            with self.assertWarns(UserWarning):
                prompt = rubrics.build_judge_prompt("t", _EMAIL)
        self.assertIn("FULL summarize", prompt)

    def test_load_rubric_reads_judge_template(self):
        data = {
            "name": "demo",
            "template": "S {sender} {date} {subject} {body}",
            "judge_template": "J {sender} {date} {subject} {body}",
        }
        with (
            mock.patch("src.llm.rubrics._find", return_value=mock.Mock()),
            mock.patch("src.llm.rubrics.yaml.safe_load", return_value=data),
        ):
            r = rubrics.load_rubric("demo")
        self.assertEqual(r.judge_template, "J {sender} {date} {subject} {body}")


if __name__ == "__main__":
    unittest.main()
