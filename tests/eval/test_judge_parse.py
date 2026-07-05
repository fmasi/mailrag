# tests/eval/test_judge_parse.py
import unittest

from src.eval.judge_parse import RUBRIC, build_answer_judge_prompt, build_judge_prompt, parse_grade


class BuildPromptTest(unittest.TestCase):
    def test_prompt_contains_query_email_and_scale(self):
        p = build_judge_prompt("when do we meet?", "Subject: Re: sync\n\nTuesday 3pm works")
        self.assertIn("when do we meet?", p)
        self.assertIn("Tuesday 3pm works", p)
        self.assertIn("0", p)
        self.assertIn("3", p)
        self.assertIn(RUBRIC.strip().splitlines()[0], p)


class TestBuildAnswerJudgePrompt(unittest.TestCase):
    def test_embeds_query_answer_reference(self):
        p = build_answer_judge_prompt("the QUERY", "the ANSWER", "the REFERENCE")
        self.assertIn("the QUERY", p)
        self.assertIn("the ANSWER", p)
        self.assertIn("the REFERENCE", p)

    def test_asks_for_single_integer_grade(self):
        p = build_answer_judge_prompt("q", "a", "r")
        self.assertIn("0", p)
        self.assertIn("3", p)
        self.assertIn("ONLY", p)


class ParseGradeTest(unittest.TestCase):
    def test_parses_bare_digit(self):
        self.assertEqual(parse_grade("2"), 2)

    def test_parses_from_chatty_output(self):
        self.assertEqual(parse_grade("I would rate this a 3 because it directly answers."), 3)

    def test_parses_json_like(self):
        self.assertEqual(parse_grade('{"grade": 1}'), 1)

    def test_clamps_out_of_range_high(self):
        self.assertEqual(parse_grade("Grade: 7"), 3)

    def test_returns_zero_when_unparseable(self):
        self.assertEqual(parse_grade("no number here"), 0)

    def test_prefers_first_grade_token(self):
        # "grade" keyword wins over an incidental later number
        self.assertEqual(parse_grade("grade 2 (out of 3)"), 2)

    def test_clamps_high_digit_via_fallback(self):
        self.assertEqual(parse_grade("9"), 3)

    def test_parses_none_input(self):
        self.assertEqual(parse_grade(None), 0)


if __name__ == "__main__":
    unittest.main()
