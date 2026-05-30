"""Tests for the eval-query validator gate (issue #18)."""
import unittest

from src.eval.query_validator import build_validation_prompt, parse_validation


class TestParseValidation(unittest.TestCase):
    def test_keep_true(self):
        self.assertEqual(
            parse_validation('{"keep": true, "reason": "specific fact"}'),
            {"keep": True, "reason": "specific fact"})

    def test_keep_false_with_reason(self):
        self.assertEqual(
            parse_validation('{"keep": false, "reason": "meta question"}'),
            {"keep": False, "reason": "meta question"})

    def test_fenced_json(self):
        raw = "```json\n{\"keep\": true, \"reason\": \"ok\"}\n```"
        self.assertEqual(parse_validation(raw), {"keep": True, "reason": "ok"})

    def test_prose_around_json(self):
        raw = 'Sure! {"keep": false, "reason": "no fact"} hope that helps'
        self.assertEqual(parse_validation(raw), {"keep": False, "reason": "no fact"})

    def test_empty_is_fail_closed(self):
        self.assertEqual(parse_validation(""), {"keep": False, "reason": "empty output"})

    def test_none_is_fail_closed(self):
        self.assertEqual(parse_validation(None), {"keep": False, "reason": "empty output"})

    def test_no_json_object_is_fail_closed(self):
        self.assertEqual(
            parse_validation("yes keep it"), {"keep": False, "reason": "no json object"})

    def test_unparseable_json_is_fail_closed(self):
        # looks like an object but isn't valid JSON
        self.assertEqual(
            parse_validation("{keep: true}"), {"keep": False, "reason": "unparseable json"})

    def test_non_bool_keep_is_fail_closed(self):
        self.assertEqual(
            parse_validation('{"keep": "yes"}'),
            {"keep": False, "reason": "missing/invalid keep"})

    def test_missing_reason_defaults_empty(self):
        self.assertEqual(
            parse_validation('{"keep": true}'), {"keep": True, "reason": ""})


class TestBuildValidationPrompt(unittest.TestCase):
    def test_embeds_all_three_inputs(self):
        p = build_validation_prompt("the QUERY", "the THREAD", "the ANSWER")
        self.assertIn("the QUERY", p)
        self.assertIn("the THREAD", p)
        self.assertIn("the ANSWER", p)

    def test_demands_content_fact_and_forbids_artifact(self):
        p = build_validation_prompt("q", "t", "a").lower()
        self.assertIn("fact", p)
        self.assertIn("artifact", p)

    def test_requests_strict_json(self):
        self.assertIn("JSON", build_validation_prompt("q", "t", "a"))


if __name__ == "__main__":
    unittest.main()
