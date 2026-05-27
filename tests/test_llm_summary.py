"""Tests for the LLM Pass-2 prompt building and response parsing (stdlib-only)."""
import unittest

from src.llm import summary


class TestBuildPrompt(unittest.TestCase):
    def test_includes_fields_and_truncates_body(self):
        email = {"sender": "a@x.com", "subject": "Hi", "date": "2024-01-01",
                 "body": "X" * 5000}
        prompt = summary.build_prompt(email, body_chars=100)
        self.assertIn("a@x.com", prompt)
        self.assertIn("Hi", prompt)
        self.assertIn("2024-01-01", prompt)
        self.assertIn("truncated", prompt)
        self.assertNotIn("X" * 200, prompt)

    def test_missing_fields_do_not_crash(self):
        prompt = summary.build_prompt({})
        self.assertIn("unknown", prompt)


class TestParseResponse(unittest.TestCase):
    def test_parses_plain_json(self):
        rec = summary.parse_response(
            '{"is_noise": false, "confidence": 0.9, "summary": "S", "reason": "R"}')
        self.assertEqual(rec, {"is_noise": False, "confidence": 0.9,
                               "summary": "S", "reason": "R"})

    def test_strips_code_fences(self):
        rec = summary.parse_response(
            '```json\n{"is_noise": true, "confidence": 1, "summary": "x", "reason": "r"}\n```')
        self.assertTrue(rec["is_noise"])

    def test_noise_clears_summary(self):
        rec = summary.parse_response(
            '{"is_noise": true, "confidence": 0.8, "summary": "should drop", "reason": "ad"}')
        self.assertEqual(rec["summary"], "")

    def test_confidence_clamped_and_defaulted(self):
        rec = summary.parse_response('{"is_noise": false, "summary": "s", "reason": "r"}')
        self.assertEqual(rec["confidence"], 0.0)
        rec2 = summary.parse_response(
            '{"is_noise": false, "confidence": 5, "summary": "s", "reason": "r"}')
        self.assertEqual(rec2["confidence"], 1.0)

    def test_missing_is_noise_raises(self):
        with self.assertRaises(ValueError):
            summary.parse_response('{"summary": "s"}')

    def test_invalid_json_raises(self):
        with self.assertRaises(ValueError):
            summary.parse_response("not json")

    def test_extracts_json_from_prose_prefix(self):
        rec = summary.parse_response(
            'Here is the JSON:\n{"is_noise": false, "confidence": 0.5, "summary": "s", "reason": "r"}')
        self.assertFalse(rec["is_noise"])
        self.assertEqual(rec["summary"], "s")

    def test_extracts_json_with_trailing_text(self):
        rec = summary.parse_response(
            '{"is_noise": true, "confidence": 0.7, "summary": "x", "reason": "ad"}\nHope that helps!')
        self.assertTrue(rec["is_noise"])

    def test_extracts_fenced_json_inside_prose(self):
        rec = summary.parse_response(
            'Sure:\n```json\n{"is_noise": false, "confidence": 0.9, "summary": "ok", "reason": "r"}\n```\nDone.')
        self.assertEqual(rec["confidence"], 0.9)

    def test_no_json_object_still_raises(self):
        with self.assertRaises(ValueError):
            summary.parse_response("there is no json here at all")


if __name__ == "__main__":
    unittest.main()
