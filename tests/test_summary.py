"""Tests for build_thread_aware_prompt and parse_response (stdlib-only, no network)."""
import unittest

from src.llm.summary import build_thread_aware_prompt, parse_response


class TestBuildThreadAwarePrompt(unittest.TestCase):
    def _email(self, body="reply body", **kw):
        e = {"sender": "x@example.com", "date": "2026-01-02", "subject": "Re: plan", "body": body}
        e.update(kw)
        return e

    def test_includes_target_body(self):
        p = build_thread_aware_prompt(self._email(body="TARGETBODY"), [])
        self.assertIn("TARGETBODY", p)

    def test_includes_preceding_context(self):
        pre = [self._email(body="ORIGINALREQUEST", subject="plan")]
        p = build_thread_aware_prompt(self._email(body="sounds good"), pre)
        self.assertIn("ORIGINALREQUEST", p)

    def test_emits_same_json_schema(self):
        p = build_thread_aware_prompt(self._email(), [])
        self.assertIn("is_noise", p)
        self.assertIn("summary", p)

    def test_asks_retrieval_oriented_summary(self):
        p = build_thread_aware_prompt(self._email(), []).lower()
        self.assertTrue("entit" in p or "decision" in p or "date" in p)

    def test_empty_preceding_is_valid(self):
        p = build_thread_aware_prompt(self._email(body="B"), [])
        self.assertIn("B", p)

    def test_preceding_is_capped(self):
        # only the last max_preceding messages are included (bounds densification/tokens)
        pre = [self._email(body=f"MSG{i}") for i in range(10)]
        p = build_thread_aware_prompt(self._email(), pre, max_preceding=2)
        self.assertIn("MSG9", p)
        self.assertIn("MSG8", p)
        self.assertNotIn("MSG0", p)

    def test_max_preceding_zero_suppresses_all_context(self):
        pre = [self._email(body="SHOULD_NOT_APPEAR")]
        p = build_thread_aware_prompt(self._email(), pre, max_preceding=0)
        self.assertNotIn("SHOULD_NOT_APPEAR", p)
        self.assertNotIn("EARLIER messages in this thread (context only", p)


class TestParseResponseLenient(unittest.TestCase):
    NOKIA = ('{"is_noise": false, "confidence": 1.0, "summary": "Darrell Jordan-Smith '
             'provides a brief "+1" endorsement of the news regarding the signed Nokia '
             'CRAN R&D collaboration agreement.", "reason": "The email is a genuine human '
             'response expressing agreement/support in an ongoing business thread."}')

    def test_recovers_unescaped_inner_quotes(self):
        r = parse_response(self.NOKIA)  # must NOT raise
        self.assertFalse(r["is_noise"])
        self.assertEqual(r["confidence"], 1.0)
        self.assertIn("+1", r["summary"])
        self.assertIn("Darrell Jordan-Smith", r["summary"])
        self.assertIn("Nokia CRAN", r["summary"])
        self.assertIn("genuine human response", r["reason"])

    def test_strict_path_unchanged_for_valid_json(self):
        good = '{"is_noise": false, "confidence": 0.9, "summary": "hello world", "reason": "ok"}'
        r = parse_response(good)
        self.assertEqual(r["summary"], "hello world")
        self.assertEqual(r["reason"], "ok")

    def test_still_raises_when_no_fields_recoverable(self):
        with self.assertRaises(ValueError):
            parse_response("total garbage, no json, no fields here")

    def test_is_noise_true_blanks_summary_in_fallback(self):
        # consistency with the strict path: is_noise=true => summary forced empty
        s = '{"is_noise": true, "confidence": 0.8, "summary": "should be "blanked" anyway", "reason": "newsletter"}'
        r = parse_response(s)
        self.assertTrue(r["is_noise"])
        self.assertEqual(r["summary"], "")


if __name__ == "__main__":
    unittest.main()
