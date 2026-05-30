"""Tests for build_thread_aware_prompt (stdlib-only, no network)."""
import unittest

from src.llm.summary import build_thread_aware_prompt


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


if __name__ == "__main__":
    unittest.main()
