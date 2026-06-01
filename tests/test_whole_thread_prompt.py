"""Tests for build_whole_thread_prompt (stdlib-only, no network).

The whole-thread prompt is the bidirectional control for the row-3 novelty claim:
unlike build_thread_aware_prompt (PRECEDING-only / causal), it conditions the
per-email summary on ALL other messages in the thread (before AND after the target),
mirroring Anthropic-style whole-document contextual retrieval.
"""
import unittest

from src.llm.summary import build_whole_thread_prompt, parse_response


class TestBuildWholeThreadPrompt(unittest.TestCase):
    def _email(self, body="reply body", **kw):
        e = {"sender": "x@example.com", "date": "2026-01-02", "subject": "Re: plan", "body": body}
        e.update(kw)
        return e

    def test_includes_target_body(self):
        p = build_whole_thread_prompt(self._email(body="TARGETBODY"), [])
        self.assertIn("TARGETBODY", p)

    def test_includes_context_from_before_and_after(self):
        # the defining property: BIDIRECTIONAL context (both earlier and later msgs)
        others = [self._email(body="EARLIERMSG"), self._email(body="LATERMSG")]
        p = build_whole_thread_prompt(self._email(body="target"), others)
        self.assertIn("EARLIERMSG", p)
        self.assertIn("LATERMSG", p)

    def test_emits_same_json_schema(self):
        p = build_whole_thread_prompt(self._email(), [])
        self.assertIn("is_noise", p)
        self.assertIn("summary", p)

    def test_asks_full_thread_resolution(self):
        # the instruction must invoke FULL/WHOLE thread context, not just earlier msgs
        p = build_whole_thread_prompt(self._email(), []).lower()
        self.assertTrue("full" in p or "whole" in p)

    def test_empty_others_is_valid(self):
        p = build_whole_thread_prompt(self._email(body="B"), [])
        self.assertIn("B", p)

    def test_others_capped(self):
        others = [self._email(body=f"MSG{i}") for i in range(40)]
        p = build_whole_thread_prompt(self._email(), others, max_others=2)
        # keeps the last max_others (tail-biased when over the cap)
        self.assertIn("MSG39", p)
        self.assertIn("MSG38", p)
        self.assertNotIn("MSG0", p)

    def test_max_others_zero_suppresses_context(self):
        others = [self._email(body="SHOULD_NOT_APPEAR")]
        p = build_whole_thread_prompt(self._email(), others, max_others=0)
        self.assertNotIn("SHOULD_NOT_APPEAR", p)

    def test_response_parses_with_shared_schema(self):
        # sanity: the schema this prompt asks for round-trips through parse_response
        rec = parse_response('{"is_noise": false, "confidence": 0.9, '
                             '"summary": "s", "reason": "r"}')
        self.assertFalse(rec["is_noise"])
        self.assertEqual(rec["summary"], "s")


if __name__ == "__main__":
    unittest.main()
