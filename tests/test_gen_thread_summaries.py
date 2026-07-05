"""Tests for gen_thread_summaries helpers (pure, no LM Studio / Qdrant needed)."""

import importlib
import json
import os
import tempfile
import unittest

# ---------------------------------------------------------------------------
# Import strategy: gen_thread_summaries has top-level imports from src.*
# which should be present in the mailrag env.  We import with
# importlib so that an ImportError here is explicit and the test can be
# skipped cleanly rather than crashing the suite.
# ---------------------------------------------------------------------------
try:
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    _mod = importlib.import_module("scripts.eval.gen_thread_summaries")
    _record_failure = _mod._record_failure
    _build_prompt_for = _mod._build_prompt_for
    _IMPORT_OK = True
except Exception as _import_err:  # noqa: BLE001
    _IMPORT_OK = False
    _import_err_msg = str(_import_err)


@unittest.skipUnless(
    _IMPORT_OK, f"gen_thread_summaries not importable: {'' if _IMPORT_OK else _import_err_msg}"
)  # type: ignore[name-defined]
class TestRecordFailure(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self._tmp.name, "failures.jsonl")

    def tearDown(self):
        self._tmp.cleanup()

    def test_appends_valid_json_line(self):
        _record_failure(self.path, source_id="foo.eml", sha="abc", error="boom", raw_response=None)
        with open(self.path, encoding="utf-8") as fh:
            lines = fh.readlines()
        self.assertEqual(len(lines), 1)
        rec = json.loads(lines[0])
        self.assertEqual(rec["source_id"], "foo.eml")
        self.assertEqual(rec["sha"], "abc")
        self.assertEqual(rec["error"], "boom")
        self.assertIsNone(rec["raw_response"])

    def test_two_calls_append_two_lines(self):
        _record_failure(self.path, source_id="a.eml", error="e1")
        _record_failure(self.path, source_id="b.eml", error="e2")
        with open(self.path, encoding="utf-8") as fh:
            lines = [l for l in fh.readlines() if l.strip()]
        self.assertEqual(len(lines), 2)
        self.assertEqual(json.loads(lines[0])["source_id"], "a.eml")
        self.assertEqual(json.loads(lines[1])["source_id"], "b.eml")

    def test_raw_response_string_preserved(self):
        _record_failure(self.path, source_id="c.eml", raw_response='{"bad json":}')
        rec = json.loads(open(self.path).read().strip())
        self.assertEqual(rec["raw_response"], '{"bad json":}')

    def test_does_not_raise_on_unwritable_path(self):
        bad_path = "/dev/null/nonexistent/failures.jsonl"
        # Must not raise — best-effort silencing
        try:
            _record_failure(bad_path, source_id="x.eml", error="irrelevant")
        except Exception as exc:  # noqa: BLE001
            self.fail(f"_record_failure raised on unwritable path: {exc}")


@unittest.skipUnless(
    _IMPORT_OK, f"gen_thread_summaries not importable: {'' if _IMPORT_OK else _import_err_msg}"
)  # type: ignore[name-defined]
class TestBuildPromptDispatch(unittest.TestCase):
    """The mode->prompt dispatch is the only branching logic in the network loop."""

    def _e(self, body):
        return {"sender": "s@x.com", "date": "2026-01-01", "subject": "Re: x", "body": body}

    def test_thread_mode_sees_preceding_not_following(self):
        ed = self._e("TARGET")
        preceding = [self._e("BEFOREMSG")]
        others = [self._e("BEFOREMSG"), self._e("AFTERMSG")]
        p = _build_prompt_for("thread", ed, preceding, others)
        self.assertIn("BEFOREMSG", p)
        self.assertNotIn("AFTERMSG", p)  # causal / append-only

    def test_whole_mode_sees_following(self):
        ed = self._e("TARGET")
        others = [self._e("BEFOREMSG"), self._e("AFTERMSG")]
        p = _build_prompt_for("whole", ed, [], others)
        self.assertIn("BEFOREMSG", p)
        self.assertIn("AFTERMSG", p)  # bidirectional

    def test_isolated_mode_sees_no_context(self):
        ed = self._e("TARGET")
        others = [self._e("BEFOREMSG"), self._e("AFTERMSG")]
        p = _build_prompt_for("isolated", ed, [self._e("BEFOREMSG")], others)
        self.assertIn("TARGET", p)
        self.assertNotIn("BEFOREMSG", p)
        self.assertNotIn("AFTERMSG", p)


if __name__ == "__main__":
    unittest.main()
