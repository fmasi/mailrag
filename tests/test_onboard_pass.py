import json
import os
import shutil
import tempfile
import unittest
from unittest import mock

from src.llm.cache import Pass2Cache
from src.llm.onboard_pass import generate_thread_judgments


class _Email:
    def __init__(self, mid, subject, body, date, thread_id=None):
        self.message_id = mid
        self.subject = subject
        self.body = body
        self.date = date
        self.thread_id = thread_id
        self.sender = "s@x.com"
        self.in_reply_to = ""
        self.references = ""


def _reply(is_noise, summary):
    return json.dumps({"is_noise": is_noise, "confidence": 0.9, "summary": summary, "reason": "r"})


class TestGenerateThreadJudgments(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.cache = Pass2Cache(os.path.join(self.dir, "p.db"))

    def tearDown(self):
        self.cache.close()
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_keeps_noise_judgment_and_summary(self):
        emails = [
            _Email("m1", "Plan", "let us meet", "2020-01-01", "t1"),
            _Email("m2", "Re: Plan", "ok", "2020-01-02", "t1"),
        ]
        with (
            mock.patch("src.llm.onboard_pass.default_model", return_value="M"),
            mock.patch("src.llm.onboard_pass.make_client", return_value="C"),
            mock.patch(
                "src.llm.onboard_pass.chat", side_effect=[_reply(False, "sum1"), _reply(True, "")]
            ),
        ):
            out = generate_thread_judgments(emails, cache=self.cache)
        self.assertEqual(out["m1"]["summary"], "sum1")
        self.assertFalse(out["m1"]["is_noise"])
        self.assertTrue(out["m2"]["is_noise"])

    def test_resumes_from_cache_without_calling_llm(self):
        emails = [_Email("m1", "Plan", "body", "2020-01-01", "t1")]
        with (
            mock.patch("src.llm.onboard_pass.default_model", return_value="M"),
            mock.patch("src.llm.onboard_pass.make_client", return_value="C"),
            mock.patch("src.llm.onboard_pass.chat", return_value=_reply(False, "first")) as chat1,
        ):
            generate_thread_judgments(emails, cache=self.cache)
            self.assertEqual(chat1.call_count, 1)
        with mock.patch("src.llm.onboard_pass.chat") as chat2:  # must NOT be called
            out = generate_thread_judgments(emails, cache=self.cache)
            chat2.assert_not_called()
        self.assertEqual(out["m1"]["summary"], "first")

    def test_llm_failure_is_conservative_keep(self):
        emails = [_Email("m1", "Plan", "body", "2020-01-01", "t1")]
        with (
            mock.patch("src.llm.onboard_pass.default_model", return_value="M"),
            mock.patch("src.llm.onboard_pass.make_client", return_value="C"),
            mock.patch("src.llm.onboard_pass.chat", side_effect=RuntimeError("boom")),
        ):
            out = generate_thread_judgments(emails, cache=self.cache)
        self.assertFalse(out["m1"]["is_noise"])
        self.assertEqual(out["m1"]["confidence"], 0.0)
        self.assertEqual(out["m1"]["summary"], "")
        self.assertTrue(out["m1"]["reason"].startswith("llm_error"))

    def test_failure_is_not_cached_and_retried(self):
        emails = [_Email("m1", "Plan", "body", "2020-01-01", "t1")]
        # First run: chat raises -> conservative keep, NOT persisted.
        with (
            mock.patch("src.llm.onboard_pass.default_model", return_value="M"),
            mock.patch("src.llm.onboard_pass.make_client", return_value="C"),
            mock.patch("src.llm.onboard_pass.chat", side_effect=RuntimeError("boom")),
        ):
            generate_thread_judgments(emails, cache=self.cache)
        # Second run (same cache): the email must be retried -> chat IS called again,
        # and this time it succeeds and is recorded.
        with (
            mock.patch("src.llm.onboard_pass.default_model", return_value="M"),
            mock.patch("src.llm.onboard_pass.make_client", return_value="C"),
            mock.patch(
                "src.llm.onboard_pass.chat", return_value=_reply(False, "recovered")
            ) as chat2,
        ):
            out = generate_thread_judgments(emails, cache=self.cache)
            self.assertEqual(chat2.call_count, 1)  # retried, not served from cache
        self.assertEqual(out["m1"]["summary"], "recovered")


if __name__ == "__main__":
    unittest.main()
