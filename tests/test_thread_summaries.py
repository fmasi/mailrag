"""Unit tests for src.llm.thread_summaries.generate_thread_summaries.

LLM calls are fully mocked — no network required.
"""

import unittest
from unittest import mock

from src.data.models import NormalizedEmail
from src.llm.thread_summaries import SummaryGenerationError, generate_thread_summaries


def _e(body, mid, subject="Re: plan", date="2026-01-01", in_reply_to=""):
    """Build a minimal NormalizedEmail.

    Two emails will land in the same thread when the second carries
    ``in_reply_to=<first_message_id>`` — that is how compute_thread_id groups
    them (subject alone is NOT a threading key in this codebase).
    """
    return NormalizedEmail(
        sender="s@x.com",
        subject=subject,
        date=date,
        body=body,
        source="enron",
        source_id=mid,
        message_id=mid,
        in_reply_to=in_reply_to,
    )


class TestGenerateThreadSummaries(unittest.TestCase):
    def test_returns_summary_per_email_with_preceding_context(self):
        """Email 2's prompt must contain email 1's body as preceding context."""
        # Email 2 replies to email 1 → same thread.
        thread = [
            _e("Can we ship Friday?", "<1>", date="2026-01-01"),
            _e("Yes, 12:30 works", "<2>", date="2026-01-02", in_reply_to="<1>"),
        ]
        calls = []

        def fake_chat(client, model, prompt):
            calls.append(prompt)
            return '{"is_noise": false, "confidence": 0.9, "summary": "S", "reason": "r"}'

        with (
            mock.patch("src.llm.thread_summaries.chat", side_effect=fake_chat),
            mock.patch("src.llm.thread_summaries.make_client", return_value=object()),
        ):
            out = generate_thread_summaries(thread, model="m", preflight=False)

        # Both message_ids must be present in the result.
        self.assertEqual(set(out), {"<1>", "<2>"})
        # Non-noise emails get their summary text.
        self.assertEqual(out["<2>"], "S")
        # The second email's prompt includes the first email's body as preceding context.
        self.assertIn("Can we ship Friday?", calls[1])

    def test_noise_email_maps_to_empty_string(self):
        """Emails flagged as noise must produce an empty summary."""
        emails = [_e("Buy now! Great deals!", "<3>", subject="Newsletter")]

        def fake_chat(client, model, prompt):
            return '{"is_noise": true, "confidence": 0.99, "summary": "", "reason": "ad"}'

        with (
            mock.patch("src.llm.thread_summaries.chat", side_effect=fake_chat),
            mock.patch("src.llm.thread_summaries.make_client", return_value=object()),
        ):
            out = generate_thread_summaries(emails, model="m", preflight=False)

        self.assertEqual(out["<3>"], "")

    def test_a_failed_call_is_absent_rather_than_empty(self):
        """A failure must be distinguishable from a noise verdict.

        Both used to map to "" — so a dead endpoint produced a corpus in which
        every email looked like the model had judged it noise, with no error
        (#135). Present-and-empty now means noise; ABSENT means failed.
        """
        emails = [_e("body one", "<1>"), _e("body two", "<2>")]
        with mock.patch("src.llm.thread_summaries.make_client"):
            with mock.patch("src.llm.thread_summaries.chat", side_effect=RuntimeError("boom")):
                out = generate_thread_summaries(
                    emails, model="m", preflight=False, max_failure_rate=1.0
                )
        self.assertEqual(out, {}, "failures must not be reported as noise verdicts")

    def test_noise_and_failure_are_distinguishable_in_one_run(self):
        seen = {"n": 0}

        def flaky(_c, _m, _p):
            seen["n"] += 1
            if seen["n"] == 1:
                return '{"is_noise": true, "summary": ""}'
            raise RuntimeError("endpoint died")

        emails = [_e("body one", "<1>"), _e("body two", "<2>")]
        with mock.patch("src.llm.thread_summaries.make_client"):
            with mock.patch("src.llm.thread_summaries.chat", side_effect=flaky):
                out = generate_thread_summaries(
                    emails, model="m", preflight=False, max_failure_rate=1.0
                )
        self.assertEqual(out.get("<1>"), "", "noise verdict should be present and empty")
        self.assertNotIn("<2>", out, "failure should be absent, not empty")

    def test_a_total_outage_raises_instead_of_returning_a_blank_corpus(self):
        """1200 failures out of 1200 is a misconfiguration; continuing silently
        is worse than stopping."""
        emails = [_e(f"body {i}", f"<{i}>") for i in range(10)]
        with mock.patch("src.llm.thread_summaries.make_client"):
            with mock.patch("src.llm.thread_summaries.chat", side_effect=RuntimeError("down")):
                with self.assertRaises(SummaryGenerationError) as cm:
                    generate_thread_summaries(emails, model="m", preflight=False)
        msg = str(cm.exception)
        self.assertIn("10", msg)
        self.assertIn("down", msg, "the underlying error must reach the operator")

    def test_a_few_transient_failures_do_not_abort_the_run(self):
        calls = {"n": 0}

        def mostly_ok(_c, _m, _p):
            calls["n"] += 1
            if calls["n"] == 3:
                raise RuntimeError("blip")
            return '{"is_noise": false, "summary": "ok"}'

        emails = [_e(f"body {i}", f"<{i}>") for i in range(10)]
        with mock.patch("src.llm.thread_summaries.make_client"):
            with mock.patch("src.llm.thread_summaries.chat", side_effect=mostly_ok):
                out = generate_thread_summaries(emails, model="m", preflight=False)
        self.assertEqual(len(out), 9)
        self.assertNotIn("<2>", out)

    def test_preflight_failure_aborts_before_spending_the_corpus(self):
        """A misconfigured endpoint should cost one call, not one per email."""
        emails = [_e(f"body {i}", f"<{i}>") for i in range(50)]
        with mock.patch("src.llm.thread_summaries.make_client"):
            with mock.patch(
                "src.llm.thread_summaries.healthcheck", side_effect=RuntimeError("401")
            ) as hc:
                with mock.patch("src.llm.thread_summaries.chat") as chat_mock:
                    with self.assertRaises(SummaryGenerationError):
                        generate_thread_summaries(emails, model="m")
        hc.assert_called_once()
        chat_mock.assert_not_called()

    def test_independent_threads_are_separated(self):
        """Two emails with no reply-to relationship stay in different threads."""
        e1 = _e("Thread A root", "<5>", subject="Topic A")
        e2 = _e("Thread B root", "<6>", subject="Topic B")

        calls = []

        def fake_chat(client, model, prompt):
            calls.append(prompt)
            return '{"is_noise": false, "confidence": 0.8, "summary": "X", "reason": "ok"}'

        with (
            mock.patch("src.llm.thread_summaries.chat", side_effect=fake_chat),
            mock.patch("src.llm.thread_summaries.make_client", return_value=object()),
        ):
            out = generate_thread_summaries([e1, e2], model="m", preflight=False)

        self.assertIn("<5>", out)
        self.assertIn("<6>", out)
        # Each email is its own thread root, so no cross-contamination:
        # email 1's body must NOT appear in email 2's prompt and vice-versa.
        # (build_thread_aware_prompt always includes the header text even when
        # preceding=[] — only the body content distinguishes "has context" from not.)
        self.assertNotIn("Thread B root", calls[0])
        self.assertNotIn("Thread A root", calls[1])

    def test_pre_built_client_is_used(self):
        """Passing a client= kwarg must bypass make_client()."""
        sentinel = object()
        seen_clients = []

        def fake_chat(client, model, prompt):
            seen_clients.append(client)
            return '{"is_noise": false, "confidence": 0.9, "summary": "Z", "reason": "r"}'

        emails = [_e("Hello", "<7>")]
        with (
            mock.patch("src.llm.thread_summaries.chat", side_effect=fake_chat),
            mock.patch("src.llm.thread_summaries.make_client") as mk,
        ):
            out = generate_thread_summaries(emails, model="m", client=sentinel, preflight=False)

        mk.assert_not_called()
        self.assertIs(seen_clients[0], sentinel)
        self.assertEqual(out["<7>"], "Z")


if __name__ == "__main__":
    unittest.main()
