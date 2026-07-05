"""Unit tests for src.llm.thread_summaries.generate_thread_summaries.

LLM calls are fully mocked — no network required.
"""

import unittest
from unittest import mock

from src.data.models import NormalizedEmail
from src.llm.thread_summaries import generate_thread_summaries


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
            out = generate_thread_summaries(thread, model="m")

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
            out = generate_thread_summaries(emails, model="m")

        self.assertEqual(out["<3>"], "")

    def test_llm_error_does_not_raise(self):
        """An LLM exception must be swallowed; the email maps to ''."""
        emails = [_e("Hello world", "<4>")]

        def fake_chat(client, model, prompt):
            raise RuntimeError("network error")

        with (
            mock.patch("src.llm.thread_summaries.chat", side_effect=fake_chat),
            mock.patch("src.llm.thread_summaries.make_client", return_value=object()),
        ):
            out = generate_thread_summaries(emails, model="m")

        self.assertIn("<4>", out)
        self.assertEqual(out["<4>"], "")

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
            out = generate_thread_summaries([e1, e2], model="m")

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
            out = generate_thread_summaries(emails, model="m", client=sentinel)

        mk.assert_not_called()
        self.assertIs(seen_clients[0], sentinel)
        self.assertEqual(out["<7>"], "Z")


if __name__ == "__main__":
    unittest.main()
