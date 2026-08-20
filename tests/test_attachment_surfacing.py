"""Tests for surfacing attachments to an agent: names on hits, sparse-text signal.

Both come from a field report (2026-08-20) in which an agent ran four retrieval
passes over a corpus holding a 22 MB deck and never once opened the attachment
tools. The cause was not that the tools were broken — they worked — but that
nothing in the search surface said the documents existed, and one field said the
opposite.
"""

import unittest

from src.mcp_server import server


class _Ctx:
    def __init__(self, thread_id="t1", emails=()):
        self.thread_id = thread_id
        self.subject = "s"
        self.text = "body"
        self.emails = list(emails)


class _Email:
    def __init__(self, message_id):
        self.message_id = message_id
        self.date = "Mon, 1 Jan 2025"
        self.sender = "a@x.com"
        self.to = "b@y.com"


class _Store:
    def __init__(self, count=5, names=("deck.pptx",)):
        self._count = count
        self._names = list(names)
        self.calls = []

    def count(self):
        return self._count

    def names_for(self, *, thread_id=None, message_ids=None):
        self.calls.append((thread_id, tuple(message_ids or ())))
        return self._names


class TestAttachmentNamesOnHits(unittest.TestCase):
    def test_names_are_surfaced_when_the_store_can_answer(self):
        meta = server._thread_meta(_Ctx(emails=[_Email("<m1>")]), store=_Store())
        self.assertEqual(meta["attachment_names"], ["deck.pptx"])

    def test_field_is_absent_when_the_store_was_never_built(self):
        """The bug that taught an agent the corpus had no attachments.

        An always-empty list does not read as "unpopulated", it reads as a
        confident "none here" — repeated on every hit of every query. Absent is
        honest; empty is a claim.
        """
        meta = server._thread_meta(_Ctx(emails=[_Email("<m1>")]), store=_Store(count=0))
        self.assertNotIn("attachment_names", meta)

    def test_empty_list_still_means_genuinely_none(self):
        # With a populated store, [] is a real answer and must be reported.
        meta = server._thread_meta(_Ctx(emails=[_Email("<m1>")]), store=_Store(names=()))
        self.assertEqual(meta["attachment_names"], [])

    def test_lookup_uses_thread_id_and_every_message_id(self):
        # Attachments can be filed against a sibling message whose thread id
        # differs from the one retrieval returned, so both keys are needed.
        store = _Store()
        server._thread_meta(
            _Ctx(thread_id="t9", emails=[_Email("<m1>"), _Email("<m2>")]), store=store
        )
        thread_id, ids = store.calls[0]
        self.assertEqual(thread_id, "t9")
        self.assertEqual(ids, ("<m1>", "<m2>"))

    def test_a_broken_store_never_breaks_search(self):
        class Exploding:
            def count(self):
                raise RuntimeError("db gone")

        meta = server._thread_meta(_Ctx(emails=[_Email("<m1>")]), store=Exploding())
        self.assertNotIn("attachment_names", meta)
        self.assertIn("message_ids", meta)


class TestSparseTextSignal(unittest.TestCase):
    """`text_status: extracted` must not read as "you now have the content"."""

    def test_slide_deck_with_a_text_layer_of_nothing_is_flagged(self):
        # The real 21 MB deck: 1,393 chars, 66 chars/MB.
        cov = server._text_coverage("x" * 1393, 22_235_496, "application/vnd.ms-powerpoint")
        self.assertEqual(cov["text_coverage"], "sparse")
        self.assertIn("pictorial", cov["note"])

    def test_diagram_pdf_is_flagged(self):
        # The real partner PDF whose key fact is a horseshoe of five names.
        cov = server._text_coverage("x" * 1883, 1_056_439, "application/pdf")
        self.assertEqual(cov["text_coverage"], "sparse")

    def test_text_bearing_document_is_not_flagged(self):
        # Sampled text PDFs measured 99k-225k chars/MB; this sits inside that.
        cov = server._text_coverage("x" * 150_000, 1_048_576, "application/pdf")
        self.assertEqual(cov["text_coverage"], "rich")
        self.assertNotIn("note", cov)

    def test_small_files_are_not_judged(self):
        # The ratio is noise below the size floor: a 900-byte spacer scores zero
        # and a 2 KB note scores enormously, neither meaningfully.
        cov = server._text_coverage("hi", 900, "image/png")
        self.assertNotIn("text_coverage", cov)
        self.assertEqual(cov["chars"], 2)

    def test_plain_text_is_never_judged(self):
        cov = server._text_coverage("short", 500_000, "text/plain")
        self.assertNotIn("text_coverage", cov)

    def test_chars_are_always_reported(self):
        for size, mime in ((900, "image/png"), (2_000_000, "application/pdf")):
            with self.subTest(size=size):
                self.assertEqual(server._text_coverage("abc", size, mime)["chars"], 3)


if __name__ == "__main__":
    unittest.main()
