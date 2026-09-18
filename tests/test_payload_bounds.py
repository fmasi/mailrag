"""Whole-text returns are bounded, and any cut is announced (#138).

`search_email` snippets and `grep_email` were already bounded, so the claim that
"a single hostile message cannot flood the context window" held for them — but
`get_thread` / `search_email(full=True)` and `get_attachment` returned their text
entire. A 20 MB PDF's OCR came back whole.

Truncation is always announced. Silently returning less than was asked for is
how a caller concludes a document does not mention something that it does.
"""

import unittest

from src.mcp_server.server import (
    HARD_ATTACHMENT_TEXT_CHARS,
    HARD_FULL_TEXT_CHARS,
    _bounded_text,
    _thread_to_full_dict,
    get_attachment,
)


class _Ctx:
    def __init__(self, text):
        self.thread_id, self.subject, self.text, self.emails = "t1", "s", text, []


class _Store:
    root = "/tmp/a"

    def __init__(self, text):
        self._text = text

    def count(self):
        return 1

    def fetch(self, sha256, extractor=None):
        return {
            "sha256": "s",
            "filename": "big.pdf",
            "mime": "application/pdf",
            "size": 20_000_000,
            "text": self._text,
            "text_status": "ok",
        }

    def close(self):
        pass


class TestBoundedText(unittest.TestCase):
    def test_short_text_is_untouched_and_unflagged(self):
        text, fields = _bounded_text("hello", 100)
        self.assertEqual(text, "hello")
        self.assertEqual(fields, {})

    def test_long_text_is_cut_and_the_cut_is_announced(self):
        text, fields = _bounded_text("x" * 500, 100)
        self.assertEqual(len(text), 100)
        self.assertTrue(fields["truncated"])
        self.assertEqual(fields["full_length"], 500)

    def test_exactly_at_the_cap_is_not_truncated(self):
        _, fields = _bounded_text("x" * 100, 100)
        self.assertEqual(fields, {})

    def test_none_is_handled(self):
        self.assertEqual(_bounded_text(None, 10), ("", {}))


class TestFullThreadBound(unittest.TestCase):
    def test_a_normal_thread_is_returned_whole_and_unflagged(self):
        row = _thread_to_full_dict(_Ctx("a short thread"))
        self.assertEqual(row["text"], "a short thread")
        self.assertNotIn("truncated", row)

    def test_a_pathological_thread_is_capped_with_a_marker(self):
        row = _thread_to_full_dict(_Ctx("x" * (HARD_FULL_TEXT_CHARS + 5000)))
        self.assertEqual(len(row["text"]), HARD_FULL_TEXT_CHARS)
        self.assertTrue(row["truncated"])
        self.assertEqual(row["full_length"], HARD_FULL_TEXT_CHARS + 5000)


class TestAttachmentTextBound(unittest.TestCase):
    def test_a_huge_extraction_is_capped_and_says_so(self):
        row = get_attachment("s", store=_Store("y" * (HARD_ATTACHMENT_TEXT_CHARS + 1)))
        self.assertEqual(len(row["text"]), HARD_ATTACHMENT_TEXT_CHARS)
        self.assertTrue(row["truncated"])

    def test_coverage_still_describes_the_extraction_not_the_transport(self):
        """A document cut for transport has not become text-sparse.

        `text_coverage` answers "did extraction get the content", so it must be
        computed on the full text — otherwise capping a rich 20 MB document
        would relabel it `sparse` and send a caller to render pages needlessly.
        """
        row = get_attachment("s", store=_Store("z" * (HARD_ATTACHMENT_TEXT_CHARS + 1)))
        self.assertEqual(row["text_coverage"], "rich")
        self.assertGreater(row["chars"], HARD_ATTACHMENT_TEXT_CHARS)

    def test_an_ordinary_attachment_is_unflagged(self):
        row = get_attachment("s", store=_Store("small document text"))
        self.assertNotIn("truncated", row)


if __name__ == "__main__":
    unittest.main()
