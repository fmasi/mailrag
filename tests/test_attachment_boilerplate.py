"""Tests for boilerplate filtering in the attachment store.

Calibrated against a real 45k-row corpus where 73% of attachment rows were
recurring inline images. The rule under test is deliberately NOT "drop inline"
or "drop images": the same auto-generated name and mime cover both a 259-byte
spacer pixel and a 12 MB pasted screenshot, so only recurrence separates
decoration from content. These tests pin that distinction — the filter must
remove the reused logo and keep the one-off screenshot.
"""

import shutil
import tempfile
import unittest

from src.attachments.store import (
    BOILERPLATE_MAX_SIZE,
    BOILERPLATE_MIN_MESSAGES,
    AttachmentStore,
)


class TestBoilerplateFilter(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="att_store_")
        self.store = AttachmentStore(self.dir)

    def tearDown(self):
        self.store.close()
        shutil.rmtree(self.dir, ignore_errors=True)

    def _put(self, data, *, msg, thread="t1", name="image001.png", mime="image/png", inline=True):
        return self.store.put(
            data,
            message_id=msg,
            thread_id=thread,
            filename=name,
            mime=mime,
            size=len(data),
            source_type="eml",
            source_ref=f"/corpus/{msg}.eml",
            inline=inline,
        )

    def _seed(self):
        """A logo reused across many messages, plus a one-off screenshot."""
        logo = b"L" * 6000
        for i in range(BOILERPLATE_MIN_MESSAGES + 2):
            self._put(logo, msg=f"m{i}")
        # Same name, same mime, same inline flag — only recurrence and size differ.
        self._put(b"S" * (BOILERPLATE_MAX_SIZE * 3), msg="m0", name="image001.png")
        self._put(b"%PDF-real", msg="m0", name="deck.pptx", mime="application/pdf", inline=False)

    def test_recurring_small_inline_image_is_dropped(self):
        self._seed()
        sizes = [m.size for m in self.store.list_for(thread_id="t1", include_boilerplate=False)]
        self.assertNotIn(6000, sizes, "the reused 6KB logo should be filtered out")

    def test_one_off_large_inline_image_is_kept(self):
        # The false positive that matters: a pasted screenshot carries the same
        # auto-generated name as the logo and is still inline.
        self._seed()
        sizes = [m.size for m in self.store.list_for(thread_id="t1", include_boilerplate=False)]
        self.assertIn(BOILERPLATE_MAX_SIZE * 3, sizes, "a one-off screenshot is content")

    def test_real_document_always_survives(self):
        self._seed()
        names = [m.filename for m in self.store.list_for(thread_id="t1", include_boilerplate=False)]
        self.assertIn("deck.pptx", names)

    def test_rarely_reused_small_image_is_kept(self):
        # Below the recurrence threshold: not enough evidence that it is decoration.
        rare = b"R" * 5000
        for i in range(BOILERPLATE_MIN_MESSAGES - 1):
            self._put(rare, msg=f"r{i}", thread="t2")
        kept = self.store.list_for(thread_id="t2", include_boilerplate=False)
        self.assertEqual(len(kept), BOILERPLATE_MIN_MESSAGES - 1)

    def test_include_boilerplate_returns_the_raw_list(self):
        self._seed()
        raw = self.store.list_for(thread_id="t1", include_boilerplate=True)
        filtered = self.store.list_for(thread_id="t1", include_boilerplate=False)
        self.assertGreater(len(raw), len(filtered))

    def test_default_is_unfiltered_at_the_store_layer(self):
        # The store stays a faithful record; only the MCP tool opts into filtering.
        self._seed()
        self.assertEqual(
            len(self.store.list_for(thread_id="t1")),
            len(self.store.list_for(thread_id="t1", include_boilerplate=True)),
        )

    def test_message_scoped_listing_filters_too(self):
        self._seed()
        rows = self.store.list_for(message_id="m0", include_boilerplate=False)
        self.assertNotIn(6000, [r.size for r in rows])
        self.assertIn("deck.pptx", [r.filename for r in rows])


if __name__ == "__main__":
    unittest.main()
