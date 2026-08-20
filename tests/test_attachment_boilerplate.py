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
    BOILERPLATE_LARGE_MIN_THREADS,
    BOILERPLATE_MAX_SIZE,
    BOILERPLATE_SMALL_MAX_SIZE,
    BOILERPLATE_SMALL_MIN_THREADS,
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
        """A logo reused across many THREADS, plus a one-off screenshot."""
        logo = b"L" * 6000
        for i in range(BOILERPLATE_SMALL_MIN_THREADS + 2):
            self._put(logo, msg=f"m{i}", thread=f"th{i}")
        # Same name, same mime, same inline flag — only recurrence and size differ.
        self._put(b"S" * (BOILERPLATE_MAX_SIZE * 3), msg="m0", thread="th0", name="image001.png")
        self._put(
            b"%PDF-real",
            msg="m0",
            thread="th0",
            name="deck.pptx",
            mime="application/pdf",
            inline=False,
        )

    def test_recurring_small_inline_image_is_dropped(self):
        self._seed()
        sizes = [m.size for m in self.store.list_for(thread_id="th0", include_boilerplate=False)]
        self.assertNotIn(6000, sizes, "the reused 6KB logo should be filtered out")

    def test_one_off_large_inline_image_is_kept(self):
        # The false positive that matters: a pasted screenshot carries the same
        # auto-generated name as the logo and is still inline.
        self._seed()
        sizes = [m.size for m in self.store.list_for(thread_id="th0", include_boilerplate=False)]
        self.assertIn(BOILERPLATE_MAX_SIZE * 3, sizes, "a one-off screenshot is content")

    def test_real_document_always_survives(self):
        self._seed()
        names = [
            m.filename for m in self.store.list_for(thread_id="th0", include_boilerplate=False)
        ]
        self.assertIn("deck.pptx", names)

    def test_rarely_reused_small_image_is_kept(self):
        # Below the recurrence threshold: not enough evidence that it is decoration.
        rare = b"R" * 5000
        for i in range(BOILERPLATE_SMALL_MIN_THREADS - 1):
            self._put(rare, msg=f"r{i}", thread=f"rare{i}")
        kept = self.store.list_for(thread_id="rare0", include_boilerplate=False)
        self.assertEqual(len(kept), 1)

    def test_image_quoted_down_one_thread_is_kept(self):
        """The regression that motivated counting threads, not messages.

        A real image attached once and quoted down a long reply chain appears in
        every message of that ONE thread. Counting messages read that as
        decoration and hid it — on the live corpus that was 36% of everything
        the rule removed, including a feature-request table and a product
        lifecycle table (both verified by eye).
        """
        quoted = b"Q" * 5000
        for i in range(18):
            self._put(quoted, msg=f"q{i}", thread="one-long-thread")
        kept = self.store.list_for(thread_id="one-long-thread", include_boilerplate=False)
        self.assertEqual(len(kept), 18, "quoting is not evidence of decoration")

    def test_large_image_needs_far_wider_reuse_to_count_as_decoration(self):
        """Size scales the bar: a useful table shared into a few threads is content.

        Verified on the live corpus — a benchmark table (88.8KB) appeared in 5
        threads and the size-blind rule hid it, while the decoration it removed
        at that recurrence was uniformly under 20KB.
        """
        table = b"T" * (BOILERPLATE_SMALL_MAX_SIZE * 4)  # 80KB, over the small tier
        for i in range(BOILERPLATE_SMALL_MIN_THREADS + 2):
            self._put(table, msg=f"tb{i}", thread=f"tb{i}")
        self.assertEqual(len(self.store.list_for(thread_id="tb0", include_boilerplate=False)), 1)

        banner = b"B" * (BOILERPLATE_SMALL_MAX_SIZE * 4)
        for i in range(BOILERPLATE_LARGE_MIN_THREADS + 1):
            self._put(banner, msg=f"bn{i}", thread=f"bn{i}")
        self.assertEqual(len(self.store.list_for(thread_id="bn0", include_boilerplate=False)), 0)

    def test_include_boilerplate_returns_the_raw_list(self):
        self._seed()
        raw = self.store.list_for(thread_id="th0", include_boilerplate=True)
        filtered = self.store.list_for(thread_id="th0", include_boilerplate=False)
        self.assertGreater(len(raw), len(filtered))

    def test_default_is_unfiltered_at_the_store_layer(self):
        # The store stays a faithful record; only the MCP tool opts into filtering.
        self._seed()
        self.assertEqual(
            len(self.store.list_for(thread_id="th0")),
            len(self.store.list_for(thread_id="th0", include_boilerplate=True)),
        )

    def test_message_scoped_listing_filters_too(self):
        self._seed()
        rows = self.store.list_for(message_id="m0", include_boilerplate=False)
        self.assertNotIn(6000, [r.size for r in rows])
        self.assertIn("deck.pptx", [r.filename for r in rows])


if __name__ == "__main__":
    unittest.main()


class TestThreadCounts(unittest.TestCase):
    """Batched blob counts must agree with the per-blob answer.

    (An earlier revision chunked this query to dodge SQLite's bound-parameter
    cap. Dropped: that cap has defaulted to 32,766 since SQLite 3.32 in 2020 and
    pyproject requires Python >=3.11 from 2022, so no supported interpreter can
    reach the historical 999 limit. Defending against it was complexity for a
    state the project's own constraints exclude.)
    """

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="counts_")
        self.store = AttachmentStore(self.dir)

    def tearDown(self):
        self.store.close()
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_subset_and_full_queries_agree(self):
        for i in range(5):
            self.store.put(
                b"same-blob",
                message_id=f"m{i}",
                thread_id=f"t{i}",
                filename="logo.png",
                mime="image/png",
                size=9,
                source_type="eml",
                source_ref="r",
                inline=True,
            )
        full = self.store.thread_counts()
        subset = self.store.thread_counts(sha256s=list(full))
        self.assertEqual(full, subset)
        self.assertEqual(list(full.values()), [5])

    def test_empty_request_short_circuits(self):
        self.assertEqual(self.store.thread_counts(sha256s=[]), {})
