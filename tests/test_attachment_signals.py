"""Tests for measured noise signals and the bulk classify pass.

The design under test: MEASURE and record, judge at read time. Signals are
corpus-agnostic (a logo has no text, a table does); thresholds are not (work
mail is 41% bulk, personal 7%). Keeping them apart means re-calibrating against
a different corpus is a SQL query, not a re-OCR of thousands of blobs.

The labelled cases below are real images from the work corpus, inspected by eye
during calibration, with their measured character counts.
"""

import shutil
import tempfile
import unittest

from src.attachments.classify import classify_blobs
from src.attachments.signals import (
    UBIQUITOUS_THREADS,
    BlobSignals,
    is_decoration,
    measure_blob,
    measure_text,
)
from src.attachments.store import AttachmentStore


def _sig(chars, *, words=0, uwords=0, digits=0, status="extracted", w=0, h=0):
    return BlobSignals(chars, words, uwords, digits, w, h, status, "tesseract")


class _FakeExtractor:
    """Returns canned text per filename, so tests need no OCR engine.

    Reports itself as ``tesseract`` because verdicts are only trusted from
    calibrated engines — an unknown extractor name deliberately yields no
    opinion (covered by TestEngineIndependence).
    """

    def __init__(self, texts=None, raises=False, name="tesseract"):
        self._texts = texts or {}
        self._raises = raises
        self._name = name

    def extract(self, data, mime, filename):
        from src.attachments.extract.result import ExtractResult

        if self._raises:
            raise RuntimeError("extractor exploded")
        text = self._texts.get(filename, "")
        return ExtractResult(
            text=text, status="extracted" if text else "empty", extractor=self._name
        )


class TestMeasureText(unittest.TestCase):
    def test_counts_chars_words_unique_and_digits(self):
        # Words are runs of 2+ letters, so "Q1" does not count but "April" and
        # the "th" of "10th" do — the signal is text volume, not grammar.
        chars, words, uwords, digits = measure_text("  Q1 Q1 April 10th  ")
        self.assertEqual(chars, len("Q1 Q1 April 10th"))
        self.assertEqual((words, uwords), (2, 2))
        self.assertEqual(digits, 4)

    def test_repeated_words_raise_count_but_not_unique_count(self):
        _, words, uwords, _ = measure_text("logo logo logo")
        self.assertEqual((words, uwords), (3, 1))

    def test_empty_text_measures_zero(self):
        self.assertEqual(measure_text(""), (0, 0, 0, 0))
        self.assertEqual(measure_text(None), (0, 0, 0, 0))


class TestEngineIndependence(unittest.TestCase):
    """The same image must measure the same whichever OCR engine ran.

    The LLM vision provider answers with a DESCRIPTION preamble before the
    transcription. Measured raw, that inflated a three-word newsletter header
    from 22 chars (tesseract) to 159 (llm) — across the 100-char "text-rich"
    threshold, i.e. decoration would have been classified as content purely
    because of which engine ran.
    """

    LLM = (
        'DESCRIPTION: The image shows the text "A Note from Enablement" in a teal-colored, '
        "monospaced font, underlined by a thin teal line.\nTEXT:\nA Note from Enablement"
    )
    TESSERACT = "A Note from\nEnablement"

    def test_llm_preamble_is_stripped_before_measuring(self):
        self.assertEqual(measure_text(self.LLM), measure_text(self.TESSERACT))

    def test_measured_length_is_the_transcription_not_the_description(self):
        self.assertEqual(measure_text(self.LLM)[0], len("A Note from Enablement"))

    def test_tesseract_output_is_unaffected(self):
        self.assertEqual(measure_text("plain text 42"), (13, 2, 2, 2))

    def test_uncalibrated_extractor_yields_no_opinion(self):
        # Thresholds are tuned per engine; an unknown one falls back to the
        # metadata heuristic rather than applying numbers meant for another.
        sig = BlobSignals(500, 80, 70, 10, 0, 0, "extracted", "some-future-ocr")
        self.assertIsNone(is_decoration(sig, thread_count=50, inline=True))


class TestIsDecoration(unittest.TestCase):
    """Verdicts for the real images labelled during calibration."""

    def test_text_rich_beats_the_recurrence_heuristic(self):
        # The quarterly reporting-deadline table: 209 chars, 15 threads. The
        # metadata heuristic removed it; measuring rescues it.
        self.assertIs(is_decoration(_sig(209), thread_count=15, inline=True), False)

    def test_text_poor_and_widely_reused_is_decoration(self):
        # The "A Note from Enablement" newsletter header: 22 chars, 52 threads.
        self.assertIs(is_decoration(_sig(22), thread_count=52, inline=True), True)

    def test_text_poor_but_not_reused_is_kept(self):
        # A pasted "401 Access is denied" screenshot: 29 chars, 1 thread. Content
        # someone sent deliberately — text-poor is NOT the same as decoration.
        self.assertIsNone(is_decoration(_sig(29), thread_count=1, inline=True))

    def test_ubiquitous_image_is_decoration_even_when_text_rich(self):
        """Text-richness must not rescue boilerplate that is simply everywhere.

        Real images from the corpus: a legal confidentiality disclaimer rendered
        as an image (748 chars, 829 threads) and a signature block with name,
        title and phone numbers (195 chars, 61 threads). Both are text-rich;
        neither is content. Nothing genuine appears in that many unrelated
        conversations.
        """
        self.assertIs(is_decoration(_sig(748), thread_count=829, inline=True), True)
        self.assertIs(is_decoration(_sig(195), thread_count=61, inline=True), True)

    def test_content_just_below_the_ubiquity_ceiling_survives(self):
        # Pinned at the boundary itself rather than at 15 (already covered by
        # test_text_rich_beats_the_recurrence_heuristic): one thread lower must
        # still be content, or the ceiling is off by one.
        self.assertIs(
            is_decoration(_sig(209), thread_count=UBIQUITOUS_THREADS - 1, inline=True), False
        )

    def test_content_at_the_ubiquity_ceiling_is_decoration(self):
        self.assertIs(is_decoration(_sig(209), thread_count=UBIQUITOUS_THREADS, inline=True), True)

    def test_ubiquity_ceiling_does_not_apply_to_real_enclosures(self):
        # A boilerplate PDF sent to hundreds of threads is still a document.
        self.assertIs(is_decoration(_sig(5000), thread_count=900, inline=False), False)

    def test_non_inline_is_never_decoration(self):
        self.assertIs(is_decoration(_sig(0), thread_count=99, inline=False), False)

    def test_unmeasured_blob_has_no_opinion(self):
        self.assertIsNone(is_decoration(None, thread_count=99, inline=True))

    def test_failed_extraction_has_no_opinion(self):
        # Must fall back to the heuristic rather than silently recategorising.
        for status in ("error", "ocr_unavailable", "unsupported", "binary"):
            with self.subTest(status=status):
                self.assertIsNone(
                    is_decoration(_sig(0, status=status), thread_count=99, inline=True)
                )


class TestClassifyPass(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="classify_")
        self.store = AttachmentStore(self.dir)

    def tearDown(self):
        self.store.close()
        shutil.rmtree(self.dir, ignore_errors=True)

    def _put(self, data, *, msg, thread, name, mime="image/png", inline=True):
        return self.store.put(
            data,
            message_id=msg,
            thread_id=thread,
            filename=name,
            mime=mime,
            size=len(data),
            source_type="eml",
            source_ref=f"/c/{msg}.eml",
            inline=inline,
        )

    def test_measures_each_blob_once_by_content_hash(self):
        logo = b"L" * 500
        for i in range(6):
            self._put(logo, msg=f"m{i}", thread=f"t{i}", name="logo.png")
        stats = classify_blobs(self.store, extractor=_FakeExtractor())
        # Six rows, one blob: measurement is keyed by sha256, so it runs once.
        self.assertEqual(stats.measured, 1)

    def test_is_resumable_and_idempotent(self):
        self._put(b"A" * 100, msg="m1", thread="t1", name="a.png")
        first = classify_blobs(self.store, extractor=_FakeExtractor())
        second = classify_blobs(self.store, extractor=_FakeExtractor())
        self.assertEqual(first.measured, 1)
        self.assertEqual(second.measured, 0, "already-measured blobs are not redone")

    def test_max_size_bounds_the_pass(self):
        self._put(b"S" * 50, msg="m1", thread="t1", name="small.png")
        self._put(b"B" * 5000, msg="m2", thread="t2", name="big.png")
        stats = classify_blobs(self.store, extractor=_FakeExtractor(), max_size=1000)
        self.assertEqual(stats.measured, 1)

    def test_extraction_failure_is_counted_not_raised(self):
        # One unreadable blob must not abort a bulk pass over thousands.
        self._put(b"X" * 100, msg="m1", thread="t1", name="x.png")
        stats = classify_blobs(self.store, extractor=_FakeExtractor(raises=True))
        self.assertEqual(stats.failed, 1)
        self.assertEqual(stats.measured, 0)

    def test_measured_signals_override_the_heuristic_in_listings(self):
        """End-to-end: the case the metadata heuristic got wrong.

        A small image reused across many threads looks like a logo, but this one
        is a reference table people quote precisely because it is useful.
        """
        table = b"T" * 5000
        for i in range(8):
            self._put(table, msg=f"m{i}", thread=f"t{i}", name="table.png")
        before = self.store.list_for(thread_id="t0", include_boilerplate=False)
        self.assertEqual(before, [], "heuristic alone filters it out")

        classify_blobs(
            self.store,
            extractor=_FakeExtractor({"table.png": "Q1 January 10th Q2 April 10th " * 6}),
        )
        after = self.store.list_for(thread_id="t0", include_boilerplate=False)
        self.assertEqual([m.filename for m in after], ["table.png"])

    def test_measurement_confirms_the_heuristic_for_real_decoration(self):
        logo = b"L" * 500
        for i in range(8):
            self._put(logo, msg=f"m{i}", thread=f"t{i}", name="logo.png")
        classify_blobs(self.store, extractor=_FakeExtractor({"logo.png": "Acme"}))
        self.assertEqual(self.store.list_for(thread_id="t0", include_boilerplate=False), [])

    def test_signals_round_trip_through_the_store(self):
        sha = self._put(b"A" * 100, msg="m1", thread="t1", name="a.png")
        self.assertIsNone(self.store.get_signals(sha))
        classify_blobs(self.store, extractor=_FakeExtractor({"a.png": "hello world 42"}))
        got = self.store.get_signals(sha)
        self.assertEqual(got.chars, len("hello world 42"))
        self.assertEqual(got.digits, 2)
        self.assertEqual(got.status, "extracted")


class TestMeasureBlob(unittest.TestCase):
    def test_records_status_and_extractor_from_the_result(self):
        sig = measure_blob(b"x", "image/png", "a.png", _FakeExtractor({"a.png": "abc 12"}))
        self.assertEqual(sig.chars, 6)
        self.assertEqual(sig.digits, 2)
        self.assertEqual(sig.status, "extracted")
        self.assertEqual(sig.extractor, "tesseract")

    def test_unreadable_image_still_measures_text(self):
        # Dimensions fail closed at (0, 0); the text signal is what decides.
        sig = measure_blob(b"not-an-image", "image/png", "a.png", _FakeExtractor({"a.png": "hi"}))
        self.assertEqual((sig.width, sig.height), (0, 0))
        self.assertEqual(sig.chars, 2)


if __name__ == "__main__":
    unittest.main()


class TestClassifyDefaultsToTesseract(unittest.TestCase):
    """The CLI must not silently upgrade the cheap pass to the vision LLM.

    Regression: `attachments build` passed `args.extractor` straight into
    `build_default_extractor()`. With no `--extractor` that is `None`, which
    resolves `$RAG_ATTACH_EXTRACTOR` and falls back to `llm` — bypassing the
    tesseract default that makes bulk measurement affordable. Measured on the
    same images, the LLM path is 16-20x slower, so the whole pass would have
    gone from minutes to hours without saying so.
    """

    def test_classify_uses_tesseract_when_no_extractor_is_requested(self):
        from unittest import mock

        import src.attachments.classify as classify

        with mock.patch.object(classify, "measure_blob"):
            with mock.patch("src.attachments.extract.build_default_extractor") as build:
                classify.classify_blobs(_OneBlobStore(), extractor=None)
        self.assertEqual(build.call_args.args[0], "tesseract")

    def test_no_engine_is_built_when_there_is_nothing_to_measure(self):
        # A re-run over an already-measured corpus should not depend on an OCR
        # engine being installable, let alone installed.
        from unittest import mock

        import src.attachments.classify as classify

        with mock.patch("src.attachments.extract.build_default_extractor") as build:
            stats = classify.classify_blobs(_EmptyStore(), extractor=None)
        build.assert_not_called()
        self.assertEqual(stats.measured, 0)

    def test_env_can_override_the_classify_engine(self):
        import os
        from unittest import mock

        import src.attachments.classify as classify

        with mock.patch.dict(os.environ, {"RAG_ATTACH_CLASSIFY_EXTRACTOR": "llm"}):
            with mock.patch("src.attachments.extract.build_default_extractor") as build:
                classify.classify_blobs(_OneBlobStore(), extractor=None)
        self.assertEqual(build.call_args.args[0], "llm")


class TestBuildVerbWiring(unittest.TestCase):
    """`attachments build` must leave the engine choice to classify_blobs."""

    def test_no_extractor_flag_means_no_extractor_is_constructed(self):
        import argparse
        from unittest import mock

        import src.cli as cli

        args = argparse.Namespace(
            profile="p.json",
            store="/tmp/s",
            limit=None,
            no_classify=False,
            classify_max_size=100_000,
            extractor=None,
        )
        with (
            mock.patch.object(cli, "CorpusProfile"),
            mock.patch.object(cli, "resolve_index_files", return_value=([], None)),
            mock.patch.object(cli, "AttachmentStore"),
            mock.patch.object(cli, "ingest_eml", return_value={}),
            mock.patch("src.attachments.classify.classify_blobs") as classify,
            mock.patch("src.attachments.extract.build_default_extractor") as build,
        ):
            cli._cmd_attachments_build(args)
        self.assertIsNone(classify.call_args.kwargs["extractor"])
        build.assert_not_called()

    def test_an_explicit_extractor_flag_is_honoured(self):
        """The complementary path: --extractor must still reach classify_blobs.

        The fix guards against building an engine when none was asked for; it
        must not also swallow one that was.
        """
        import argparse
        from unittest import mock

        import src.cli as cli

        args = argparse.Namespace(
            profile="p.json",
            store="/tmp/s",
            limit=None,
            no_classify=False,
            classify_max_size=100_000,
            extractor="llm",
        )
        with (
            mock.patch.object(cli, "CorpusProfile"),
            mock.patch.object(cli, "resolve_index_files", return_value=([], None)),
            mock.patch.object(cli, "AttachmentStore"),
            mock.patch.object(cli, "ingest_eml", return_value={}),
            mock.patch("src.attachments.classify.classify_blobs") as classify,
            mock.patch("src.attachments.extract.build_default_extractor") as build,
        ):
            cli._cmd_attachments_build(args)
        build.assert_called_once_with("llm")
        self.assertIs(classify.call_args.kwargs["extractor"], build.return_value)


class _EmptyStore:
    """No work to do — classify must return before constructing an engine."""

    def unmeasured_blobs(self, **kw):
        return []


class _OneBlobStore:
    def unmeasured_blobs(self, **kw):
        return [("sha", "image/png", "a.png", 100)]

    def path_for(self, sha):
        return "/nonexistent/blob"

    def put_signals(self, *a, **k):
        pass
