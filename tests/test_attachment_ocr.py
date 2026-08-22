import io
import os
import unittest
from unittest import mock

from src.attachments.extract.ocr.base import ChainedOcr, OcrResult
from src.attachments.extract.ocr.registry import default_extractor_name, resolve
from src.attachments.extract.ocr.tesseract import TesseractOcr
from src.attachments.extract.registry import Extractor
from src.attachments.extract.result import ExtractResult, Status, ok


class TestExtractResult(unittest.TestCase):
    def test_status_constants(self):
        self.assertEqual(Status.EXTRACTED, "extracted")
        self.assertEqual(Status.EMPTY, "empty")
        self.assertEqual(Status.BINARY, "binary")
        self.assertEqual(Status.UNSUPPORTED, "unsupported")
        self.assertEqual(Status.OCR_UNAVAILABLE, "ocr_unavailable")
        self.assertEqual(Status.ERROR, "error")

    def test_extract_result_fields(self):
        r = ExtractResult(text="hi", status=Status.EXTRACTED, extractor="plaintext")
        self.assertEqual((r.text, r.status, r.extractor), ("hi", "extracted", "plaintext"))

    def test_ok_with_text(self):
        r = ok("hello", "plaintext")
        self.assertEqual((r.status, r.text, r.extractor), (Status.EXTRACTED, "hello", "plaintext"))

    def test_ok_empty_string(self):
        self.assertEqual(ok("", "x").status, Status.EMPTY)

    def test_ok_whitespace_only(self):
        self.assertEqual(ok("   \n", "x").status, Status.EMPTY)


class _Fake:
    def __init__(self, result):
        self.result = result
        self.calls = 0

    def read(self, data, mime, filename):
        self.calls += 1
        return self.result


class TestChainedOcr(unittest.TestCase):
    def test_first_success_wins_second_not_called(self):
        a = _Fake(OcrResult("hello", Status.EXTRACTED, "a"))
        b = _Fake(OcrResult("nope", Status.EXTRACTED, "b"))
        out = ChainedOcr([a, b]).read(b"x", "image/png", "x.png")
        self.assertEqual(out.text, "hello")
        self.assertEqual(b.calls, 0)

    def test_unavailable_falls_through(self):
        a = _Fake(OcrResult("", Status.OCR_UNAVAILABLE, "a"))
        b = _Fake(OcrResult("found", Status.EXTRACTED, "b"))
        out = ChainedOcr([a, b]).read(b"x", "image/png", "x.png")
        self.assertEqual((out.text, out.provider), ("found", "b"))

    def test_empty_falls_through(self):
        a = _Fake(OcrResult("", Status.EMPTY, "a"))
        b = _Fake(OcrResult("found", Status.EXTRACTED, "b"))
        self.assertEqual(ChainedOcr([a, b]).read(b"x", "image/png", "x.png").text, "found")

    def test_all_unavailable_returns_ocr_unavailable(self):
        a = _Fake(OcrResult("", Status.OCR_UNAVAILABLE, "a"))
        b = _Fake(OcrResult("", Status.OCR_UNAVAILABLE, "b"))
        out = ChainedOcr([a, b]).read(b"x", "image/png", "x.png")
        self.assertEqual(out.status, Status.OCR_UNAVAILABLE)
        self.assertEqual(out.provider, "b")

    def test_error_is_returned_not_swallowed(self):
        a = _Fake(OcrResult("", Status.ERROR, "a"))
        b = _Fake(OcrResult("found", Status.EXTRACTED, "b"))
        self.assertEqual(ChainedOcr([a, b]).read(b"x", "image/png", "x.png").text, "found")

    def test_empty_chain_returns_unavailable_sentinel(self):
        out = ChainedOcr([]).read(b"x", "image/png", "x.png")
        self.assertEqual(out.status, Status.OCR_UNAVAILABLE)
        self.assertEqual(out.provider, "none")


class TestTesseractOcr(unittest.TestCase):
    def test_unavailable_when_pytesseract_missing(self):
        with mock.patch.dict("sys.modules", {"pytesseract": None, "PIL": None}):
            out = TesseractOcr().read(b"\x89PNG...", "image/png", "x.png")
        self.assertEqual(out.status, Status.OCR_UNAVAILABLE)
        self.assertEqual(out.provider, "tesseract")

    def test_runtime_failure_is_error_not_unavailable(self):
        fake_pt = mock.MagicMock()
        fake_pt.image_to_string.side_effect = RuntimeError("tesseract blew up")
        fake_pil = mock.MagicMock()
        with mock.patch.dict("sys.modules", {"pytesseract": fake_pt, "PIL": fake_pil}):
            out = TesseractOcr().read(b"\x89PNG...", "image/png", "x.png")
        self.assertEqual(out.status, Status.ERROR)

    @staticmethod
    def _jpeg():
        from PIL import Image

        buf = io.BytesIO()
        Image.new("RGB", (40, 30), "white").save(buf, format="JPEG")
        return buf.getvalue()

    def test_a_missing_tesseract_binary_is_unavailable_not_error(self):
        """The engine being absent is an ENVIRONMENT verdict, not a bad attachment.

        ``pytesseract`` imports fine without the ``tesseract`` binary and only
        fails when it shells out, so the import probe above cannot catch this.
        Classifying it as ERROR poisons the cache: ``AttachmentStore`` refuses to
        cache OCR_UNAVAILABLE precisely so a later run with a working PATH
        retries, but it caches ERROR forever (GH #37). A scheduled sync inherits
        no PATH, which is exactly when this fires.
        """
        with mock.patch.dict(os.environ, {"PATH": "/nonexistent"}):
            out = TesseractOcr().read(self._jpeg(), "image/jpeg", "x.jpg")
        self.assertEqual(out.status, Status.OCR_UNAVAILABLE)
        self.assertEqual(out.provider, "tesseract")

    def test_a_missing_engine_is_told_apart_from_a_corrupt_image(self):
        """Both arrive as OSError, so the two must not be separated by type alone.

        ``TesseractNotFoundError`` subclasses ``OSError`` and so does Pillow's
        "image file is truncated" — catching ``OSError`` would collapse them.
        """

        class TesseractNotFoundError(OSError):
            pass

        fake_pt = mock.MagicMock()
        fake_pt.image_to_string.side_effect = TesseractNotFoundError("not installed")
        fake_pil = mock.MagicMock()
        with mock.patch.dict("sys.modules", {"pytesseract": fake_pt, "PIL": fake_pil}):
            out = TesseractOcr().read(b"\x89PNG...", "image/png", "x.png")
        self.assertEqual(out.status, Status.OCR_UNAVAILABLE)

        # A same-typed failure that is NOT the engine going missing stays ERROR.
        fake_pt.image_to_string.side_effect = OSError("image file is truncated")
        with mock.patch.dict("sys.modules", {"pytesseract": fake_pt, "PIL": fake_pil}):
            out = TesseractOcr().read(b"\x89PNG...", "image/png", "x.png")
        self.assertEqual(out.status, Status.ERROR)

    def test_a_missing_poppler_is_unavailable_not_error(self):
        """Same contract on the PDF path — ``render_pdf_pages`` documents it."""

        class PopplerNotInstalledError(Exception):
            pass

        fake_p2i = mock.MagicMock()
        fake_p2i.convert_from_bytes.side_effect = PopplerNotInstalledError("poppler missing")
        fake_pt = mock.MagicMock()
        with mock.patch.dict("sys.modules", {"pdf2image": fake_p2i, "pytesseract": fake_pt}):
            out = TesseractOcr().read(b"%PDF-1.4", "application/pdf", "x.pdf")
        self.assertEqual(out.status, Status.OCR_UNAVAILABLE)

    def _pdf_status(self, side_effect):
        """Classify a PDF-render failure with poppler's own exceptions mocked out.

        ``pdfinfo`` is made to succeed so the failure can only come from the
        pdftoppm/pdftocairo half of poppler — the partial-install shape.
        """
        fake_p2i = mock.MagicMock()
        fake_p2i.pdfinfo_from_bytes.return_value = {"Pages": 1}
        fake_p2i.convert_from_bytes.side_effect = side_effect
        fake_pt = mock.MagicMock()
        with mock.patch.dict("sys.modules", {"pdf2image": fake_p2i, "pytesseract": fake_pt}):
            return TesseractOcr().read(b"%PDF-1.4", "application/pdf", "x.pdf")

    def test_a_partial_poppler_install_is_unavailable_not_error(self):
        """pdfinfo present but pdftoppm absent raises a BARE ``FileNotFoundError``.

        pdf2image guards only its ``pdfinfo`` call (OSError ->
        PDFInfoNotInstalledError). The ``pdftoppm``/``pdftocairo`` version probe
        inside ``convert_from_path`` is an unguarded ``Popen``, so a half-installed
        poppler escapes every name in ``_ENGINE_MISSING`` and used to be cached as
        ERROR forever — GH #37's exact failure, reached by a narrower path.
        """
        out = self._pdf_status(FileNotFoundError(2, "No such file or directory", "pdftoppm"))
        self.assertEqual(out.status, Status.OCR_UNAVAILABLE)
        self.assertEqual(out.provider, "tesseract")

    def test_a_missing_pdftocairo_is_unavailable_too(self):
        """pdf2image probes pdftocairo instead for transparent/PNG-ish formats."""
        out = self._pdf_status(FileNotFoundError(2, "No such file or directory", "pdftocairo"))
        self.assertEqual(out.status, Status.OCR_UNAVAILABLE)

    def test_a_poppler_binary_missing_from_an_explicit_prefix_is_unavailable(self):
        """``poppler_path`` makes the failed executable an absolute path, not a name."""
        out = self._pdf_status(
            FileNotFoundError(2, "No such file or directory", "/opt/poppler/bin/pdftoppm")
        )
        self.assertEqual(out.status, Status.OCR_UNAVAILABLE)

    def test_a_missing_binary_wrapped_by_another_error_is_still_unavailable(self):
        """The cause/context walk must see through a re-raise."""

        def boom(*_a, **_k):
            try:
                raise FileNotFoundError(2, "No such file or directory", "pdftoppm")
            except FileNotFoundError as exc:
                raise RuntimeError("render failed") from exc

        out = self._pdf_status(boom)
        self.assertEqual(out.status, Status.OCR_UNAVAILABLE)

    def test_a_missing_data_file_stays_error_not_unavailable(self):
        """Precision guard: not every ``FileNotFoundError`` is a missing engine.

        Calling a genuinely unreadable input "environment issue, retry later"
        would be the same misclassification in reverse — the attachment would be
        re-OCR'd on every run and never settle.
        """
        out = self._pdf_status(FileNotFoundError(2, "No such file or directory", "/tmp/scan.pdf"))
        self.assertEqual(out.status, Status.ERROR)

    def test_a_filenameless_missing_file_stays_error(self):
        """No ``filename`` means no evidence it was poppler; default to ERROR."""
        out = self._pdf_status(FileNotFoundError("something went missing"))
        self.assertEqual(out.status, Status.ERROR)

    def test_pdf_unavailable_when_pdf2image_missing(self):
        with mock.patch.dict("sys.modules", {"pdf2image": None, "pytesseract": None}):
            out = TesseractOcr().read(b"%PDF-1.4", "application/pdf", "x.pdf")
        self.assertEqual(out.status, Status.OCR_UNAVAILABLE)
        self.assertEqual(out.provider, "tesseract")

    def test_pdf_runtime_failure_is_error(self):
        fake_p2i = mock.MagicMock()
        fake_p2i.convert_from_bytes.side_effect = RuntimeError("poppler blew up")
        fake_pt = mock.MagicMock()
        with mock.patch.dict("sys.modules", {"pdf2image": fake_p2i, "pytesseract": fake_pt}):
            out = TesseractOcr().read(b"%PDF-1.4", "application/pdf", "x.pdf")
        self.assertEqual(out.status, Status.ERROR)

    def test_load_truncated_images_flag_is_restored_after_success(self):
        """GH #185: the process-wide PIL flag must not leak past this one call.

        Other PIL call sites (signals.py, pages.py, llm_vision.py) rely on PIL's
        default of raising on a truncated image to detect corrupt uploads. If
        ``_image`` sets ``ImageFile.LOAD_TRUNCATED_IMAGES = True`` and never
        restores it, every later PIL call in the same long-running process
        (MCP server, batch job) silently accepts truncated data forever, with no
        way to tell OCR was ever invoked.
        """
        from PIL import ImageFile

        fake_pt = mock.MagicMock()
        fake_pt.image_to_string.return_value = "text"
        with mock.patch.object(ImageFile, "LOAD_TRUNCATED_IMAGES", False):
            with mock.patch.dict("sys.modules", {"pytesseract": fake_pt}):
                TesseractOcr()._image(self._jpeg())
            self.assertFalse(
                ImageFile.LOAD_TRUNCATED_IMAGES,
                "flag must be restored to its pre-call value (False), not leaked as True",
            )

    def test_load_truncated_images_flag_is_restored_after_failure(self):
        """Same contract when the call raises: still must not leak the flag."""
        from PIL import ImageFile

        fake_pt = mock.MagicMock()
        fake_pt.image_to_string.side_effect = RuntimeError("tesseract blew up")
        with mock.patch.object(ImageFile, "LOAD_TRUNCATED_IMAGES", False):
            with mock.patch.dict("sys.modules", {"pytesseract": fake_pt}):
                TesseractOcr()._image(self._jpeg())
            self.assertFalse(ImageFile.LOAD_TRUNCATED_IMAGES)

    def test_load_truncated_images_flag_restored_when_already_true(self):
        """A prior leak (real or from another call) must not be masked as fixed.

        If the flag started ``True`` (e.g. another call site, or an earlier
        leak, already flipped it) the restore must put it back to ``True``, not
        ``False`` — a naive ``finally: ImageFile.LOAD_TRUNCATED_IMAGES = False``
        would pass the "starts False" test above while still corrupting state
        for callers that intentionally rely on it being True.
        """
        from PIL import ImageFile

        fake_pt = mock.MagicMock()
        fake_pt.image_to_string.return_value = "text"
        with mock.patch.object(ImageFile, "LOAD_TRUNCATED_IMAGES", True):
            with mock.patch.dict("sys.modules", {"pytesseract": fake_pt}):
                TesseractOcr()._image(self._jpeg())
            self.assertTrue(ImageFile.LOAD_TRUNCATED_IMAGES)


class TestExtractorFacade(unittest.TestCase):
    def setUp(self):
        self.ocr = _Fake(OcrResult("img text", Status.EXTRACTED, "fake"))
        self.ex = Extractor(self.ocr)

    def test_dispatches_text(self):
        self.assertEqual(
            self.ex.extract(b"hi there", "text/plain", "a.txt").status, Status.EXTRACTED
        )

    def test_image_uses_injected_ocr(self):
        r = self.ex.extract(b"\x89PNG", "image/png", "x.png")
        self.assertEqual(r.text, "img text")
        self.assertEqual(self.ocr.calls, 1)

    def test_unknown_type_is_unsupported(self):
        r = self.ex.extract(b"\x00\x01", "application/x-thing", "x.bin")
        self.assertEqual(r.status, Status.UNSUPPORTED)


class TestOcrResolve(unittest.TestCase):
    def test_tesseract_name(self):
        from src.attachments.extract.ocr.tesseract import TesseractOcr

        self.assertIsInstance(resolve("tesseract"), TesseractOcr)

    def test_cloud_is_optin_stub(self):
        with self.assertRaises(NotImplementedError):
            resolve("cloud")

    def test_default_name_from_env(self):
        with mock.patch.dict(os.environ, {"RAG_ATTACH_EXTRACTOR": "tesseract"}):
            self.assertEqual(default_extractor_name(), "tesseract")

    def test_llm_resolves_to_chain(self):
        # LlmVision has landed: "llm" resolves to ChainedOcr([LlmVision, TesseractOcr]).
        from src.attachments.extract.ocr.base import ChainedOcr

        self.assertIsInstance(resolve("llm"), ChainedOcr)


class TestLlmVision(unittest.TestCase):
    @staticmethod
    def _png():
        import io

        from PIL import Image

        b = io.BytesIO()
        Image.new("RGB", (8, 8), "white").save(b, format="PNG")
        return b.getvalue()

    def _provider(self, vision_return="DESCRIPTION: a form\nTEXT:\ninvoice 42"):
        from src.attachments.extract.ocr.llm_vision import LlmVision

        return LlmVision(
            client=mock.MagicMock(), model="gemma", chat_vision=lambda *a, **k: vision_return
        )

    def test_returns_description_and_transcription(self):
        out = self._provider().read(self._png(), "image/png", "x.png")
        self.assertEqual(out.status, Status.EXTRACTED)
        self.assertIn("DESCRIPTION", out.text)
        self.assertIn("invoice 42", out.text)
        self.assertEqual(out.provider, "llm_vision")

    def test_unavailable_when_no_model(self):
        from src.attachments.extract.ocr.llm_vision import LlmVision

        out = LlmVision(client=mock.MagicMock(), model="", chat_vision=lambda *a, **k: "x").read(
            self._png(), "image/png", "x.png"
        )
        self.assertEqual(out.status, Status.OCR_UNAVAILABLE)

    def test_connection_error_is_unavailable_falls_through(self):
        def boom(*a, **k):
            raise ConnectionError("LM Studio down")

        from src.attachments.extract.ocr.llm_vision import LlmVision

        out = LlmVision(client=mock.MagicMock(), model="gemma", chat_vision=boom).read(
            self._png(), "image/png", "x.png"
        )
        self.assertEqual(out.status, Status.OCR_UNAVAILABLE)

    def _fake_p2i(self, pages_total):
        from PIL import Image

        fake = mock.MagicMock()
        fake.pdfinfo_from_bytes.return_value = {"Pages": pages_total}
        fake.convert_from_bytes.return_value = [Image.new("RGB", (4, 4), "white")]
        return fake

    def test_pdf_rendering_is_capped_and_truncation_logged(self):
        from src.attachments.extract.ocr.llm_vision import LlmVision

        fake_p2i = self._fake_p2i(12)
        log = mock.MagicMock()
        p = LlmVision(
            client=mock.MagicMock(),
            model="gemma",
            chat_vision=lambda *a, **k: "DESCRIPTION: x\nTEXT:\nhi",
            log=log,
        )
        with mock.patch.dict("sys.modules", {"pdf2image": fake_p2i}):
            out = p.read(b"%PDF-1.4", "application/pdf", "scan.pdf")
        self.assertEqual(out.status, Status.EXTRACTED)
        _, kwargs = fake_p2i.convert_from_bytes.call_args
        self.assertEqual(
            kwargs.get("last_page"), 10, "must render at most the cap, not the whole PDF"
        )
        log.assert_called_once()
        self.assertIn("12", log.call_args[0][0])

    def test_default_log_emits_to_logging(self):
        from src.attachments.extract.ocr.llm_vision import LlmVision

        fake_p2i = self._fake_p2i(12)
        p = LlmVision(
            client=mock.MagicMock(),
            model="gemma",
            chat_vision=lambda *a, **k: "DESCRIPTION: x\nTEXT:\nhi",
        )
        with mock.patch.dict("sys.modules", {"pdf2image": fake_p2i}):
            with self.assertLogs("mailrag.attachments", level="WARNING"):
                p.read(b"%PDF-1.4", "application/pdf", "scan.pdf")


class TestRenderPdfPages(unittest.TestCase):
    """The shared page-capped PDF renderer used by every OCR provider."""

    def _fake_p2i(self, pages_total):
        fake = mock.MagicMock()
        fake.pdfinfo_from_bytes.return_value = {"Pages": pages_total}
        fake.convert_from_bytes.return_value = ["img"]
        return fake

    def test_never_renders_past_the_cap(self):
        fake = self._fake_p2i(30)
        with mock.patch.dict("sys.modules", {"pdf2image": fake}):
            from src.attachments.extract.ocr.pages import render_pdf_pages

            render_pdf_pages(b"%PDF", log=mock.MagicMock())
        _, kwargs = fake.convert_from_bytes.call_args
        self.assertEqual(kwargs.get("last_page"), 10)

    def test_logs_truncation_when_longer_than_cap(self):
        fake = self._fake_p2i(30)
        log = mock.MagicMock()
        with mock.patch.dict("sys.modules", {"pdf2image": fake}):
            from src.attachments.extract.ocr.pages import render_pdf_pages

            render_pdf_pages(b"%PDF", log=log)
        log.assert_called_once()
        self.assertIn("30", log.call_args[0][0])

    def test_no_truncation_log_when_under_cap(self):
        fake = self._fake_p2i(3)
        log = mock.MagicMock()
        with mock.patch.dict("sys.modules", {"pdf2image": fake}):
            from src.attachments.extract.ocr.pages import render_pdf_pages

            render_pdf_pages(b"%PDF", log=log)
        log.assert_not_called()

    def test_cap_from_env(self):
        fake = self._fake_p2i(30)
        with mock.patch.dict(os.environ, {"RAG_ATTACH_MAX_PAGES": "2"}):
            with mock.patch.dict("sys.modules", {"pdf2image": fake}):
                from src.attachments.extract.ocr.pages import render_pdf_pages

                render_pdf_pages(b"%PDF", log=mock.MagicMock())
        _, kwargs = fake.convert_from_bytes.call_args
        self.assertEqual(kwargs.get("last_page"), 2)

    def test_pdfinfo_failure_still_caps(self):
        fake = self._fake_p2i(0)
        fake.pdfinfo_from_bytes.side_effect = RuntimeError("no poppler info")
        log = mock.MagicMock()
        with mock.patch.dict("sys.modules", {"pdf2image": fake}):
            from src.attachments.extract.ocr.pages import render_pdf_pages

            render_pdf_pages(b"%PDF", log=log)
        _, kwargs = fake.convert_from_bytes.call_args
        self.assertEqual(kwargs.get("last_page"), 10)
        log.assert_not_called()


class TestTesseractPdfCap(unittest.TestCase):
    def test_pdf_rendering_is_page_capped(self):
        fake_p2i = mock.MagicMock()
        fake_p2i.pdfinfo_from_bytes.return_value = {"Pages": 500}
        fake_p2i.convert_from_bytes.return_value = []
        fake_pt = mock.MagicMock()
        with mock.patch.dict("sys.modules", {"pdf2image": fake_p2i, "pytesseract": fake_pt}):
            TesseractOcr().read(b"%PDF-1.4", "application/pdf", "x.pdf")
        _, kwargs = fake_p2i.convert_from_bytes.call_args
        self.assertEqual(
            kwargs.get("last_page"), 10, "tesseract must not OCR every page of an unbounded PDF"
        )

    def test_mixed_case_mime_routes_to_pdf_path(self):
        fake_p2i = mock.MagicMock()
        fake_p2i.pdfinfo_from_bytes.return_value = {"Pages": 1}
        fake_p2i.convert_from_bytes.return_value = []
        fake_pt = mock.MagicMock()
        with mock.patch.dict("sys.modules", {"pdf2image": fake_p2i, "pytesseract": fake_pt}):
            TesseractOcr().read(b"%PDF-1.4", "Application/PDF", "scan.bin")
        self.assertTrue(
            fake_p2i.convert_from_bytes.called, "mixed-case PDF mime must reach the PDF path"
        )


if __name__ == "__main__":
    unittest.main()
