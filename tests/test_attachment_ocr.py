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
