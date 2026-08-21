"""Image attachments must reach OCR, and damaged ones must not be discarded.

Three failures found by auditing extraction over a real corpus:

* `ImageHandler.can_handle` tested the mime only, while `is_pdf` had always
  accepted a filename fallback. An image declared `application/octet-stream` —
  routine from senders and exporters — therefore came back `unsupported` while a
  PDF in the identical situation extracted fine.
* PIL refuses a JPEG missing its final bytes, discarding a whole photograph over
  20 unread ones.
* pytesseract rejects some image modes outright; a 4032x3024 photo failed that
  way and produced 560 characters once converted to RGB.

Scale is honest: 14 blobs of 12,333 hit the first case. The reason to fix it is
the inconsistency and the silent loss, not the volume.
"""

import io
import unittest

from src.attachments.extract.mime import is_image, is_pdf


class TestImagePredicate(unittest.TestCase):
    def test_image_mime_is_an_image(self):
        for mime in ("image/png", "image/jpeg", "image/gif", "IMAGE/PNG"):
            with self.subTest(mime=mime):
                self.assertTrue(is_image(mime, "whatever"))

    def test_image_suffix_survives_a_wrong_mime(self):
        # The actual bug: routine octet-stream labelling sent images nowhere.
        for name in ("receipt.png", "photo.JPG", "scan.jpeg", "diagram.tiff", "x.heic"):
            with self.subTest(name=name):
                self.assertTrue(is_image("application/octet-stream", name))

    def test_non_images_are_not_images(self):
        self.assertFalse(is_image("application/pdf", "doc.pdf"))
        self.assertFalse(is_image("application/octet-stream", "notes.txt"))
        self.assertFalse(is_image(None, None))

    def test_it_mirrors_the_pdf_predicate(self):
        # Both predicates should behave the same way about a wrong mime; the
        # asymmetry between them was the defect.
        self.assertTrue(is_pdf("application/octet-stream", "a.pdf"))
        self.assertTrue(is_image("application/octet-stream", "a.png"))


class TestTruncatedAndAwkwardImages(unittest.TestCase):
    """Real mail carries damaged images; refusing them loses the content."""

    def _jpeg(self, size=(40, 30), mode="RGB"):
        from PIL import Image

        buf = io.BytesIO()
        Image.new(mode, size, "white").save(buf, format="JPEG")
        return buf.getvalue()

    def test_a_truncated_jpeg_still_decodes(self):
        from src.attachments.extract.ocr.tesseract import TesseractOcr

        data = self._jpeg()[:-20]  # chop the tail, as a real damaged file has
        result = TesseractOcr().read(data, "image/jpeg", "cut.jpg")
        # Either OCR ran (extracted/empty) or the engine is absent; what must not
        # happen is an error caused by the truncation itself.
        self.assertIn(result.status, ("extracted", "empty", "ocr_unavailable"))

    def test_a_mode_pytesseract_dislikes_is_converted(self):
        from src.attachments.extract.ocr.tesseract import TesseractOcr

        data = self._jpeg(mode="CMYK")
        result = TesseractOcr().read(data, "image/jpeg", "cmyk.jpg")
        self.assertIn(result.status, ("extracted", "empty", "ocr_unavailable"))

    def test_genuine_rubbish_still_reports_an_error(self):
        # The fixes must not turn every failure into a silent success.
        from src.attachments.extract.ocr.tesseract import TesseractOcr

        result = TesseractOcr().read(b"not an image at all", "image/png", "x.png")
        self.assertEqual(result.status, "error")


class TestRouting(unittest.TestCase):
    def test_an_octet_stream_image_reaches_the_image_handler(self):
        from src.attachments.extract.handlers.image import ImageHandler

        class _Ocr:
            def read(self, data, mime, filename):
                from src.attachments.extract.ocr.base import OcrResult

                return OcrResult("text from ocr", "extracted", "fake")

        handler = ImageHandler(_Ocr())
        self.assertTrue(handler.can_handle("application/octet-stream", "photo.png"))
        self.assertEqual(
            handler.extract(b"x", "application/octet-stream", "photo.png").text, "text from ocr"
        )


if __name__ == "__main__":
    unittest.main()
