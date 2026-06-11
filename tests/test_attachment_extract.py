import unittest
from src.attachments.extract import Extractor, Status
from src.attachments.extract.ocr.base import OcrResult


class _FakeOcr:
    def __init__(self, result):
        self.result = result

    def read(self, data, mime, filename):
        return self.result


def _ex(ocr_text="ocr text", ocr_status=Status.EXTRACTED):
    return Extractor(_FakeOcr(OcrResult(ocr_text, ocr_status, "fake")))


class TestExtractBehaviour(unittest.TestCase):
    def test_plain_text(self):
        r = _ex().extract(b"hello world", "text/plain", "a.txt")
        self.assertEqual(r.status, Status.EXTRACTED)
        self.assertIn("hello world", r.text)

    def test_html_strips_tags(self):
        r = _ex().extract(b"<p>Hi <b>there</b></p>", "text/html", "x.html")
        self.assertEqual(r.status, Status.EXTRACTED)
        self.assertIn("Hi", r.text)
        self.assertNotIn("<p>", r.text)

    def test_unknown_type_is_unsupported(self):
        r = _ex().extract(b"\x00\x01\x02", "application/x-thing", "x.bin")
        self.assertEqual(r.status, Status.UNSUPPORTED)

    def test_image_uses_ocr(self):
        r = _ex(ocr_text="scanned").extract(b"\x89PNG", "image/png", "x.png")
        self.assertEqual(r.status, Status.EXTRACTED)
        self.assertEqual(r.text, "scanned")


if __name__ == "__main__":
    unittest.main()
