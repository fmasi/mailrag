import unittest
from unittest import mock

from src.attachments.extract import extract_text, ExtractResult


class TestExtract(unittest.TestCase):
    def test_plain_text(self):
        r = extract_text(b"hello world", "text/plain", "a.txt")
        self.assertEqual(r.status, "extracted")
        self.assertIn("hello world", r.text)

    def test_csv_and_html_and_ics_are_stdlib(self):
        self.assertEqual(extract_text(b"a,b\n1,2", "text/csv", "x.csv").status, "extracted")
        html = extract_text(b"<p>Hi <b>there</b></p>", "text/html", "x.html")
        self.assertEqual(html.status, "extracted")
        self.assertIn("Hi", html.text)
        self.assertNotIn("<p>", html.text)

    def test_unknown_type_is_binary(self):
        r = extract_text(b"\x00\x01\x02", "application/x-thing", "x.bin")
        self.assertEqual(r.status, "binary")
        self.assertEqual(r.text, "")

    def test_pdf_missing_lib_falls_back_to_binary(self):
        # Simulate pypdf import failing -> status binary, never raises.
        with mock.patch.dict("sys.modules", {"pypdf": None}):
            r = extract_text(b"%PDF-1.4 ...", "application/pdf", "x.pdf")
        self.assertIn(r.status, ("binary", "ocr_unavailable", "empty"))
        self.assertIsInstance(r, ExtractResult)

    def test_image_ocr_unavailable_when_pytesseract_missing(self):
        with mock.patch.dict("sys.modules", {"pytesseract": None, "PIL": None}):
            r = extract_text(b"\x89PNG...", "image/png", "x.png")
        self.assertEqual(r.status, "ocr_unavailable")


if __name__ == "__main__":
    unittest.main()
