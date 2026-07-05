import unittest

from src.attachments.extract.mime import is_pdf, mime_base, mime_charset


class TestMimeBase(unittest.TestCase):
    def test_strips_parameters_and_lowercases(self):
        self.assertEqual(mime_base("Text/Plain; charset=utf-8"), "text/plain")
        self.assertEqual(mime_base("application/pdf"), "application/pdf")

    def test_none_and_empty(self):
        self.assertEqual(mime_base(None), "")
        self.assertEqual(mime_base(""), "")


class TestMimeCharset(unittest.TestCase):
    def test_reads_charset_parameter(self):
        self.assertEqual(mime_charset("text/plain; charset=Shift_JIS"), "shift_jis")

    def test_quoted_value(self):
        self.assertEqual(mime_charset('text/plain; charset="iso-8859-1"'), "iso-8859-1")

    def test_absent_charset_is_none(self):
        self.assertIsNone(mime_charset("text/plain"))
        self.assertIsNone(mime_charset(None))

    def test_other_params_ignored(self):
        self.assertEqual(mime_charset("text/plain; format=flowed; charset=utf-8"), "utf-8")


class TestIsPdf(unittest.TestCase):
    def test_by_mime_case_insensitive(self):
        self.assertTrue(is_pdf("Application/PDF", "x.bin"))
        self.assertTrue(is_pdf("application/pdf; name=x", "x.bin"))

    def test_by_filename(self):
        self.assertTrue(is_pdf("", "REPORT.PDF"))
        self.assertTrue(is_pdf(None, "a.pdf"))

    def test_negative(self):
        self.assertFalse(is_pdf("text/plain", "a.txt"))
        self.assertFalse(is_pdf(None, None))


if __name__ == "__main__":
    unittest.main()
