"""printable(): email-derived text made safe for a terminal."""

import unittest

from src.terminal_text import printable


class TestPrintable(unittest.TestCase):
    def test_control_codes_are_replaced(self):
        self.assertEqual(printable("a\x1b[2Jb\x07c\rd\ne"), "a?[2Jb?c?d?e")

    def test_ordinary_text_and_unicode_pass_through(self):
        self.assertEqual(printable("Café — 日本 [x]"), "Café — 日本 [x]")

    def test_empty(self):
        self.assertEqual(printable(""), "")


if __name__ == "__main__":
    unittest.main()
