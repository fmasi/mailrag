"""Unit tests for the Mail Archive X loader."""

import os
import tempfile
import unittest
from unittest.mock import patch

from src.data.loaders.mail_archive_x import MailArchiveXLoader
from src.data.models import NormalizedEmail


class TestMailArchiveXLoader(unittest.TestCase):
    def test_load_limits_samples_and_normalizes(self):
        eml_files = ["/tmp/a.eml", "/tmp/b.eml", "/tmp/c.eml"]
        parsed = {
            "sender": "alice@example.com",
            "subject": "Hello",
            "date": None,
            "body": "Body one",
            "recipients": "bob@example.com",
            "error": None,
        }

        with patch("os.path.isdir", return_value=True):
            loader = MailArchiveXLoader("/tmp")

        with patch.object(loader, "_discover_eml_files", return_value=eml_files):
            with patch.object(loader, "_parse_eml_file", return_value=parsed):
                emails = loader.load(num_samples=2)

        self.assertEqual(len(emails), 2)
        self.assertTrue(all(isinstance(e, NormalizedEmail) for e in emails))
        self.assertEqual(emails[0].source, "mail_archive_x")
        self.assertEqual(emails[0].source_id, eml_files[0])
        self.assertIn("Body", emails[0].body)


class TestBulkHeaderDetection(unittest.TestCase):
    """The loader flags bulk mail (List-Unsubscribe / Precedence:bulk) so the
    conservative Pass-1 noise filter can act on it downstream."""

    def _load_one(self, raw: str) -> NormalizedEmail:
        with tempfile.NamedTemporaryFile("w", suffix=".eml", delete=False, encoding="utf-8") as f:
            f.write(raw)
            path = f.name
        try:
            return MailArchiveXLoader(eml_files=[path]).load()[0]
        finally:
            os.unlink(path)

    def test_list_unsubscribe_header_sets_is_bulk(self):
        raw = (
            "From: News <news@example.com>\r\n"
            "Subject: Weekly digest\r\n"
            "List-Unsubscribe: <https://example.com/unsub>\r\n"
            "\r\n"
            "This week's stories.\r\n"
        )
        self.assertTrue(self._load_one(raw).is_bulk)

    def test_precedence_bulk_sets_is_bulk(self):
        raw = (
            "From: News <news@example.com>\r\n"
            "Subject: Notice\r\n"
            "Precedence: bulk\r\n"
            "\r\n"
            "Automated notice.\r\n"
        )
        self.assertTrue(self._load_one(raw).is_bulk)

    def test_plain_human_email_is_not_bulk(self):
        raw = (
            "From: Alice <alice@example.com>\r\n"
            "Subject: Lunch?\r\n"
            "\r\n"
            "Wanna grab lunch tomorrow?\r\n"
        )
        self.assertFalse(self._load_one(raw).is_bulk)


class TestLoaderVerbosity(unittest.TestCase):
    """The loader is chatty by default (progress for batch loads) but can be
    silenced for the per-email hot path, so the threaded Pass-2/calibrate loaders
    don't need to hijack the process-global ``sys.stdout`` (which is not
    thread-safe and corrupts later prints under ``--workers > 1``)."""

    def _eml(self) -> str:
        f = tempfile.NamedTemporaryFile("w", suffix=".eml", delete=False, encoding="utf-8")
        f.write("From: a@example.com\r\nSubject: Hi\r\n\r\nHello body.\r\n")
        f.close()
        return f.name

    def test_quiet_loader_prints_nothing(self):
        import io
        from contextlib import redirect_stdout

        path = self._eml()
        try:
            buf = io.StringIO()
            with redirect_stdout(buf):
                emails = MailArchiveXLoader(eml_files=[path], verbose=False).load()
            self.assertEqual(len(emails), 1)
            self.assertEqual(buf.getvalue(), "")
        finally:
            os.unlink(path)

    def test_verbose_default_still_prints(self):
        import io
        from contextlib import redirect_stdout

        path = self._eml()
        try:
            buf = io.StringIO()
            with redirect_stdout(buf):
                MailArchiveXLoader(eml_files=[path]).load()
            self.assertIn("Loaded 1 emails", buf.getvalue())
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
