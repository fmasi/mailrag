"""Unit tests for the Mail Archive X loader."""

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


if __name__ == "__main__":
    unittest.main()
