"""Unit tests for the unified load_emails function."""

import unittest
from unittest.mock import MagicMock, patch

import pytest
from llama_index.core import Document

from src.data.loader import load_emails, load_enron_dataset


class TestLoadEmails(unittest.TestCase):
    def test_load_emails_enron_source(self):
        """Test loading from Enron source."""
        mock_normalized_email = MagicMock()
        mock_normalized_email.source = "enron"
        mock_document = Document(text="Body", metadata={"sender": "a@b.com"})
        mock_normalized_email.to_document.return_value = mock_document

        with patch("src.data.loader.EnronDatasetLoader") as MockLoader:
            mock_loader = MagicMock()
            mock_loader.load.return_value = [mock_normalized_email]
            MockLoader.return_value = mock_loader

            docs = load_emails(source="enron", num_samples=10)

        self.assertEqual(len(docs), 1)
        self.assertIsInstance(docs[0], Document)
        MockLoader.assert_called_once()
        mock_loader.load.assert_called_once_with(num_samples=10)

    def test_load_emails_mail_archive_x_source(self):
        """Test loading from Mail Archive X source."""
        mock_normalized_email = MagicMock()
        mock_normalized_email.source = "mail_archive_x"
        mock_document = Document(text="Body", metadata={"sender": "a@b.com"})
        mock_normalized_email.to_document.return_value = mock_document

        with patch("src.data.loader.MailArchiveXLoader") as MockLoader:
            mock_loader = MagicMock()
            mock_loader.load.return_value = [mock_normalized_email]
            MockLoader.return_value = mock_loader

            docs = load_emails(
                source="mail_archive_x",
                backup_dir="/path/to/backup",
                num_samples=10,
            )

        self.assertEqual(len(docs), 1)
        self.assertIsInstance(docs[0], Document)
        MockLoader.assert_called_once_with("/path/to/backup")
        mock_loader.load.assert_called_once_with(num_samples=10)

    def test_load_emails_azure_blob_source(self):
        """Dispatch into the (lazily-imported) Azure loader.

        azure-storage-blob is optional, so skip on a minimal install (#44).
        Patching the real class location works because load_emails resolves
        ``from src.data.loaders import AzureBlobEmailLoader`` lazily at call time.
        A module-level importorskip can't be used here (the rest of this file's
        enron / mail_archive_x tests must still run without Azure), and an
        unguarded ``create=True`` patch would re-trigger the Azure import via the
        loaders package ``__getattr__`` — so the guard is per-test.
        """
        pytest.importorskip("azure.storage.blob")
        mock_normalized_email = MagicMock()
        mock_normalized_email.source = "azure_blob"
        mock_document = Document(text="Body", metadata={"sender": "a@b.com"})
        mock_normalized_email.to_document.return_value = mock_document

        with patch("src.data.loaders.azure_blob.AzureBlobEmailLoader") as MockLoader:
            mock_loader = MagicMock()
            mock_loader.load.return_value = [mock_normalized_email]
            MockLoader.return_value = mock_loader

            docs = load_emails(source="azure_blob")

        self.assertEqual(len(docs), 1)
        self.assertIsInstance(docs[0], Document)
        MockLoader.assert_called_once()
        mock_loader.load.assert_called_once_with(num_samples=None)

    def test_load_emails_mail_archive_x_missing_backup_dir(self):
        """Test that backup_dir is required for Mail Archive X."""
        with self.assertRaises(ValueError) as context:
            load_emails(source="mail_archive_x", num_samples=10)
        self.assertIn("backup_dir required", str(context.exception))

    def test_load_emails_invalid_source(self):
        """Test that invalid source raises error."""
        with self.assertRaises(ValueError) as context:
            load_emails(source="invalid_source")
        self.assertIn("Unknown source", str(context.exception))

    def test_load_enron_dataset_backward_compat(self):
        """Test backward-compatible load_enron_dataset wrapper."""
        mock_normalized_email = MagicMock()
        mock_normalized_email.source = "enron"
        mock_document = Document(text="Body", metadata={"sender": "a@b.com"})
        mock_normalized_email.to_document.return_value = mock_document

        with patch("src.data.loader.EnronDatasetLoader") as MockLoader:
            mock_loader = MagicMock()
            mock_loader.load.return_value = [mock_normalized_email]
            MockLoader.return_value = mock_loader

            docs = load_enron_dataset(num_samples=5)

        self.assertEqual(len(docs), 1)
        MockLoader.assert_called_once()
        mock_loader.load.assert_called_once_with(num_samples=5)


if __name__ == "__main__":
    unittest.main()
