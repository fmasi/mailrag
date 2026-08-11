"""Unit tests for the Azure Blob Storage email loader."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

# Azure SDK is optional; skip this module (don't error at collection) when it's
# not installed, so `pytest tests/` passes out of the box on a minimal env (#44).
pytest.importorskip("azure.storage.blob")

from src.data.loaders.azure_blob import AzureBlobEmailLoader
from src.data.models import NormalizedEmail


def _make_blob(name: str) -> SimpleNamespace:
    """Create a fake blob object with a ``.name`` attribute."""
    return SimpleNamespace(name=name)


class TestAzureBlobEmailLoader(unittest.TestCase):
    """Tests for AzureBlobEmailLoader with mocked Azure SDK."""

    @patch("src.data.loaders.azure_blob.MailArchiveXLoader")
    @patch("src.data.loaders.azure_blob.BlobServiceClient", create=True)
    def test_load_downloads_and_delegates(self, MockBlobService, MockMailLoader):
        """Blobs are downloaded to a temp dir and MailArchiveXLoader is invoked."""
        # Set up mock blobs
        blobs = [_make_blob("inbox/msg1.eml"), _make_blob("inbox/msg2.eml")]

        mock_container = MagicMock()
        mock_container.list_blobs.return_value = blobs

        mock_blob_client = MagicMock()
        mock_blob_client.download_blob.return_value.readall.return_value = (
            b"From: a@b.com\nSubject: hi\n\nBody"
        )
        mock_container.get_blob_client.return_value = mock_blob_client

        MockBlobService.from_connection_string.return_value.get_container_client.return_value = (
            mock_container
        )

        expected = [
            NormalizedEmail(
                sender="a@b.com",
                subject="hi",
                date=None,
                body="Body",
                source="mail_archive_x",
                source_id="f",
            )
        ]
        MockMailLoader.return_value.load.return_value = expected

        loader = AzureBlobEmailLoader(connection_string="conn", container_name="c", blob_prefix="")

        # Patch BlobServiceClient at the module where it is imported
        with patch("src.data.loaders.azure_blob.BlobServiceClient", MockBlobService):
            result = loader.load()

        self.assertEqual(result, expected)
        MockMailLoader.return_value.load.assert_called_once()
        # Two blobs should have been downloaded
        self.assertEqual(mock_container.get_blob_client.call_count, 2)

    @patch("src.data.loaders.azure_blob.MailArchiveXLoader")
    @patch("src.data.loaders.azure_blob.BlobServiceClient", create=True)
    def test_num_samples_limits_downloads(self, MockBlobService, MockMailLoader):
        """Only num_samples blobs should be downloaded."""
        blobs = [_make_blob(f"msg{i}.eml") for i in range(10)]

        mock_container = MagicMock()
        mock_container.list_blobs.return_value = blobs
        mock_blob_client = MagicMock()
        mock_blob_client.download_blob.return_value.readall.return_value = b"data"
        mock_container.get_blob_client.return_value = mock_blob_client

        MockBlobService.from_connection_string.return_value.get_container_client.return_value = (
            mock_container
        )
        MockMailLoader.return_value.load.return_value = []

        loader = AzureBlobEmailLoader(connection_string="conn", container_name="c")
        with patch("src.data.loaders.azure_blob.BlobServiceClient", MockBlobService):
            loader.load(num_samples=3)

        self.assertEqual(mock_container.get_blob_client.call_count, 3)

    def test_get_source_info(self):
        """get_source_info returns correct metadata dict."""
        with patch.dict("os.environ", {"AZURE_STORAGE_CONNECTION_STRING": "x"}):
            loader = AzureBlobEmailLoader(
                connection_string="x", container_name="my-cont", blob_prefix="pf"
            )

        info = loader.get_source_info()
        self.assertEqual(info["source"], "azure_blob")
        self.assertEqual(info["container"], "my-cont")
        self.assertEqual(info["prefix"], "pf")

    def test_env_var_fallback(self):
        """Constructor reads from env vars when no args are given."""
        env = {
            "AZURE_STORAGE_CONNECTION_STRING": "from-env",
            "AZURE_BLOB_CONTAINER": "env-container",
            "AZURE_BLOB_PREFIX": "env-prefix",
        }
        with patch.dict("os.environ", env):
            loader = AzureBlobEmailLoader()

        self.assertEqual(loader.connection_string, "from-env")
        self.assertEqual(loader.container_name, "env-container")
        self.assertEqual(loader.blob_prefix, "env-prefix")

    def test_missing_connection_string_is_rejected(self):
        """No argument and no env var must fail in the constructor.

        The connection string is bound only after this guard, so every method
        can rely on it being a real string rather than None.
        """
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(ValueError) as ctx:
                AzureBlobEmailLoader()
        self.assertIn("Azure connection string is required", str(ctx.exception))

    @patch("src.data.loaders.azure_blob.MailArchiveXLoader")
    @patch("src.data.loaders.azure_blob.BlobServiceClient", create=True)
    def test_blob_prefix_filtering(self, MockBlobService, MockMailLoader):
        """list_blobs receives the configured prefix."""
        mock_container = MagicMock()
        mock_container.list_blobs.return_value = []
        MockBlobService.from_connection_string.return_value.get_container_client.return_value = (
            mock_container
        )
        MockMailLoader.return_value.load.return_value = []

        loader = AzureBlobEmailLoader(
            connection_string="conn", container_name="c", blob_prefix="Inbox/"
        )
        with patch("src.data.loaders.azure_blob.BlobServiceClient", MockBlobService):
            loader.load()

        mock_container.list_blobs.assert_called_once_with(name_starts_with="Inbox/")


if __name__ == "__main__":
    unittest.main()
