"""Azure Blob Storage email loader implementation."""

from __future__ import annotations

import os
import tempfile
from typing import Any, Dict, List, Optional

from azure.storage.blob import BlobServiceClient

from src.data.loaders.base import EmailLoader
from src.data.loaders.mail_archive_x import MailArchiveXLoader
from src.data.models import NormalizedEmail


class AzureBlobEmailLoader(EmailLoader):
    """Load .eml files from Azure Blob Storage.

    Downloads blobs to a temporary directory and delegates parsing
    to the existing ``MailArchiveXLoader``.
    """

    def __init__(
        self,
        connection_string: str | None = None,
        container_name: str | None = None,
        blob_prefix: str | None = None,
    ):
        """
        Initialize the Azure Blob email loader.

        Args:
            connection_string: Azure Storage connection string.
                Falls back to ``AZURE_STORAGE_CONNECTION_STRING`` env var.
            container_name: Blob container name.
                Falls back to ``AZURE_BLOB_CONTAINER`` env var (default ``eml-archive``).
            blob_prefix: Optional prefix to scope blob listing.
                Falls back to ``AZURE_BLOB_PREFIX`` env var.
        """
        self.connection_string = (
            connection_string or os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
        )
        if not self.connection_string:
            raise ValueError(
                "Azure connection string is required. Pass connection_string "
                "or set AZURE_STORAGE_CONNECTION_STRING environment variable."
            )
        self.container_name = (
            container_name
            or os.environ.get("AZURE_BLOB_CONTAINER", "eml-archive")
        )
        self.blob_prefix = (
            blob_prefix or os.environ.get("AZURE_BLOB_PREFIX", "")
        )

    def load(self, num_samples: Optional[int] = None) -> List[NormalizedEmail]:
        """Download .eml blobs and parse via MailArchiveXLoader."""
        service_client = BlobServiceClient.from_connection_string(
            self.connection_string
        )
        container_client = service_client.get_container_client(
            self.container_name
        )

        # List all .eml blobs (filtered by prefix)
        blobs = [
            b
            for b in container_client.list_blobs(
                name_starts_with=self.blob_prefix or None
            )
            if b.name.endswith(".eml")
        ]

        if num_samples:
            blobs = blobs[:num_samples]

        print(
            f"Downloading {len(blobs)} .eml files from "
            f"Azure Blob '{self.container_name}'..."
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            for blob in blobs:
                blob_client = container_client.get_blob_client(blob.name)
                local_path = os.path.join(temp_dir, blob.name)
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                with open(local_path, "wb") as f:
                    f.write(blob_client.download_blob().readall())

            loader = MailArchiveXLoader(temp_dir)
            return loader.load()

    def get_source_info(self) -> Dict[str, Any]:
        """Return Azure Blob source metadata."""
        return {
            "source": "azure_blob",
            "container": self.container_name,
            "prefix": self.blob_prefix,
        }
