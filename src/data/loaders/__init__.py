"""Email loader implementations and interfaces.

The base interface and the local Mail Archive X loader are imported eagerly.
The source-specific loaders (Azure Blob, Enron) are imported lazily so their
optional heavy dependencies (azure-storage-blob, HuggingFace datasets) aren't
required just to do local .eml indexing.
"""

from src.data.loaders.base import EmailLoader
from src.data.loaders.mail_archive_x import MailArchiveXLoader

__all__ = [
    "EmailLoader",
    "EnronDatasetLoader",
    "MailArchiveXLoader",
    "AzureBlobEmailLoader",
]


def __getattr__(name):
    if name == "AzureBlobEmailLoader":
        from src.data.loaders.azure_blob import AzureBlobEmailLoader

        return AzureBlobEmailLoader
    if name == "EnronDatasetLoader":
        from src.data.loaders.enron import EnronDatasetLoader

        return EnronDatasetLoader
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
