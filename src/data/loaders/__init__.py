"""Email loader implementations and interfaces.

The base interface and the local Mail Archive X loader are imported eagerly.
The Enron loader is imported lazily so its optional heavy dependency
(HuggingFace datasets) isn't required just to do local .eml indexing.
"""

from src.data.loaders.base import EmailLoader
from src.data.loaders.mail_archive_x import MailArchiveXLoader

__all__ = [
    "EmailLoader",
    "EnronDatasetLoader",
    "MailArchiveXLoader",
]


def __getattr__(name):
    if name == "EnronDatasetLoader":
        from src.data.loaders.enron import EnronDatasetLoader

        return EnronDatasetLoader
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
