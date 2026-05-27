"""Enron dataset loader implementation."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from datasets import load_dataset

from src.config.settings import RAGConfig
from src.data.loaders.base import EmailLoader
from src.data.models import NormalizedEmail, normalize_enron_record


class EnronDatasetLoader(EmailLoader):
    """Load emails from the Enron QA dataset on Hugging Face."""

    def load(self, num_samples: Optional[int] = None) -> List[NormalizedEmail]:
        """Load the dataset and normalize each record."""
        print(f"Loading {RAGConfig.DATASET_NAME} dataset...")

        dataset = load_dataset(
            RAGConfig.DATASET_NAME,
            split=RAGConfig.DATASET_SPLIT,
            cache_dir=RAGConfig.get_data_cache_dir(),
        )

        if num_samples:
            # Limit to requested count without exceeding dataset size.
            dataset = dataset.select(range(min(num_samples, len(dataset))))

        emails: List[NormalizedEmail] = []
        for i, record in enumerate(dataset):
            if (i + 1) % 1000 == 0:
                print(f"  Processed {i + 1}/{len(dataset)} emails...")
            emails.append(normalize_enron_record(record, i))

        print(f"Loaded {len(emails)} emails from Enron")
        return emails

    def get_source_info(self) -> Dict[str, Any]:
        """Return Enron source metadata."""
        return {
            "source": "enron",
            "dataset_name": RAGConfig.DATASET_NAME,
            "dataset_split": RAGConfig.DATASET_SPLIT,
            "cache_dir": RAGConfig.get_data_cache_dir(),
        }
