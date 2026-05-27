"""Unit tests for the Enron dataset loader."""

import unittest
from unittest.mock import patch

from src.data.loaders.enron import EnronDatasetLoader
from src.data.models import NormalizedEmail


class DummyDataset:
    """Minimal dataset stub to mimic Hugging Face dataset behavior."""

    def __init__(self, records):
        self._records = records

    def __len__(self):
        return len(self._records)

    def __iter__(self):
        return iter(self._records)

    def select(self, indices):
        return DummyDataset([self._records[i] for i in indices])


class TestEnronDatasetLoader(unittest.TestCase):
    def test_load_limits_samples_and_normalizes(self):
        records = [
            {
                "email": "From: alice@example.com\nSubject: Hello\nDate: Thu, 01 Jan 1970 00:00:00 +0000\n\nBody one",
            },
            {
                "email": "From: bob@example.com\nSubject: Hi\nDate: Thu, 01 Jan 1970 00:00:00 +0000\n\nBody two",
            },
            {
                "email": "From: carol@example.com\nSubject: Hey\nDate: Thu, 01 Jan 1970 00:00:00 +0000\n\nBody three",
            },
        ]
        dummy_dataset = DummyDataset(records)

        with patch("src.data.loaders.enron.load_dataset", return_value=dummy_dataset):
            loader = EnronDatasetLoader()
            emails = loader.load(num_samples=2)

        self.assertEqual(len(emails), 2)
        self.assertTrue(all(isinstance(e, NormalizedEmail) for e in emails))
        self.assertEqual(emails[0].source, "enron")
        self.assertEqual(emails[0].source_id, "enron_0")
        self.assertIn("Body", emails[0].body)


if __name__ == "__main__":
    unittest.main()
