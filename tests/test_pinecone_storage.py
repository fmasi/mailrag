"""Unit tests for Pinecone-backed StorageManager."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.config.settings import RAGConfig


class TestPineconeStorage(unittest.TestCase):
    """Tests for StorageManager Pinecone integration with mocked client."""

    def setUp(self):
        """Save original provider and force pinecone for most tests."""
        self._orig_provider = RAGConfig.VECTOR_STORE_PROVIDER
        RAGConfig.VECTOR_STORE_PROVIDER = "pinecone"

    def tearDown(self):
        RAGConfig.VECTOR_STORE_PROVIDER = self._orig_provider

    @patch.dict(
        "os.environ",
        {"PINECONE_API_KEY": "test-key", "PINECONE_INDEX_NAME": "test-idx"},
    )
    def test_index_exists_pinecone_with_vectors(self):
        """index_exists returns True when Pinecone index has vectors."""
        mock_pc_instance = MagicMock()
        mock_index = MagicMock()
        mock_index.describe_index_stats.return_value = SimpleNamespace(total_vector_count=500)
        mock_pc_instance.Index.return_value = mock_index

        mock_pinecone_cls = MagicMock(return_value=mock_pc_instance)

        with patch.dict("sys.modules", {"pinecone": MagicMock(Pinecone=mock_pinecone_cls)}):
            from src.storage.persist import StorageManager

            result = StorageManager.index_exists()

        self.assertTrue(result)

    @patch.dict(
        "os.environ",
        {"PINECONE_API_KEY": "test-key", "PINECONE_INDEX_NAME": "test-idx"},
    )
    def test_index_exists_pinecone_empty(self):
        """index_exists returns False when Pinecone index is empty."""
        mock_pc_instance = MagicMock()
        mock_index = MagicMock()
        mock_index.describe_index_stats.return_value = SimpleNamespace(total_vector_count=0)
        mock_pc_instance.Index.return_value = mock_index

        mock_pinecone_cls = MagicMock(return_value=mock_pc_instance)

        with patch.dict("sys.modules", {"pinecone": MagicMock(Pinecone=mock_pinecone_cls)}):
            from src.storage.persist import StorageManager

            result = StorageManager.index_exists()

        self.assertFalse(result)

    @patch.dict(
        "os.environ",
        {"PINECONE_API_KEY": "test-key", "PINECONE_INDEX_NAME": "test-idx"},
    )
    @patch("src.storage.persist._get_pinecone_vector_store")
    @patch("src.storage.persist.VectorStoreIndex")
    def test_load_index_pinecone(self, MockVSI, mock_get_vs):
        """load_index uses VectorStoreIndex.from_vector_store with Pinecone store."""
        mock_vs = MagicMock()
        mock_get_vs.return_value = mock_vs
        mock_index = MagicMock()
        MockVSI.from_vector_store.return_value = mock_index

        from src.storage.persist import StorageManager

        result = StorageManager.load_index()

        mock_get_vs.assert_called_once()
        MockVSI.from_vector_store.assert_called_once_with(mock_vs)
        self.assertEqual(result, mock_index)

    @patch.dict(
        "os.environ",
        {"PINECONE_API_KEY": "test-key", "PINECONE_INDEX_NAME": "test-idx"},
    )
    @patch("src.storage.persist._get_pinecone_vector_store")
    @patch("src.storage.persist.VectorStoreIndex")
    @patch("src.storage.persist.StorageContext")
    def test_create_and_save_index_pinecone(self, MockSC, MockVSI, mock_get_vs):
        """create_and_save_index uses Pinecone vector store and StorageContext."""
        mock_vs = MagicMock()
        mock_get_vs.return_value = mock_vs
        mock_sc = MagicMock()
        MockSC.from_defaults.return_value = mock_sc
        mock_index = MagicMock()
        MockVSI.from_documents.return_value = mock_index

        from src.storage.persist import StorageManager

        docs = [MagicMock()]
        result = StorageManager.create_and_save_index(docs, verbose=False)

        mock_get_vs.assert_called_once()
        MockSC.from_defaults.assert_called_once_with(vector_store=mock_vs)
        MockVSI.from_documents.assert_called_once()
        self.assertEqual(result, mock_index)

    def test_simple_provider_unchanged(self):
        """When VECTOR_STORE_PROVIDER=simple, the original file-check path is used."""
        RAGConfig.VECTOR_STORE_PROVIDER = "simple"

        from src.storage.persist import StorageManager

        # With no storage dir on disk, index_exists should return False
        with patch.object(RAGConfig, "get_storage_dir", return_value="/tmp/nonexistent_dir_xyz"):
            result = StorageManager.index_exists()

        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
