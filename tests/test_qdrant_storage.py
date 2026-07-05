"""Unit tests for Qdrant-backed StorageManager."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from llama_index.core import Document, Settings, VectorStoreIndex
from llama_index.core.embeddings.mock_embed_model import MockEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from src.config.settings import RAGConfig


class TestQdrantStorage(unittest.TestCase):
    """Tests for StorageManager Qdrant integration with mocked clients."""

    def setUp(self):
        self._orig_provider = RAGConfig.VECTOR_STORE_PROVIDER
        RAGConfig.VECTOR_STORE_PROVIDER = "qdrant"

    def tearDown(self):
        RAGConfig.VECTOR_STORE_PROVIDER = self._orig_provider

    @patch("src.storage.persist._get_qdrant_client_and_collection")
    def test_index_exists_qdrant_with_vectors(self, mock_get_client):
        """index_exists returns True when Qdrant collection has vectors."""
        mock_client = MagicMock()
        mock_client.collection_exists.return_value = True
        mock_client.get_collection.return_value = SimpleNamespace(points_count=25)
        mock_get_client.return_value = (mock_client, "test-col")

        from src.storage.persist import StorageManager

        result = StorageManager.index_exists()

        self.assertTrue(result)

    @patch("src.storage.persist._get_qdrant_client_and_collection")
    def test_index_exists_qdrant_missing_collection(self, mock_get_client):
        """index_exists returns False when Qdrant collection does not exist."""
        mock_client = MagicMock()
        mock_client.collection_exists.return_value = False
        mock_get_client.return_value = (mock_client, "test-col")

        from src.storage.persist import StorageManager

        result = StorageManager.index_exists()

        self.assertFalse(result)

    @patch("src.storage.persist._get_qdrant_client_and_collection")
    def test_index_exists_qdrant_zero_points(self, mock_get_client):
        """index_exists returns False when Qdrant collection is empty."""
        mock_client = MagicMock()
        mock_client.collection_exists.return_value = True
        mock_client.get_collection.return_value = SimpleNamespace(points_count=0)
        mock_get_client.return_value = (mock_client, "test-col")

        from src.storage.persist import StorageManager

        result = StorageManager.index_exists()

        self.assertFalse(result)

    @patch("src.storage.persist._get_qdrant_vector_store")
    @patch("src.storage.persist.VectorStoreIndex")
    def test_load_index_qdrant(self, mock_vsi, mock_get_vs):
        """load_index uses VectorStoreIndex.from_vector_store with Qdrant."""
        mock_vs = MagicMock()
        mock_get_vs.return_value = mock_vs
        mock_index = MagicMock()
        mock_vsi.from_vector_store.return_value = mock_index

        from src.storage.persist import StorageManager

        result = StorageManager.load_index()

        mock_get_vs.assert_called_once()
        mock_vsi.from_vector_store.assert_called_once_with(mock_vs)
        self.assertEqual(result, mock_index)

    @patch("src.storage.persist._get_qdrant_vector_store")
    @patch("src.storage.persist._ingest_to_qdrant")
    @patch("src.storage.persist.VectorStoreIndex")
    def test_create_and_save_index_qdrant(self, mock_vsi, mock_ingest, mock_get_vs):
        """create_and_save_index runs the IngestionPipeline and returns a Qdrant-backed index."""
        mock_vs = MagicMock()
        mock_get_vs.return_value = mock_vs
        mock_index = MagicMock()
        mock_vsi.from_vector_store.return_value = mock_index

        from src.storage.persist import StorageManager

        docs = [MagicMock()]
        result = StorageManager.create_and_save_index(docs, verbose=False)

        mock_get_vs.assert_called_once()
        mock_ingest.assert_called_once_with(docs, mock_vs, False)
        # a lightweight index handle is built via from_vector_store, not from_documents
        mock_vsi.from_vector_store.assert_called_once_with(mock_vs)
        self.assertEqual(result, mock_index)

    @patch("src.storage.persist._get_qdrant_vector_store")
    @patch("src.storage.persist._ingest_to_qdrant")
    @patch("src.storage.persist.VectorStoreIndex")
    def test_create_and_save_index_qdrant_return_stats(
        self,
        mock_vsi,
        mock_ingest,
        mock_get_vs,
    ):
        """create_and_save_index can return ingest timing stats when requested."""
        mock_vs = MagicMock()
        mock_get_vs.return_value = mock_vs
        mock_index = MagicMock()
        mock_vsi.from_vector_store.return_value = mock_index
        mock_stats = {
            "embed_secs": 12.3,
            "upload_secs": 0.7,
            "combined_secs": 13.0,
        }
        mock_ingest.return_value = mock_stats

        from src.storage.persist import StorageManager

        docs = [MagicMock()]
        index, stats = StorageManager.create_and_save_index(
            docs,
            verbose=False,
            return_stats=True,
        )

        mock_get_vs.assert_called_once()
        mock_ingest.assert_called_once_with(docs, mock_vs, False)
        mock_vsi.from_vector_store.assert_called_once_with(mock_vs)
        self.assertEqual(index, mock_index)
        self.assertEqual(stats, mock_stats)

    def test_in_memory_qdrant_retrieval_stack(self):
        """Real retrieval works against in-memory Qdrant with the installed client/adapter pair."""
        original_embed_model = getattr(Settings, "_embed_model", None)
        try:
            Settings.embed_model = MockEmbedding(embed_dim=8)
            client = QdrantClient(location=":memory:")
            vector_store = QdrantVectorStore(
                client=client,
                collection_name="qdrant-regression-test",
            )
            index = VectorStoreIndex.from_documents(
                [
                    Document(
                        text="meeting schedule for monday morning",
                        metadata={"subject": "Meeting"},
                    )
                ],
                vector_store=vector_store,
            )

            results = index.as_retriever(similarity_top_k=1).retrieve("meeting schedule")

            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].metadata.get("subject"), "Meeting")
        finally:
            Settings._embed_model = original_embed_model


if __name__ == "__main__":
    unittest.main()
