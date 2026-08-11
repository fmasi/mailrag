"""Integration guard for the installed qdrant-client + LlamaIndex adapter pair.

This is deliberately *not* a mocked test. It drives a real in-memory Qdrant
through the real ``QdrantVectorStore`` adapter and asserts that a document can
be indexed and retrieved, so a version skew between the two packages fails here
rather than in production.

That skew is not hypothetical. ``pyproject.toml`` caps ``qdrant-client`` below
1.19 because 1.19 removed ``IDF_EMBEDDING_MODELS`` from
``qdrant_client.qdrant_fastembed`` while ``llama-index-vector-stores-qdrant``
still imports it at module import time — which takes down every dense, hybrid
and MCP code path we have (issue #106). This test is what turns the next such
break into a red build instead of a silent one.

Embeddings come from ``MockEmbedding`` so the test needs no model weights and no
network; the thing under test is the client/adapter seam, not retrieval quality.

Ported out of the deleted ``tests/test_qdrant_storage.py`` when the legacy
StorageManager was retired in #49 — the surrounding StorageManager tests went
with that module, but this one never depended on it.
"""

import unittest

from llama_index.core import Document, Settings, VectorStoreIndex
from llama_index.core.embeddings.mock_embed_model import MockEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient


class TestQdrantAdapterIntegration(unittest.TestCase):
    def setUp(self):
        # Settings.embed_model is process-global; snapshot the private attribute
        # so restoring it cannot trigger the property's lazy-default construction.
        self._original_embed_model = getattr(Settings, "_embed_model", None)
        Settings.embed_model = MockEmbedding(embed_dim=8)

    def tearDown(self):
        Settings._embed_model = self._original_embed_model

    def test_documents_round_trip_through_the_real_adapter(self):
        """Index one document into in-memory Qdrant and retrieve it back."""
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
        # Metadata surviving the round trip is the part that actually exercises
        # the adapter's payload mapping, not just that a vector came back.
        self.assertEqual(results[0].metadata.get("subject"), "Meeting")

    def test_the_adapter_imports_without_a_version_skew(self):
        """Pin the #106 failure mode directly: the import itself must not raise.

        ``QdrantVectorStore`` resolving at module import time is the precise
        thing the qdrant-client cap protects. Asserting it here means the guard
        holds even if the retrieval test above is ever skipped or refactored.
        """
        from llama_index.vector_stores.qdrant import QdrantVectorStore as Adapter

        self.assertTrue(callable(Adapter))


if __name__ == "__main__":
    unittest.main()
