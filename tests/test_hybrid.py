"""Tests for build_hybrid_searcher wiring + opt-in rerank."""
import unittest
from unittest.mock import MagicMock, patch

from src.query import hybrid


class TestBuildHybridSearcher(unittest.TestCase):
    def _patches(self):
        return (
            patch("src.query.hybrid.QdrantVectorStore"),
            patch("src.query.hybrid.VectorStoreIndex"),
            patch("src.query.hybrid._make_reranker"),
        )

    def test_hybrid_mode_wires_qdrant_and_no_reranker(self):
        p_vs, p_idx, p_rr = self._patches()
        with p_vs as VS, p_idx as IDX, p_rr as RR:
            client, embedder = MagicMock(), MagicMock()
            searcher = hybrid.build_hybrid_searcher(
                "work-rag", client=client, embedder=embedder,
                mode="hybrid", rerank=False, dense_top_k=20, sparse_top_k=20,
            )
            _, kwargs = VS.call_args
            self.assertTrue(kwargs["enable_hybrid"])
            self.assertEqual(kwargs["dense_vector_name"], "dense")
            self.assertEqual(kwargs["sparse_vector_name"], "sparse")
            self.assertTrue(callable(kwargs["sparse_query_fn"]))
            # sparse_doc_fn must also be supplied, else QdrantVectorStore builds a
            # fastembed/SPLADE default doc encoder in its constructor.
            self.assertTrue(callable(kwargs["sparse_doc_fn"]))
            self.assertIs(kwargs["hybrid_fusion_fn"], hybrid.reciprocal_rank_fusion)
            retr_kwargs = IDX.from_vector_store.return_value.as_retriever.call_args.kwargs
            self.assertEqual(retr_kwargs["vector_store_query_mode"], "hybrid")
            self.assertEqual(retr_kwargs["similarity_top_k"], 20)
            self.assertEqual(retr_kwargs["sparse_top_k"], 20)
            RR.assert_not_called()
            self.assertIsNone(searcher._reranker)

    def test_dense_mode_uses_default_query_mode(self):
        p_vs, p_idx, p_rr = self._patches()
        with p_vs, p_idx as IDX, p_rr:
            hybrid.build_hybrid_searcher(
                "work-rag", client=MagicMock(), embedder=MagicMock(), mode="dense",
            )
            retr_kwargs = IDX.from_vector_store.return_value.as_retriever.call_args.kwargs
            self.assertEqual(retr_kwargs["vector_store_query_mode"], "default")

    def test_sparse_mode_uses_sparse_query_mode(self):
        p_vs, p_idx, p_rr = self._patches()
        with p_vs, p_idx as IDX, p_rr:
            hybrid.build_hybrid_searcher(
                "work-rag", client=MagicMock(), embedder=MagicMock(), mode="sparse",
            )
            retr_kwargs = IDX.from_vector_store.return_value.as_retriever.call_args.kwargs
            self.assertEqual(retr_kwargs["vector_store_query_mode"], "sparse")

    def test_custom_fusion_fn_is_passed_to_vector_store(self):
        p_vs, p_idx, p_rr = self._patches()
        with p_vs as VS, p_idx, p_rr:
            sentinel = object()
            hybrid.build_hybrid_searcher(
                "work-rag", client=MagicMock(), embedder=MagicMock(),
                mode="hybrid", fusion_fn=sentinel,
            )
            _, kwargs = VS.call_args
            self.assertIs(kwargs["hybrid_fusion_fn"], sentinel)

    def test_default_fusion_fn_is_reciprocal_rank_fusion(self):
        p_vs, p_idx, p_rr = self._patches()
        with p_vs as VS, p_idx, p_rr:
            hybrid.build_hybrid_searcher(
                "work-rag", client=MagicMock(), embedder=MagicMock(), mode="hybrid",
            )
            _, kwargs = VS.call_args
            self.assertIs(kwargs["hybrid_fusion_fn"], hybrid.reciprocal_rank_fusion)

    def test_rerank_true_attaches_reranker(self):
        p_vs, p_idx, p_rr = self._patches()
        with p_vs, p_idx, p_rr as RR:
            RR.return_value = MagicMock(name="reranker")
            searcher = hybrid.build_hybrid_searcher(
                "work-rag", client=MagicMock(), embedder=MagicMock(),
                mode="hybrid", rerank=True, top_n=5,
            )
            RR.assert_called_once_with(top_n=5)
            self.assertIs(searcher._reranker, RR.return_value)

    def test_rerank_with_summary_attaches_summary_reranker(self):
        with patch("src.query.hybrid.QdrantVectorStore"), \
             patch("src.query.hybrid.VectorStoreIndex"), \
             patch("src.query.hybrid.make_summary_reranker") as MSR, \
             patch("src.query.hybrid._make_reranker") as MR:
            MSR.return_value = MagicMock(name="sumrr")
            searcher = hybrid.build_hybrid_searcher(
                "work-rag", client=MagicMock(), embedder=MagicMock(),
                rerank_with_summary=True, top_n=10,
            )
            MSR.assert_called_once_with(top_n=10)
            MR.assert_not_called()
            self.assertIs(searcher._reranker, MSR.return_value)

    def test_builds_client_and_embedder_when_not_injected(self):
        with patch("src.query.hybrid.QdrantVectorStore"), \
             patch("src.query.hybrid.VectorStoreIndex"), \
             patch("src.query.hybrid._make_reranker"), \
             patch("src.query.hybrid._qdrant_client") as QC, \
             patch("src.ingest.embedder.BgeM3Embedder") as EMB:
            hybrid.build_hybrid_searcher("work-rag", mode="hybrid")
            QC.assert_called_once()
            EMB.assert_called_once()


class TestHybridSearcherSearch(unittest.TestCase):
    def test_search_applies_reranker_when_present(self):
        retriever = MagicMock()
        retriever.retrieve.return_value = ["n1", "n2", "n3"]
        reranker = MagicMock()
        reranker.postprocess_nodes.return_value = ["n2"]
        searcher = hybrid.HybridSearcher(retriever, reranker)
        out = searcher.search("q")
        retriever.retrieve.assert_called_once_with("q")
        reranker.postprocess_nodes.assert_called_once_with(["n1", "n2", "n3"], query_str="q")
        self.assertEqual(out, ["n2"])

    def test_search_without_reranker_returns_retrieved(self):
        retriever = MagicMock()
        retriever.retrieve.return_value = ["n1", "n2"]
        searcher = hybrid.HybridSearcher(retriever, None)
        self.assertEqual(searcher.search("q"), ["n1", "n2"])


class TestThreadExpansion(unittest.TestCase):
    def test_search_threads_calls_assemble(self):
        retriever = MagicMock()
        retriever.retrieve.return_value = ["n1"]
        client = MagicMock()
        searcher = hybrid.HybridSearcher(
            retriever, reranker=None, client=client, collection="work-rag",
        )
        with patch("src.query.hybrid.assemble_threads") as AT:
            AT.return_value = ["ctx1"]
            out = searcher.search_threads("q")
            AT.assert_called_once_with(["n1"], client, "work-rag")
            self.assertEqual(out, ["ctx1"])

    def test_search_threads_requires_client(self):
        searcher = hybrid.HybridSearcher(MagicMock(), reranker=None)
        with self.assertRaises(ValueError):
            searcher.search_threads("q")

    def test_build_searcher_passes_client_and_collection(self):
        with patch("src.query.hybrid.QdrantVectorStore"), \
             patch("src.query.hybrid.VectorStoreIndex"), \
             patch("src.query.hybrid._make_reranker"):
            client = MagicMock()
            searcher = hybrid.build_hybrid_searcher(
                "work-rag", client=client, embedder=MagicMock(), mode="hybrid",
            )
            self.assertIs(searcher._client, client)
            self.assertEqual(searcher._collection, "work-rag")


if __name__ == "__main__":
    unittest.main()
