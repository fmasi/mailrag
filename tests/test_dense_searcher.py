"""Tests for the dense-only searcher + NIM reranker + injectable reranker.

The C (NVIDIA-native) query path: dense retrieval over a NIM-embedded collection
(no sparse leg), with the NVIDIA reranking NIM as an injectable node-postprocessor.
Hermetic — QdrantVectorStore/VectorStoreIndex/NVIDIARerank are all mocked.
"""
import unittest
from unittest.mock import MagicMock, patch

from src.query import hybrid


class TestBuildDenseSearcher(unittest.TestCase):
    def test_dense_searcher_has_no_hybrid_or_sparse(self):
        with patch("src.query.hybrid.QdrantVectorStore") as VS, \
             patch("src.query.hybrid.VectorStoreIndex") as IDX:
            hybrid.build_dense_searcher(
                "e5col", embed_model=MagicMock(), client=MagicMock(), dense_top_k=20)
            _, kwargs = VS.call_args
            self.assertEqual(kwargs["dense_vector_name"], "dense")
            self.assertNotIn("sparse_query_fn", kwargs)
            self.assertNotEqual(kwargs.get("enable_hybrid"), True)
            retr = IDX.from_vector_store.return_value.as_retriever.call_args.kwargs
            self.assertEqual(retr["similarity_top_k"], 20)

    def test_dense_searcher_uses_injected_embed_model(self):
        with patch("src.query.hybrid.QdrantVectorStore"), \
             patch("src.query.hybrid.VectorStoreIndex") as IDX:
            em = MagicMock(name="nim_embed")
            hybrid.build_dense_searcher("e5col", embed_model=em, client=MagicMock())
            self.assertIs(IDX.from_vector_store.call_args.kwargs["embed_model"], em)

    def test_dense_searcher_attaches_injected_reranker(self):
        with patch("src.query.hybrid.QdrantVectorStore"), \
             patch("src.query.hybrid.VectorStoreIndex"):
            rr = MagicMock(name="nim_reranker")
            s = hybrid.build_dense_searcher(
                "e5col", embed_model=MagicMock(), client=MagicMock(), reranker=rr)
            self.assertIs(s._reranker, rr)

    def test_dense_searcher_no_reranker_by_default(self):
        with patch("src.query.hybrid.QdrantVectorStore"), \
             patch("src.query.hybrid.VectorStoreIndex"):
            s = hybrid.build_dense_searcher("e5col", embed_model=MagicMock(), client=MagicMock())
            self.assertIsNone(s._reranker)


class TestInjectableRerankerOnHybrid(unittest.TestCase):
    def test_injected_reranker_overrides_flags(self):
        with patch("src.query.hybrid.QdrantVectorStore"), \
             patch("src.query.hybrid.VectorStoreIndex"), \
             patch("src.query.hybrid._make_reranker") as MR:
            rr = MagicMock(name="injected")
            s = hybrid.build_hybrid_searcher(
                "c", client=MagicMock(), embedder=MagicMock(),
                mode="hybrid", rerank=True, reranker=rr)
            self.assertIs(s._reranker, rr)
            MR.assert_not_called()  # injected reranker wins over rerank=True


class TestMakeNimReranker(unittest.TestCase):
    def setUp(self):
        # The NVIDIA rerank connector is an optional `nvidia` extra; this test
        # mocks it but needs the module importable to patch it. Skip cleanly
        # when absent (CI installs without --extras nvidia).
        import pytest
        pytest.importorskip("llama_index.postprocessor.nvidia_rerank")

    def test_builds_nvidia_rerank_with_endpoint_and_model(self):
        with patch("llama_index.postprocessor.nvidia_rerank.NVIDIARerank") as NR:
            r = hybrid.make_nim_reranker(top_n=7, api_key="k")
            NR.assert_called_once()
            kwargs = NR.call_args.kwargs
            self.assertEqual(kwargs["model"], "nvidia/rerank-qa-mistral-4b")
            self.assertEqual(kwargs["base_url"], "https://ai.api.nvidia.com/v1")
            self.assertEqual(kwargs["top_n"], 7)
            self.assertIs(r, NR.return_value)


if __name__ == "__main__":
    unittest.main()
