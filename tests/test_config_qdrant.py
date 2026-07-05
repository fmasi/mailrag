"""Unit tests for the single Qdrant client seam (src.config.qdrant.get_qdrant_client).

This factory is the one place that turns explicit args + the QDRANT_* environment
into a QdrantClient, so the three former call sites (query/hybrid, storage/persist,
ingest/hybrid_qdrant) cannot drift apart. Contract verified here:

  url        explicit arg > $QDRANT_URL > ValueError; whitespace stripped.
  api_key    explicit arg > $QDRANT_API_KEY > None (empty string treated as None).
  prefer_grpc explicit bool > $QDRANT_PREFER_GRPC (truthy set) > False.
"""

import os
import unittest
from unittest.mock import patch

from src.config.qdrant import get_qdrant_client


class TestGetQdrantClient(unittest.TestCase):
    def test_explicit_url_overrides_env(self):
        with (
            patch("qdrant_client.QdrantClient") as QClient,
            patch.dict(os.environ, {"QDRANT_URL": "http://host.docker.internal:6333"}),
        ):
            get_qdrant_client("http://localhost:6333")
            self.assertEqual(QClient.call_args.kwargs["url"], "http://localhost:6333")

    def test_falls_back_to_env_when_no_arg(self):
        with (
            patch("qdrant_client.QdrantClient") as QClient,
            patch.dict(os.environ, {"QDRANT_URL": "http://env-host:6333"}),
        ):
            get_qdrant_client()
            self.assertEqual(QClient.call_args.kwargs["url"], "http://env-host:6333")

    def test_missing_url_raises(self):
        with patch("qdrant_client.QdrantClient"), patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "QDRANT_URL"):
                get_qdrant_client()

    def test_url_is_stripped(self):
        with patch("qdrant_client.QdrantClient") as QClient, patch.dict(os.environ, {}, clear=True):
            get_qdrant_client("  http://localhost:6333  ")
            self.assertEqual(QClient.call_args.kwargs["url"], "http://localhost:6333")

    def test_api_key_from_env(self):
        with (
            patch("qdrant_client.QdrantClient") as QClient,
            patch.dict(os.environ, {"QDRANT_URL": "http://h:6333", "QDRANT_API_KEY": "secret"}),
        ):
            get_qdrant_client()
            self.assertEqual(QClient.call_args.kwargs["api_key"], "secret")

    def test_empty_api_key_becomes_none(self):
        with (
            patch("qdrant_client.QdrantClient") as QClient,
            patch.dict(os.environ, {"QDRANT_URL": "http://h:6333", "QDRANT_API_KEY": ""}),
        ):
            get_qdrant_client()
            self.assertIsNone(QClient.call_args.kwargs["api_key"])

    def test_explicit_api_key_overrides_env(self):
        with (
            patch("qdrant_client.QdrantClient") as QClient,
            patch.dict(os.environ, {"QDRANT_URL": "http://h:6333", "QDRANT_API_KEY": "env-key"}),
        ):
            get_qdrant_client(api_key="arg-key")
            self.assertEqual(QClient.call_args.kwargs["api_key"], "arg-key")

    def test_prefer_grpc_parsed_from_env_truthy(self):
        for raw in ("1", "true", "YES", "on"):
            with self.subTest(raw=raw):
                with (
                    patch("qdrant_client.QdrantClient") as QClient,
                    patch.dict(
                        os.environ, {"QDRANT_URL": "http://h:6333", "QDRANT_PREFER_GRPC": raw}
                    ),
                ):
                    get_qdrant_client()
                    self.assertTrue(QClient.call_args.kwargs["prefer_grpc"])

    def test_prefer_grpc_defaults_false_when_unset(self):
        with (
            patch("qdrant_client.QdrantClient") as QClient,
            patch.dict(os.environ, {"QDRANT_URL": "http://h:6333"}, clear=True),
        ):
            get_qdrant_client()
            self.assertFalse(QClient.call_args.kwargs["prefer_grpc"])

    def test_explicit_prefer_grpc_overrides_env(self):
        with (
            patch("qdrant_client.QdrantClient") as QClient,
            patch.dict(os.environ, {"QDRANT_URL": "http://h:6333", "QDRANT_PREFER_GRPC": "true"}),
        ):
            get_qdrant_client(prefer_grpc=False)
            self.assertFalse(QClient.call_args.kwargs["prefer_grpc"])


if __name__ == "__main__":
    unittest.main()
