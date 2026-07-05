import unittest
from unittest import mock

from src.ingest.device import pick_device


class TestPickDevice(unittest.TestCase):
    def test_prefers_cuda_then_mps_then_cpu(self):
        with (
            mock.patch("src.ingest.device._has_cuda", return_value=True),
            mock.patch("src.ingest.device._has_mps", return_value=True),
        ):
            self.assertEqual(pick_device(), "cuda")
        with (
            mock.patch("src.ingest.device._has_cuda", return_value=False),
            mock.patch("src.ingest.device._has_mps", return_value=True),
        ):
            self.assertEqual(pick_device(), "mps")
        with (
            mock.patch("src.ingest.device._has_cuda", return_value=False),
            mock.patch("src.ingest.device._has_mps", return_value=False),
        ):
            self.assertEqual(pick_device(), "cpu")

    def test_env_override(self):
        with mock.patch.dict("os.environ", {"RAG_EMBED_DEVICE": "cpu"}):
            self.assertEqual(pick_device(), "cpu")
