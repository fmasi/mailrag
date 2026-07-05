"""Tests for the bench_models eval tool's pure helper (no LM Studio needed)."""

import importlib
import os
import unittest

# Import strategy mirrors test_gen_thread_summaries: bench_models has top-level
# imports from src.*; import via importlib so a missing dep skips cleanly.
try:
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    _mod = importlib.import_module("scripts.eval.bench_models")
    native_api_base = _mod.native_api_base
    _IMPORT_OK = True
except Exception as _import_err:  # noqa: BLE001
    _IMPORT_OK = False
    _import_err_msg = str(_import_err)


@unittest.skipUnless(
    _IMPORT_OK, f"bench_models not importable: {'' if _IMPORT_OK else _import_err_msg}"
)  # type: ignore[name-defined]
class TestNativeApiBase(unittest.TestCase):
    """Strips the OpenAI-compatible /v1 to reach LM Studio's native /api/v0 host."""

    def test_strips_v1_suffix(self):
        self.assertEqual(native_api_base("http://h:1234/v1"), "http://h:1234")

    def test_strips_v1_with_trailing_slash(self):
        self.assertEqual(native_api_base("http://h:1234/v1/"), "http://h:1234")

    def test_bare_host_unchanged(self):
        self.assertEqual(native_api_base("http://h:1234"), "http://h:1234")

    def test_hostname_ending_in_v1_preserved(self):
        # only a /v1 path segment is stripped, never a hostname that ends in v1
        self.assertEqual(native_api_base("http://hostv1:1234"), "http://hostv1:1234")

    def test_v1_mid_path_preserved(self):
        # /v1 not at the end is left untouched (only a trailing /v1 is stripped)
        self.assertEqual(native_api_base("http://h:1234/v1/chat"), "http://h:1234/v1/chat")


if __name__ == "__main__":
    unittest.main()
