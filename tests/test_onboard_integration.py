import os
import unittest

import pytest

from src.onboard import run_onboard

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures", "eml")


@pytest.mark.integration
class TestOnboardEndToEnd(unittest.TestCase):
    """Real Qdrant + LLM + bge-m3. Run with:
       conda run -n rag python -m pytest tests/test_onboard_integration.py -m integration -v
    (set RAG_LLM_MODEL / QDRANT_URL as in the handover for host runs)."""

    def test_onboard_builds_and_validates(self):
        report = run_onboard(FIXTURES, collection="mailrag-itest", validate=False)
        self.assertGreaterEqual(report.kept, 1)
        self.assertGreaterEqual(report.chunks, 1)
        self.assertEqual(report.collection, "mailrag-itest")


if __name__ == "__main__":
    unittest.main()
