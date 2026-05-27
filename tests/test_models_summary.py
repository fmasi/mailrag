# tests/test_models_summary.py
"""The optional summary is stored in payload but excluded from embedding."""
import unittest

from src.data.models import NormalizedEmail


def _email(**kw):
    base = dict(sender="a@x.com", subject="S", date=None, body="B",
                source="local", source_id="/p/a.eml")
    base.update(kw)
    return NormalizedEmail(**base)


class TestSummaryMetadata(unittest.TestCase):
    def test_summary_present_in_payload_and_embed_excluded(self):
        doc = _email(summary="A short summary").to_document(doc_id="local_0")
        self.assertEqual(doc.metadata["summary"], "A short summary")
        self.assertIn("summary", doc.excluded_embed_metadata_keys)
        self.assertIn("summary", doc.excluded_llm_metadata_keys)

    def test_no_summary_means_no_key(self):
        doc = _email().to_document(doc_id="local_0")
        self.assertNotIn("summary", doc.metadata)


if __name__ == "__main__":
    unittest.main()
