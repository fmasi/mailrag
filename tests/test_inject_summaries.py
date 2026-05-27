# tests/test_inject_summaries.py
"""inject_summaries copies cached, non-noise summaries onto email objects."""
import os
import tempfile
import unittest

from src.data.blacklist import file_sha256
from src.llm import pass2
from src.llm.cache import Pass2Cache


class _Email:
    """Minimal stand-in carrying the fields inject_summaries touches."""
    def __init__(self, source_id):
        self.source_id = source_id
        self.summary = None


class TestInjectSummaries(unittest.TestCase):
    def test_injects_only_kept_with_summary(self):
        with tempfile.TemporaryDirectory() as d:
            kept = os.path.join(d, "kept.eml")
            noise = os.path.join(d, "noise.eml")
            uncached = os.path.join(d, "uncached.eml")
            for p, data in ((kept, b"k"), (noise, b"n"), (uncached, b"u")):
                with open(p, "wb") as fh:
                    fh.write(data)
            cache = Pass2Cache(os.path.join(d, "c.db"))
            cache.put(file_sha256(kept),
                      {"is_noise": False, "confidence": 0.9, "summary": "keep me", "reason": "r"})
            cache.put(file_sha256(noise),
                      {"is_noise": True, "confidence": 0.9, "summary": "", "reason": "ad"})

            emails = [_Email(kept), _Email(noise), _Email(uncached)]
            n = pass2.inject_summaries(emails, cache)

            self.assertEqual(n, 1)
            self.assertEqual(emails[0].summary, "keep me")
            self.assertIsNone(emails[1].summary)   # noise -> no summary
            self.assertIsNone(emails[2].summary)   # uncached -> untouched
            cache.close()


if __name__ == "__main__":
    unittest.main()
