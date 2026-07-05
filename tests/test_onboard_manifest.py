import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src import onboard
from src.onboard import OnboardReport, latest_manifest_collection, write_manifest


def _report(coll, validated=True):
    return OnboardReport(
        collection=coll,
        kept=10,
        noise_dropped=2,
        llm_failures=0,
        chunks=12,
        chunk_size=512,
        coverage_at3=0.8 if validated else None,
        n_queries=5 if validated else 0,
        validated=validated,
    )


class TestReportManifest(unittest.TestCase):
    def test_one_line_validated_and_skipped(self):
        self.assertIn("coverage@3 = 80%", _report("c").one_line())
        self.assertIn("validation skipped", _report("c", validated=False).one_line())

    def test_manifest_roundtrip_and_latest(self):
        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d, True)
        with mock.patch.object(onboard, "MANIFEST_DIR", Path(d)):
            # First manifest is written only so the "latest wins" assertion
            # below has an earlier entry to beat; its path is unused.
            write_manifest(_report("mailrag-a"), source="/x", model="M")
            p2 = write_manifest(_report("mailrag-b"), source="/y", model="M")
            data = json.loads(Path(p2).read_text())
            self.assertEqual(data["collection"], "mailrag-b")
            self.assertEqual(data["defaults"]["sparse_weight"], 1)
            # most-recently-written manifest wins
            self.assertEqual(latest_manifest_collection(), "mailrag-b")


if __name__ == "__main__":
    unittest.main()
