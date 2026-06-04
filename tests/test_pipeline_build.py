import unittest
from unittest import mock
from src.profile import CorpusProfile
from src.data.models import NormalizedEmail


def _email(sender):
    return NormalizedEmail(sender=sender, subject="s", date=None, body="b",
                           source="t", source_id="t0")


class TestBuildStage(unittest.TestCase):
    def test_runs_pass1_and_disables_internal_filter(self):
        from src.pipeline import build
        prof = CorpusProfile(root="/r", selection_rules=[{"type": "prefix", "value": "a/"}],
                             chunk_size=1024, collection="c", qdrant_url="http://x")
        emails = [_email("a@junk.example"), _email("b@real.example")]
        with mock.patch("src.pipeline.build.resolve_index_files", return_value=(["/r/a/x.eml"], [])), \
             mock.patch("src.pipeline.build.MailArchiveXLoader") as Loader, \
             mock.patch("src.pipeline.build.build_contextual_index") as bci, \
             mock.patch("src.pipeline.build.NoiseFilter") as NF:
            Loader.return_value.load.return_value = emails
            NF.from_project_rules.return_value.matched_category.side_effect = \
                lambda e: "junk" if "junk" in e.sender else None
            build.run(prof, embedder=mock.Mock(), recreate=True)
        _, kwargs = bci.call_args
        self.assertIs(kwargs["apply_noise_filter"], False)
        self.assertEqual(kwargs["collection"], "c")
        self.assertEqual(kwargs["chunk_size"], 1024)
        self.assertTrue(emails[0].noise_candidate)
        self.assertFalse(emails[1].noise_candidate)


if __name__ == "__main__":
    unittest.main()
