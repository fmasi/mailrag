import unittest
from unittest import mock
from src.profile import CorpusProfile


class TestProfileStage(unittest.TestCase):
    def test_returns_suggested_chunk_size_and_sets_profile(self):
        from src.pipeline import profile as profile_stage
        prof = CorpusProfile(root="/r", selection_rules=[{"type": "prefix", "value": "a/"}])
        with mock.patch("src.pipeline.profile._cleaned_token_lengths", return_value=[100, 300, 2000]), \
             mock.patch("src.pipeline.profile.suggest_chunk_size", return_value=1024):
            report = profile_stage.run(prof, set_profile=True)
        self.assertEqual(report.suggested_chunk_size, 1024)
        self.assertEqual(prof.chunk_size, 1024)


if __name__ == "__main__":
    unittest.main()
