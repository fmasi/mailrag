import unittest
from types import SimpleNamespace
from unittest import mock

from src.pipeline import pass2 as pass2_stage
from src.pipeline import build as build_stage


def _profile(blacklist):
    return SimpleNamespace(
        resolved_root=lambda: "/root", selection_rules=[], blacklist=blacklist,
        pass2_cache="/tmp/c.db", rubric="personal", collection="c",
        chunk_size=512, chunk_overlap=64, qdrant_url="http://localhost:6333")


class TestBlacklistThreading(unittest.TestCase):
    def test_summarize_passes_profile_blacklist_to_resolver(self):
        prof = _profile("/tmp/bl.txt")
        with mock.patch("src.pipeline.pass2.resolve_index_files",
                        return_value=([], [])) as rif, \
             mock.patch("src.pipeline.pass2.Pass2Cache"), \
             mock.patch("src.pipeline.pass2.llm_client"), \
             mock.patch("src.pipeline.pass2.run_pass", return_value={}):
            pass2_stage.run(prof, model="m", progress=False)
        self.assertEqual(rif.call_args.args[2], "/tmp/bl.txt")

    def test_index_passes_profile_blacklist_to_resolver(self):
        prof = _profile("/tmp/bl.txt")
        res = SimpleNamespace(chunks=0, collection="c")
        with mock.patch("src.pipeline.build.resolve_index_files",
                        return_value=([], [])) as rif, \
             mock.patch("src.pipeline.build.MailArchiveXLoader") as loader, \
             mock.patch("src.pipeline.build.pass1") as p1, \
             mock.patch("src.pipeline.build.build_contextual_index", return_value=res):
            loader.return_value.load.return_value = []
            p1.run.return_value = ([], SimpleNamespace(dropped=0, kept=0, tagged=0))
            build_stage.run(prof, embedder=mock.Mock())
        self.assertEqual(rif.call_args.args[2], "/tmp/bl.txt")

    def test_none_blacklist_is_passed_through(self):
        prof = _profile(None)
        with mock.patch("src.pipeline.pass2.resolve_index_files",
                        return_value=([], [])) as rif, \
             mock.patch("src.pipeline.pass2.Pass2Cache"), \
             mock.patch("src.pipeline.pass2.llm_client"), \
             mock.patch("src.pipeline.pass2.run_pass", return_value={}):
            pass2_stage.run(prof, model="m", progress=False)
        self.assertIsNone(rif.call_args.args[2])


if __name__ == "__main__":
    unittest.main()
