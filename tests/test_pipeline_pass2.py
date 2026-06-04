import unittest
from unittest import mock
from src.profile import CorpusProfile


class TestPass2Stage(unittest.TestCase):
    def test_sweeps_selection_with_workers(self):
        from src.pipeline import pass2 as pass2_stage
        prof = CorpusProfile(root="/r", selection_rules=[{"type": "prefix", "value": "a/"}],
                             pass2_cache="/tmp/p.db")
        with mock.patch("src.pipeline.pass2.resolve_index_files", return_value=(["/r/a/x.eml"], [])), \
             mock.patch("src.pipeline.pass2.Pass2Cache"), \
             mock.patch("src.pipeline.pass2.llm_client") as cl, \
             mock.patch("src.pipeline.pass2.run_pass", return_value={"done": 1, "cached": 0, "error": 0}) as rp:
            cl.make_client.return_value = mock.Mock()
            counts = pass2_stage.run(prof, model="gemma", workers=4)
        self.assertEqual(counts["done"], 1)
        _, kwargs = rp.call_args
        self.assertEqual(kwargs["workers"], 4)


if __name__ == "__main__":
    unittest.main()
