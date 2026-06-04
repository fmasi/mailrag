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


    def test_summarize_uses_profile_rubric(self):
        from src.pipeline import pass2 as pass2_stage
        prof = CorpusProfile(root="/r", selection_rules=[{"type": "prefix", "value": "a/"}],
                             pass2_cache="/tmp/p.db", rubric="personal")
        captured = {}

        def fake_run_pass(paths, cache, load_email, summarize, model, **kw):
            # Drive the summarize closure once to observe which rubric it builds.
            summarize({"sender": "s", "subject": "j", "date": "d", "body": "b"})
            return {"done": 1, "cached": 0, "error": 0}

        with mock.patch("src.pipeline.pass2.resolve_index_files", return_value=(["/r/a/x.eml"], [])), \
             mock.patch("src.pipeline.pass2.Pass2Cache"), \
             mock.patch("src.pipeline.pass2.llm_client") as cl, \
             mock.patch("src.pipeline.pass2.rubrics.build_prompt", return_value="P") as bp, \
             mock.patch("src.pipeline.pass2.summary.parse_response",
                        return_value={"is_noise": False, "confidence": 1.0,
                                      "summary": "x", "reason": "y"}), \
             mock.patch("src.pipeline.pass2.run_pass", side_effect=fake_run_pass):
            cl.make_client.return_value = mock.Mock()
            cl.chat.return_value = "{}"
            pass2_stage.run(prof, model="gemma", workers=1)
        self.assertEqual(bp.call_args.args[0], "personal")


if __name__ == "__main__":
    unittest.main()
