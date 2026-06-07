import unittest
from types import SimpleNamespace
from unittest import mock

from src.pipeline.judge import select_suspects, run


def _scan(clusters):
    return {"clusters": clusters}


class TestSelectSuspects(unittest.TestCase):
    def test_collects_paths_from_high_score_clusters(self):
        scan = _scan([
            {"score": 0.8, "members": [{"thread_id": "t1", "paths": ["a.eml", "b.eml"]}]},
            {"score": 0.3, "members": [{"thread_id": "t2", "paths": ["c.eml"]}]},
            {"score": 0.6, "members": [{"thread_id": "t3", "paths": ["d.eml", "a.eml"]}]},
        ])
        out = select_suspects(scan, min_score=0.6)
        self.assertEqual(out, ["a.eml", "b.eml", "d.eml"])   # >=0.6, deduped, in order

    def test_threshold_excludes_below(self):
        scan = _scan([{"score": 0.59, "members": [{"thread_id": "t", "paths": ["x.eml"]}]}])
        self.assertEqual(select_suspects(scan, min_score=0.6), [])


class TestJudgeRun(unittest.TestCase):
    def test_judges_only_suspects(self):
        prof = SimpleNamespace(pass2_cache="/tmp/c.db", rubric="personal")
        scan = _scan([
            {"score": 0.9, "members": [{"thread_id": "t1", "paths": ["a.eml"]}]},
            {"score": 0.1, "members": [{"thread_id": "t2", "paths": ["z.eml"]}]},
        ])
        with mock.patch("src.pipeline.judge.os.path.exists", return_value=True), \
             mock.patch("src.pipeline.judge._load_json", return_value=scan), \
             mock.patch("src.pipeline.judge.Pass2Cache"), \
             mock.patch("src.pipeline.judge.llm_client"), \
             mock.patch("src.pipeline.judge._make_load_email"), \
             mock.patch("src.pipeline.judge.run_pass",
                        return_value={"done": 1, "cached": 0, "error": 0}) as rp:
            counts = run(prof, model="m", scan_json="/tmp/s.json",
                         min_score=0.6, progress=False)
        passed_paths = rp.call_args.args[0]
        self.assertEqual(passed_paths, ["a.eml"])      # only the high-score suspect
        self.assertEqual(counts["suspects"], 1)

    def test_missing_scan_json_raises(self):
        prof = SimpleNamespace(pass2_cache="/tmp/c.db", rubric="personal")
        with mock.patch("src.pipeline.judge.os.path.exists", return_value=False):
            with self.assertRaises(ValueError):
                run(prof, model="m", scan_json="/nope.json")


if __name__ == "__main__":
    unittest.main()
