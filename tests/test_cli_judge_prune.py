import unittest
from unittest import mock

from src import cli


def _prof(**kw):
    base = dict(rubric="personal", calibration={"rubric": "personal", "passed": True},
                blacklist=None, pass2_cache="/tmp/c.db")
    base.update(kw)
    return mock.Mock(**base)


class TestJudgeVerb(unittest.TestCase):
    def test_judge_routes_when_calibrated(self):
        with mock.patch("src.cli.CorpusProfile.load", return_value=_prof()), \
             mock.patch("src.cli.judge_stage.run",
                        return_value={"done": 3, "suspects": 3}) as run:
            rc = cli.main(["judge", "--profile", "p.json", "--model", "m",
                           "--min-score", "0.7"])
        self.assertEqual(rc, 0)
        self.assertEqual(run.call_args.kwargs["min_score"], 0.7)
        self.assertEqual(run.call_args.kwargs["scan_json"], "p.scan.json")

    def test_judge_uncalibrated_exits_2(self):
        with mock.patch("src.cli.CorpusProfile.load",
                        return_value=_prof(calibration=None)):
            rc = cli.main(["judge", "--profile", "p.json", "--model", "m"])
        self.assertEqual(rc, 2)


class TestPruneVerb(unittest.TestCase):
    def test_prune_dry_run_without_yes(self):
        prof = _prof()
        with mock.patch("src.cli.CorpusProfile.load", return_value=prof), \
             mock.patch("src.cli.prune_stage.collect",
                        return_value=(["h1", "h2"], ["0.9 newsletter"])), \
             mock.patch("src.cli.prune_stage.run") as run:
            rc = cli.main(["prune", "--profile", "p.json", "--from", "judge"])
        self.assertEqual(rc, 0)
        run.assert_not_called()                 # dry run: no write without --yes
        self.assertEqual(prof.blacklist, "p.blacklist.txt")  # defaulted

    def test_prune_writes_with_yes(self):
        prof = _prof()
        with mock.patch("src.cli.CorpusProfile.load", return_value=prof), \
             mock.patch("src.cli.prune_stage.collect",
                        return_value=(["h1"], ["0.9 x"])), \
             mock.patch("src.cli.prune_stage.run", return_value=1) as run:
            rc = cli.main(["prune", "--profile", "p.json", "--from", "judge", "--yes"])
        self.assertEqual(rc, 0)
        run.assert_called_once()
        prof.save.assert_called_once_with("p.json")

    def test_prune_nothing_to_do(self):
        with mock.patch("src.cli.CorpusProfile.load", return_value=_prof()), \
             mock.patch("src.cli.prune_stage.collect", return_value=([], [])):
            rc = cli.main(["prune", "--profile", "p.json", "--from", "tag"])
        self.assertEqual(rc, 0)


if __name__ == "__main__":
    unittest.main()
