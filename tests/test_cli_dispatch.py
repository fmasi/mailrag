import json, os, tempfile, unittest
from unittest import mock
from src.profile import CorpusProfile


class TestCliDispatch(unittest.TestCase):
    def _profile_file(self, d):
        fp = os.path.join(d, "p.json")
        CorpusProfile(root="/r", selection_rules=[{"type": "prefix", "value": "a/"}],
                      collection="c", chunk_size=512).save(fp)
        return fp

    def test_pass1_verb_previews_partition(self):
        from src import cli
        with tempfile.TemporaryDirectory() as d:
            fp = self._profile_file(d)
            with mock.patch("src.cli.resolve_index_files", return_value=(["/r/a/x.eml"], [])), \
                 mock.patch("src.cli.MailArchiveXLoader") as Loader, \
                 mock.patch("src.cli.NoiseFilter") as NF:
                from src.data.models import NormalizedEmail
                Loader.return_value.load.return_value = [
                    NormalizedEmail(sender="a@junk.example", subject="s", date=None,
                                    body="b", source="t", source_id="t0")]
                NF.from_project_rules.return_value.matched_category.return_value = "junk"
                rc = cli.main(["pass1", "--profile", fp])
        self.assertEqual(rc, 0)

    def test_build_verb_saves_profile(self):
        from src import cli
        with tempfile.TemporaryDirectory() as d:
            fp = self._profile_file(d)
            with mock.patch("src.cli.build_stage") as bs, \
                 mock.patch("src.cli.BgeM3Embedder"):
                bs.run.return_value = mock.Mock(chunks=10, collection="c")
                rc = cli.main(["build", "--profile", fp, "--limit", "1"])
        self.assertEqual(rc, 0)
        self.assertTrue(bs.run.called)
        self.assertIs(bs.run.call_args.kwargs["embed_summary"], False)

    def test_build_embed_summary_flows_flag_and_threshold(self):
        from src import cli
        with tempfile.TemporaryDirectory() as d:
            fp = self._profile_file(d)
            with mock.patch("src.cli.build_stage") as bs, \
                 mock.patch("src.cli.BgeM3Embedder"):
                bs.run.return_value = mock.Mock(chunks=5, collection="c")
                rc = cli.main(["build", "--profile", fp, "--embed-summary",
                               "--noise-confidence", "0.8"])
        self.assertEqual(rc, 0)
        self.assertIs(bs.run.call_args.kwargs["embed_summary"], True)
        self.assertEqual(bs.run.call_args.kwargs["noise_min_confidence"], 0.8)


    def test_calibrate_verb_records_calibration(self):
        from src import cli
        from src.llm.calibration import CalibrationReport
        from io import StringIO
        with tempfile.TemporaryDirectory() as d:
            fp = self._profile_file(d)
            report = CalibrationReport(rubric="personal", sample=5, noise_rate=0.6,
                                       false_noise=[], false_keep=[])
            with mock.patch("src.cli.calibrate_stage") as cs, \
                 mock.patch("sys.stdout", new_callable=StringIO) as out:
                cs.run.return_value = report
                rc = cli.main(["calibrate", "--profile", fp, "--model", "gemma"])
            self.assertEqual(rc, 0)
            printed = out.getvalue()
            self.assertIn("FALSE-NOISE", printed)   # the buckets are surfaced
            self.assertIn("personal", printed)
            saved = CorpusProfile.load(fp)
            self.assertIsNotNone(saved.calibration)
            self.assertEqual(saved.calibration["rubric"], "personal")
            self.assertTrue(saved.calibration["passed"])
            self.assertAlmostEqual(saved.calibration["noise_rate"], 0.6)

    def test_pass2_blocked_without_calibration(self):
        from src import cli
        with tempfile.TemporaryDirectory() as d:
            fp = self._profile_file(d)  # no calibration recorded
            with mock.patch("src.cli.pass2_stage") as ps:
                rc = cli.main(["pass2", "--profile", fp, "--model", "gemma"])
        self.assertEqual(rc, 2)
        self.assertFalse(ps.run.called)

    def test_pass2_blocked_when_calibration_rubric_mismatches(self):
        from src import cli
        with tempfile.TemporaryDirectory() as d:
            fp = self._profile_file(d)
            prof = CorpusProfile.load(fp)
            prof.calibration = {"rubric": "work", "passed": True, "noise_rate": 0.8,
                                "sample": 5, "false_noise": 0, "false_keep": 0, "at": "t"}
            prof.save(fp)  # profile.rubric is "personal", calibration is for "work"
            with mock.patch("src.cli.pass2_stage") as ps:
                rc = cli.main(["pass2", "--profile", fp, "--model", "gemma"])
        self.assertEqual(rc, 2)
        self.assertFalse(ps.run.called)

    def test_pass2_runs_with_matching_calibration(self):
        from src import cli
        with tempfile.TemporaryDirectory() as d:
            fp = self._profile_file(d)
            prof = CorpusProfile.load(fp)
            prof.calibration = {"rubric": "personal", "passed": True, "noise_rate": 0.6,
                                "sample": 5, "false_noise": 0, "false_keep": 0, "at": "t"}
            prof.save(fp)
            with mock.patch("src.cli.pass2_stage") as ps:
                ps.run.return_value = {"done": 1, "cached": 0, "error": 0}
                rc = cli.main(["pass2", "--profile", fp, "--model", "gemma"])
        self.assertEqual(rc, 0)
        self.assertTrue(ps.run.called)

    def test_pass2_force_bypasses_gate(self):
        from src import cli
        with tempfile.TemporaryDirectory() as d:
            fp = self._profile_file(d)  # no calibration
            with mock.patch("src.cli.pass2_stage") as ps:
                ps.run.return_value = {"done": 1, "cached": 0, "error": 0}
                rc = cli.main(["pass2", "--profile", fp, "--model", "gemma", "--force"])
        self.assertEqual(rc, 0)
        self.assertTrue(ps.run.called)


    def test_pass2_errors_when_profile_has_no_rubric(self):
        from src import cli
        with tempfile.TemporaryDirectory() as d:
            fp = self._profile_file(d)
            prof = CorpusProfile.load(fp)
            prof.rubric = ""
            prof.save(fp)
            with mock.patch("src.cli.pass2_stage") as ps:
                rc = cli.main(["pass2", "--profile", fp, "--model", "gemma"])
        self.assertEqual(rc, 1)  # ValueError -> exit 1
        self.assertFalse(ps.run.called)


    def test_calibrate_then_pass2_roundtrip_unlocks_gate(self):
        # The dict _cmd_calibrate WRITES must be ACCEPTED by the _cmd_pass2 gate
        # (no hand-built calibration dict here — calibrate writes it, pass2 reads it).
        from src import cli
        from src.llm.calibration import CalibrationReport
        with tempfile.TemporaryDirectory() as d:
            fp = self._profile_file(d)  # profile.rubric defaults to "personal"
            report = CalibrationReport(rubric="personal", sample=5, noise_rate=0.6,
                                       false_noise=[], false_keep=[])
            with mock.patch("src.cli.calibrate_stage") as cs:
                cs.run.return_value = report
                rc_cal = cli.main(["calibrate", "--profile", fp, "--model", "gemma"])
            self.assertEqual(rc_cal, 0)
            # Now pass2 on the SAME profile file should pass the gate (no --force).
            with mock.patch("src.cli.pass2_stage") as ps:
                ps.run.return_value = {"done": 1, "cached": 0, "error": 0}
                rc_p2 = cli.main(["pass2", "--profile", fp, "--model", "gemma"])
            self.assertEqual(rc_p2, 0)
            self.assertTrue(ps.run.called)


if __name__ == "__main__":
    unittest.main()
