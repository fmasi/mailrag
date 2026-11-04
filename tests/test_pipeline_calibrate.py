import unittest
from unittest import mock

from src.llm.calibration import CalibrationReport
from src.profile import CorpusProfile


class TestJudgeSample(unittest.TestCase):
    def test_collects_records_and_skips_errors(self):
        from src.pipeline import calibrate

        paths = ["a", "b", "boom"]

        def load_email(p):
            if p == "boom":
                raise ValueError("bad parse")
            return {"sender": "s", "subject": p, "date": "d", "body": "x"}

        def judge(email):
            return {
                "is_noise": email["subject"] == "a",
                "confidence": 1.0,
                "summary": "sum",
                "reason": "rs",
            }

        recs = calibrate.judge_sample(paths, load_email, judge, workers=1)
        self.assertEqual(len(recs), 2)
        subjects = sorted(r["subject"] for r in recs)
        self.assertEqual(subjects, ["a", "b"])
        self.assertTrue(any(r["is_noise"] for r in recs))

    def test_threaded_collects_and_skips_errors(self):
        from src.pipeline import calibrate

        paths = ["a", "b", "boom", "c"]

        def load_email(p):
            if p == "boom":
                raise ValueError("bad parse")
            return {"sender": "s", "subject": p, "date": "d", "body": "x"}

        def judge(email):
            return {"is_noise": False, "confidence": 1.0, "summary": "s", "reason": "r"}

        recs = calibrate.judge_sample(paths, load_email, judge, workers=2)
        self.assertEqual(sorted(r["subject"] for r in recs), ["a", "b", "c"])


class TestCalibrateRun(unittest.TestCase):
    def test_run_returns_report_for_profile_rubric(self):
        from src.pipeline import calibrate

        prof = CorpusProfile(
            root="/r", selection_rules=[{"type": "prefix", "value": "a/"}], rubric="work"
        )
        sample_paths = ["/r/a/1.eml", "/r/a/2.eml"]

        def fake_judge_sample(paths, load_email, judge, workers, progress=False):
            return [
                {
                    "sender": "s",
                    "subject": "Your invoice",
                    "is_noise": True,
                    "confidence": 1.0,
                    "summary": "",
                    "reason": "digest",
                },
                {
                    "sender": "s",
                    "subject": "Hi there",
                    "is_noise": False,
                    "confidence": 1.0,
                    "summary": "a note",
                    "reason": "human",
                },
            ]

        with (
            mock.patch(
                "src.pipeline.calibrate.resolve_index_files", return_value=(sample_paths, [])
            ),
            mock.patch("src.pipeline.calibrate.sample_files", return_value=sample_paths),
            mock.patch("src.pipeline.calibrate.llm_client") as cl,
            mock.patch("src.pipeline.calibrate._make_load_email", return_value=lambda p: {}),
            mock.patch("src.pipeline.calibrate.judge_sample", side_effect=fake_judge_sample),
        ):
            cl.make_client.return_value = mock.Mock()
            report = calibrate.run(prof, model="gemma", sample=2, workers=1)

        self.assertIsInstance(report, CalibrationReport)
        self.assertEqual(report.rubric, "work")
        self.assertEqual(report.sample, 2)
        self.assertAlmostEqual(report.noise_rate, 0.5)
        self.assertEqual(len(report.false_noise), 1)  # "Your invoice" flagged noise

    def test_run_skips_files_on_the_profile_skip_list(self):
        """Calibration samples what judge and index would see: skip-listed files are out."""
        import os
        import tempfile

        from src.data.blacklist import append_to_blacklist, file_sha256
        from src.pipeline import calibrate

        with tempfile.TemporaryDirectory() as root:
            os.makedirs(os.path.join(root, "a"))
            kept_path = os.path.join(root, "a", "keep.eml")
            skip_path = os.path.join(root, "a", "skip.eml")
            for path, body in (
                (kept_path, b"Subject: hi\n\nhello"),
                (skip_path, b"Subject: r\n\nx"),
            ):
                with open(path, "wb") as fh:
                    fh.write(body)
            blacklist = os.path.join(root, "skip.txt")
            append_to_blacklist(blacklist, [file_sha256(skip_path)])
            prof = CorpusProfile(
                root=root,
                selection_rules=[{"type": "prefix", "value": "a/"}],
                rubric="work",
                blacklist=blacklist,
            )
            seen = []

            def fake_judge_sample(paths, load_email, judge, workers, progress=False):
                seen.extend(paths)
                return [
                    {
                        "sender": "s",
                        "subject": "hi",
                        "is_noise": False,
                        "confidence": 1.0,
                        "summary": "x",
                        "reason": "r",
                    }
                ]

            with (
                mock.patch("src.pipeline.calibrate.llm_client"),
                mock.patch("src.pipeline.calibrate.judge_sample", side_effect=fake_judge_sample),
            ):
                calibrate.run(prof, model="gemma", sample=10, workers=1)

        self.assertEqual([os.path.basename(x) for x in seen], ["keep.eml"])

    def test_run_raises_when_no_email_could_be_judged(self):
        """Every sampled email failing must not come back as a 0-email report."""
        from src.pipeline import calibrate

        prof = CorpusProfile(
            root="/r", selection_rules=[{"type": "prefix", "value": "a/"}], rubric="work"
        )
        with (
            mock.patch(
                "src.pipeline.calibrate.resolve_index_files", return_value=(["/r/a/1.eml"], [])
            ),
            mock.patch("src.pipeline.calibrate.llm_client"),
            mock.patch("src.pipeline.calibrate.judge_sample", return_value=[]),
        ):
            with self.assertRaises(RuntimeError) as ctx:
                calibrate.run(prof, model="gemma", sample=1, workers=1)
        self.assertIn("0 of 1", str(ctx.exception))

    def test_cli_records_nothing_when_calibration_fails(self):
        import argparse
        import json
        import os
        import tempfile

        from src import cli

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "p.json")
            CorpusProfile(root=d, selection_rules=[], rubric="work").save(path)
            args = argparse.Namespace(profile=path, model="m", sample=1, seed=1, workers=1)
            with mock.patch(
                "src.cli.calibrate_stage.run", side_effect=RuntimeError("judged 0 of 1")
            ):
                with self.assertRaises(RuntimeError):
                    cli._cmd_calibrate(args)
            with open(path) as fh:
                self.assertFalse(json.load(fh).get("calibration"))

    def test_run_fails_fast_on_an_unknown_rubric(self):
        """A missing rubric is a config error: raise before sampling, never per email."""
        from src.pipeline import calibrate

        prof = CorpusProfile(
            root="/r", selection_rules=[{"type": "prefix", "value": "a/"}], rubric="no-such-rubric"
        )
        with (
            mock.patch("src.pipeline.calibrate.resolve_index_files") as resolve,
            mock.patch("src.pipeline.calibrate.llm_client"),
        ):
            with self.assertRaises(ValueError):
                calibrate.run(prof, model="gemma", sample=1, workers=1)
        resolve.assert_not_called()


if __name__ == "__main__":
    unittest.main()
