import unittest
from unittest import mock

from src.profile import CorpusProfile
from src.llm.calibration import CalibrationReport


class TestJudgeSample(unittest.TestCase):
    def test_collects_records_and_skips_errors(self):
        from src.pipeline import calibrate
        paths = ["a", "b", "boom"]

        def load_email(p):
            if p == "boom":
                raise ValueError("bad parse")
            return {"sender": "s", "subject": p, "date": "d", "body": "x"}

        def judge(email):
            return {"is_noise": email["subject"] == "a", "confidence": 1.0,
                    "summary": "sum", "reason": "rs"}

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
        prof = CorpusProfile(root="/r", selection_rules=[{"type": "prefix", "value": "a/"}],
                             rubric="personal")
        sample_paths = ["/r/a/1.eml", "/r/a/2.eml"]

        def fake_judge_sample(paths, load_email, judge, workers, progress=False):
            return [{"sender": "s", "subject": "Your invoice", "is_noise": True,
                     "confidence": 1.0, "summary": "", "reason": "digest"},
                    {"sender": "s", "subject": "Hi there", "is_noise": False,
                     "confidence": 1.0, "summary": "a note", "reason": "human"}]

        with mock.patch("src.pipeline.calibrate.resolve_index_files",
                        return_value=(sample_paths, [])), \
             mock.patch("src.pipeline.calibrate.sample_files", return_value=sample_paths), \
             mock.patch("src.pipeline.calibrate.llm_client") as cl, \
             mock.patch("src.pipeline.calibrate._make_load_email", return_value=lambda p: {}), \
             mock.patch("src.pipeline.calibrate.judge_sample", side_effect=fake_judge_sample):
            cl.make_client.return_value = mock.Mock()
            report = calibrate.run(prof, model="gemma", sample=2, workers=1)

        self.assertIsInstance(report, CalibrationReport)
        self.assertEqual(report.rubric, "personal")
        self.assertEqual(report.sample, 2)
        self.assertAlmostEqual(report.noise_rate, 0.5)
        self.assertEqual(len(report.false_noise), 1)  # "Your invoice" flagged noise


if __name__ == "__main__":
    unittest.main()
