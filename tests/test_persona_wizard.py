import unittest
from types import SimpleNamespace
from unittest import mock

from src.persona.wizard import run_wizard


class _Ans:
    def __init__(self, val):
        self._val = val

    def ask(self):
        return self._val


class FakeQ:
    """Scripted questionary stand-in: pops the next answer per prompt type."""
    def __init__(self, selects=None, texts=None, confirms=None):
        self.selects = list(selects or [])
        self.texts = list(texts or [])
        self.confirms = list(confirms or [])
        self.select_messages = []

    def select(self, message, choices=None, default=None):
        self.select_messages.append(message)
        return _Ans(self.selects.pop(0))

    def text(self, message):
        return _Ans(self.texts.pop(0))

    def confirm(self, message):
        return _Ans(self.confirms.pop(0))


class FakeConsole:
    def __init__(self):
        self.lines = []

    def print(self, *a, **k):
        self.lines.append(" ".join(str(x) for x in a))


def _recording_handlers(calls):
    def make(verb):
        def handler(prof, **params):
            calls.append(verb)
            return f"{verb}-report"
        return handler
    return {v: make(v) for v in
            ("scope", "measure", "tag", "scan", "calibrate", "summarize", "index",
             "judge", "prune")}


class TestWizard(unittest.TestCase):
    def _patch(self, calls):
        prof = SimpleNamespace(rubric="personal", save=mock.Mock())
        return (
            mock.patch("src.persona.wizard.build_handlers",
                       return_value=_recording_handlers(calls)),
            mock.patch("src.persona.wizard.CorpusProfile.load", return_value=prof),
            mock.patch("src.persona.wizard._read_recommendation", return_value=None),
            prof,
        )

    def test_llm_none_runs_steps_in_order_no_model(self):
        calls = []
        bh, load, rec, prof = self._patch(calls)
        q = FakeQ(selects=["llm-none"])
        with bh, load, rec:
            rc = run_wizard("p.json", questionary=q, console=FakeConsole())
        self.assertEqual(rc, 0)
        self.assertEqual(calls, ["scope", "measure", "scan", "tag", "prune", "index"])
        prof.save.assert_called_once_with("p.json")

    def test_llm_all_calibrate_gate_proceed(self):
        calls = []
        bh, load, rec, prof = self._patch(calls)
        q = FakeQ(selects=["llm-all", "proceed to the LLM pass"],
                  texts=["mymodel"], confirms=[True])
        with bh, load, rec, \
             mock.patch("src.llm.calibration.format_report", return_value="BUCKETS"):
            rc = run_wizard("p.json", questionary=q, console=FakeConsole())
        self.assertEqual(rc, 0)
        self.assertEqual(calls, ["scope", "measure", "calibrate", "summarize",
                                 "prune", "index"])

    def test_calibrate_gate_abort_stops_before_summarize(self):
        calls = []
        bh, load, rec, prof = self._patch(calls)
        q = FakeQ(selects=["llm-all", "abort"], texts=["m"])
        with bh, load, rec, \
             mock.patch("src.llm.calibration.format_report", return_value="B"):
            rc = run_wizard("p.json", questionary=q, console=FakeConsole())
        self.assertEqual(rc, 1)
        self.assertIn("calibrate", calls)
        self.assertNotIn("summarize", calls)

    def test_calibrate_gate_retune_changes_rubric_then_proceeds(self):
        calls = []
        bh, load, rec, prof = self._patch(calls)
        q = FakeQ(selects=["llm-all",
                           "re-tune (pick another rubric and re-calibrate)",
                           "work",
                           "proceed to the LLM pass"],
                  texts=["m"], confirms=[True])
        with bh, load, rec, \
             mock.patch("src.llm.calibration.format_report", return_value="B"), \
             mock.patch("src.llm.rubrics.names", return_value=["personal", "work"]):
            rc = run_wizard("p.json", questionary=q, console=FakeConsole())
        self.assertEqual(rc, 0)
        self.assertEqual(prof.rubric, "work")             # re-tune applied
        self.assertEqual(calls.count("calibrate"), 2)      # calibrated twice

    def test_summarize_confirm_no_stops(self):
        calls = []
        bh, load, rec, prof = self._patch(calls)
        q = FakeQ(selects=["llm-all", "proceed to the LLM pass"],
                  texts=["m"], confirms=[False])
        with bh, load, rec, \
             mock.patch("src.llm.calibration.format_report", return_value="B"):
            rc = run_wizard("p.json", questionary=q, console=FakeConsole())
        self.assertEqual(rc, 1)
        self.assertNotIn("summarize", calls)

    def test_llm_verify_runs_judge_and_prune(self):
        calls = []
        bh, load, rec, prof = self._patch(calls)
        # llm-verify: scope, measure, scan, calibrate, judge, prune, summarize, index
        q = FakeQ(selects=["llm-verify", "proceed to the LLM pass"],
                  texts=["mymodel"], confirms=[True])
        with bh, load, rec, \
             mock.patch("src.llm.calibration.format_report", return_value="B"):
            rc = run_wizard("p.json", questionary=q, console=FakeConsole())
        self.assertEqual(rc, 0)
        self.assertEqual(calls, ["scope", "measure", "scan", "calibrate",
                                 "judge", "prune", "summarize", "index"])

    def test_passes_prune_confirm_to_build_handlers(self):
        calls = []
        prof = SimpleNamespace(rubric="personal", save=mock.Mock())
        captured = {}

        def fake_build_handlers(**kwargs):
            captured.update(kwargs)
            return _recording_handlers(calls)

        q = FakeQ(selects=["llm-none"])
        with mock.patch("src.persona.wizard.build_handlers",
                        side_effect=fake_build_handlers), \
             mock.patch("src.persona.wizard.CorpusProfile.load", return_value=prof), \
             mock.patch("src.persona.wizard._read_recommendation", return_value=None):
            run_wizard("p.json", questionary=q, console=FakeConsole())
        self.assertTrue(callable(captured.get("prune_confirm")))


if __name__ == "__main__":
    unittest.main()
