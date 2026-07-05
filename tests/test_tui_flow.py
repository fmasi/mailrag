"""Unit tests for the presentation-free wizard logic (src/tui/flow.py).

These mirror the classic-wizard tests: the flow is driven with a scripted
:class:`FakeUI` instead of a Textual app, so gates, ordering, and artifacts are
all checked headlessly."""

import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

from src.persona.registry import load_registry
from src.tui import flow


def _recording_handlers(calls):
    def make(verb):
        def handler(prof, **params):
            calls.append(verb)
            return f"{verb}-report"

        return handler

    return {
        v: make(v)
        for v in (
            "scope",
            "measure",
            "tag",
            "scan",
            "calibrate",
            "summarize",
            "index",
            "judge",
            "prune",
        )
    }


class FakeUI:
    """Scripted WizardUI: records events, pops pre-baked gate answers."""

    def __init__(self, gates=None, rubrics=None, spend=True):
        self.gates = list(gates or [])
        self.rubrics = list(rubrics or [])
        self.spend = spend
        self.events = []
        self.logs = []

    def on_step_start(self, index, step):
        self.events.append(("start", step.verb))

    def on_step_done(self, index, step, result):
        self.events.append(("done", step.verb))

    def on_step_skip(self, index, step):
        self.events.append(("skip", step.verb))

    def log(self, message):
        self.logs.append(message)

    def calibrate_gate(self, report_text):
        return self.gates.pop(0)

    def pick_rubric(self, names, current):
        return self.rubrics.pop(0)

    def confirm_spend(self):
        return self.spend


def _profile():
    return SimpleNamespace(rubric="personal", save=mock.Mock(), selection_rules=[])


class TestPersonaCards(unittest.TestCase):
    def setUp(self):
        self.reg = load_registry()

    def test_one_card_per_persona_with_costs(self):
        cards = flow.persona_cards(self.reg)
        self.assertEqual([c.name for c in cards], self.reg.names())
        llm_none = next(c for c in cards if c.name == "llm-none")
        self.assertEqual(len(llm_none.verbs), len(llm_none.costs))
        self.assertIn("index", llm_none.verbs)
        self.assertFalse(any(c.recommended for c in cards))

    def test_recommended_flag(self):
        cards = flow.persona_cards(self.reg, recommended="llm-verify")
        flagged = [c.name for c in cards if c.recommended]
        self.assertEqual(flagged, ["llm-verify"])

    def test_needs_model(self):
        self.assertFalse(flow.needs_model(self.reg.get("llm-none")))
        self.assertTrue(flow.needs_model(self.reg.get("llm-all")))
        self.assertTrue(flow.needs_model(self.reg.get("llm-verify")))


class TestValidateModel(unittest.TestCase):
    def test_strips_and_returns(self):
        self.assertEqual(flow.validate_model("  my-model "), "my-model")

    def test_blank_and_none_rejected(self):
        self.assertIsNone(flow.validate_model(""))
        self.assertIsNone(flow.validate_model("   "))
        self.assertIsNone(flow.validate_model(None))


class TestScopeTree(unittest.TestCase):
    def _mailbox(self, tmp, rels):
        for rel in rels:
            path = os.path.join(tmp, *rel.split("/"))
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as fh:
                fh.write("x")

    def test_build_scope_tree_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._mailbox(
                tmp,
                [
                    "root.eml",
                    "Inbox/a.eml",
                    "Inbox/Acme/b.eml",
                    "Inbox/Beta/c.eml",
                    "Archive/d.eml",
                ],
            )
            nodes, has_root = flow.build_scope_tree(tmp)
        self.assertTrue(has_root)
        self.assertEqual([n.node_id for n in nodes], ["Archive/", "Inbox/"])
        inbox = nodes[1]
        # direct-files pseudo node first, then the two children
        self.assertEqual(
            [c.node_id for c in inbox.children], ["Inbox/::direct", "Inbox/Acme/", "Inbox/Beta/"]
        )
        self.assertEqual(inbox.children[0].rule, {"type": "direct-root-files", "root": "Inbox/"})
        self.assertEqual(nodes[0].children, [])  # Archive has no level-2 folders

    def test_build_scope_tree_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            nodes, has_root = flow.build_scope_tree(tmp)
        self.assertEqual(nodes, [])
        self.assertFalse(has_root)

    def test_selection_to_rules_parent_wins(self):
        nodes = [
            flow.ScopeNode(
                node_id="Inbox/",
                label="Inbox/",
                rule={"type": "prefix", "value": "Inbox/"},
                children=[
                    flow.ScopeNode(
                        node_id="Inbox/Acme/",
                        label="Inbox/Acme/",
                        rule={"type": "prefix", "value": "Inbox/Acme/"},
                    )
                ],
            )
        ]
        rules = flow.selection_to_rules(nodes, ["Inbox/", "Inbox/Acme/"])
        self.assertEqual(rules, [{"type": "prefix", "value": "Inbox/"}])

    def test_selection_to_rules_children_and_root(self):
        nodes = [
            flow.ScopeNode(
                node_id="Inbox/",
                label="Inbox/",
                rule={"type": "prefix", "value": "Inbox/"},
                children=[
                    flow.ScopeNode(
                        node_id="Inbox/::direct",
                        label="(direct)",
                        rule={"type": "direct-root-files", "root": "Inbox/"},
                    ),
                    flow.ScopeNode(
                        node_id="Inbox/Acme/",
                        label="Inbox/Acme/",
                        rule={"type": "prefix", "value": "Inbox/Acme/"},
                    ),
                ],
            )
        ]
        rules = flow.selection_to_rules(
            nodes, ["Inbox/::direct", "Inbox/Acme/"], include_root_files=True
        )
        self.assertEqual(
            rules,
            [
                {"type": "container-root"},
                {"type": "direct-root-files", "root": "Inbox/"},
                {"type": "prefix", "value": "Inbox/Acme/"},
            ],
        )

    def test_selection_to_rules_nothing_checked(self):
        self.assertEqual(flow.selection_to_rules([], []), [])


class TestPlanAndHandlers(unittest.TestCase):
    def setUp(self):
        self.reg = load_registry()

    def test_plan_marks_missing_optional_as_skipped(self):
        persona = self.reg.get("llm-none")
        handlers = _recording_handlers([])
        del handlers["scan"]  # optional in llm-none
        planned = flow.plan_steps(persona, handlers, self.reg)
        self.assertEqual([s.verb for s in planned if s.skipped], ["scan"])
        # params filtered of the 'optional' marker
        scan = next(s for s in planned if s.verb == "scan")
        self.assertNotIn("optional", scan.params)

    def test_plan_carries_verb_metadata(self):
        persona = self.reg.get("llm-all")
        planned = flow.plan_steps(persona, _recording_handlers([]), self.reg)
        index = next(s for s in planned if s.verb == "index")
        self.assertEqual(index.cost, "gpu")
        self.assertTrue(index.does)
        self.assertEqual(index.params, {"embed": "summary"})

    def test_make_scope_handler_applies_rules(self):
        prof = _profile()
        rules = [{"type": "prefix", "value": "Inbox/"}]
        handler = flow.make_scope_handler(rules)
        out = handler(prof)
        self.assertEqual(prof.selection_rules, rules)
        self.assertIsNot(out[0], rules[0])  # defensive copy

    def test_prepare_handlers_overrides_scope_and_passes_prune_confirm(self):
        captured = {}

        def fake_build_handlers(**kwargs):
            captured.update(kwargs)
            return _recording_handlers([])

        confirm = mock.Mock(return_value=True)
        with mock.patch("src.tui.flow.build_handlers", side_effect=fake_build_handlers):
            handlers = flow.prepare_handlers(
                profile_path="p.json",
                model="m",
                limit=5,
                scope_rules=[{"type": "container-root"}],
                confirm_prune=confirm,
            )
        self.assertEqual(captured["prune_confirm"], confirm)
        self.assertEqual(captured["limit"], 5)
        prof = _profile()
        handlers["scope"](prof)
        self.assertEqual(prof.selection_rules, [{"type": "container-root"}])

    def test_prepare_handlers_keeps_default_scope_without_rules(self):
        handlers_map = _recording_handlers([])
        with mock.patch("src.tui.flow.build_handlers", return_value=handlers_map):
            handlers = flow.prepare_handlers(profile_path="p.json")
        self.assertIs(handlers["scope"], handlers_map["scope"])


class TestExecutePlan(unittest.TestCase):
    def setUp(self):
        self.reg = load_registry()

    def _run(self, persona_name, ui, calls=None, drop_handlers=()):
        calls = calls if calls is not None else []
        handlers = _recording_handlers(calls)
        for verb in drop_handlers:
            del handlers[verb]
        persona = self.reg.get(persona_name)
        planned = flow.plan_steps(persona, handlers, self.reg)
        prof = _profile()
        rc = flow.execute_plan(prof, "p.json", planned, handlers, ui)
        return rc, calls, prof

    def test_llm_none_runs_steps_in_order(self):
        ui = FakeUI()
        rc, calls, prof = self._run("llm-none", ui)
        self.assertEqual(rc, 0)
        self.assertEqual(calls, ["scope", "measure", "scan", "tag", "prune", "index"])
        prof.save.assert_called_once_with("p.json")

    def test_optional_step_without_handler_is_skipped(self):
        ui = FakeUI()
        rc, calls, _ = self._run("llm-none", ui, drop_handlers=("scan",))
        self.assertEqual(rc, 0)
        self.assertNotIn("scan", calls)
        self.assertIn(("skip", "scan"), ui.events)

    def test_calibrate_gate_proceed(self):
        ui = FakeUI(gates=["proceed"])
        with mock.patch("src.llm.calibration.format_report", return_value="BUCKETS"):
            rc, calls, _ = self._run("llm-all", ui)
        self.assertEqual(rc, 0)
        self.assertEqual(calls, ["scope", "measure", "calibrate", "summarize", "prune", "index"])

    def test_calibrate_gate_abort_stops_before_summarize(self):
        ui = FakeUI(gates=["abort"])
        with mock.patch("src.llm.calibration.format_report", return_value="B"):
            rc, calls, prof = self._run("llm-all", ui)
        self.assertEqual(rc, 1)
        self.assertIn("calibrate", calls)
        self.assertNotIn("summarize", calls)
        prof.save.assert_called_once_with("p.json")  # partial progress kept

    def test_calibrate_gate_dismissed_counts_as_abort(self):
        ui = FakeUI(gates=[None])
        with mock.patch("src.llm.calibration.format_report", return_value="B"):
            rc, _, _ = self._run("llm-all", ui)
        self.assertEqual(rc, 1)

    def test_calibrate_gate_retune_changes_rubric_then_proceeds(self):
        ui = FakeUI(gates=["retune", "proceed"], rubrics=["work"])
        with (
            mock.patch("src.llm.calibration.format_report", return_value="B"),
            mock.patch("src.llm.rubrics.names", return_value=["personal", "work"]),
        ):
            rc, calls, prof = self._run("llm-all", ui)
        self.assertEqual(rc, 0)
        self.assertEqual(prof.rubric, "work")
        self.assertEqual(calls.count("calibrate"), 2)

    def test_calibrate_gate_retune_cancelled_reopens_gate_without_recalibrating(self):
        ui = FakeUI(gates=["retune", "proceed"], rubrics=[None])
        with (
            mock.patch("src.llm.calibration.format_report", return_value="B"),
            mock.patch("src.llm.rubrics.names", return_value=["personal", "work"]),
        ):
            rc, calls, prof = self._run("llm-all", ui)
        self.assertEqual(rc, 0)
        self.assertEqual(prof.rubric, "personal")
        # No rubric change -> the gate re-opens on the same report; the LLM
        # calibrate pass is NOT spent again.
        self.assertEqual(calls.count("calibrate"), 1)
        self.assertEqual(ui.gates, [])  # gate shown twice

    def test_spend_declined_stops_before_summarize(self):
        ui = FakeUI(gates=["proceed"], spend=False)
        with mock.patch("src.llm.calibration.format_report", return_value="B"):
            rc, calls, prof = self._run("llm-all", ui)
        self.assertEqual(rc, 1)
        self.assertNotIn("summarize", calls)
        prof.save.assert_called_once_with("p.json")

    def test_profile_saved_even_when_a_handler_raises(self):
        ui = FakeUI()
        handlers = _recording_handlers([])

        def boom(prof, **params):
            raise RuntimeError("index exploded")

        handlers["index"] = boom
        persona = self.reg.get("llm-none")
        planned = flow.plan_steps(persona, handlers, self.reg)
        prof = _profile()
        with self.assertRaises(RuntimeError):
            flow.execute_plan(prof, "p.json", planned, handlers, ui)
        prof.save.assert_called_once_with("p.json")  # partial progress kept

    def test_llm_verify_runs_judge_and_prune(self):
        ui = FakeUI(gates=["proceed"])
        with mock.patch("src.llm.calibration.format_report", return_value="B"):
            rc, calls, _ = self._run("llm-verify", ui)
        self.assertEqual(rc, 0)
        self.assertEqual(
            calls,
            ["scope", "measure", "scan", "calibrate", "judge", "prune", "summarize", "index"],
        )


class TestReadRecommendation(unittest.TestCase):
    def test_reads_scan_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            profile_path = os.path.join(tmp, "box.profile.json")
            with open(os.path.join(tmp, "box.profile.scan.json"), "w") as fh:
                json.dump({"recommended_persona": "llm-verify"}, fh)
            self.assertEqual(flow.read_recommendation(profile_path), "llm-verify")

    def test_missing_or_bad_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            profile_path = os.path.join(tmp, "box.profile.json")
            self.assertIsNone(flow.read_recommendation(profile_path))
            with open(os.path.join(tmp, "box.profile.scan.json"), "w") as fh:
                fh.write("{not json")
            self.assertIsNone(flow.read_recommendation(profile_path))


class TestShortResult(unittest.TestCase):
    def test_collapses_whitespace(self):
        self.assertEqual(flow.short_result("a\n  b\tc"), "a b c")

    def test_truncates_long_results(self):
        out = flow.short_result("x" * 500, limit=20)
        self.assertEqual(len(out), 20)
        self.assertTrue(out.endswith("…"))


if __name__ == "__main__":
    unittest.main()
