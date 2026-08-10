import unittest
from unittest import mock

from src import cli


class TestRunVerb(unittest.TestCase):
    def test_unknown_persona_exits_2(self):
        self.assertEqual(cli.main(["run", "--profile", "p.json", "--persona", "nope"]), 2)

    def test_llm_verify_is_runnable(self):
        # judge + prune now exist, so llm-verify dispatches end-to-end.
        prof = mock.Mock()
        recorded = {}

        def fake_run_persona(profile, persona, handlers, **kw):
            recorded["persona"] = persona.name
            recorded["handler_keys"] = set(handlers)
            return []

        with (
            mock.patch("src.cli.CorpusProfile.load", return_value=prof),
            mock.patch("src.cli.persona_executor.run_persona", side_effect=fake_run_persona),
        ):
            rc = cli.main(["run", "--profile", "p.json", "--persona", "llm-verify", "--model", "m"])
        self.assertEqual(rc, 0)
        self.assertEqual(recorded["persona"], "llm-verify")
        self.assertIn("judge", recorded["handler_keys"])
        self.assertIn("prune", recorded["handler_keys"])

    def test_llm_all_without_model_exits_2(self):
        rc = cli.main(["run", "--profile", "p.json", "--persona", "llm-all"])
        self.assertEqual(rc, 2)

    def test_llm_none_dispatches_in_order(self):
        prof = mock.Mock()
        recorded = {}

        def fake_run_persona(profile, persona, handlers, **kw):
            recorded["persona"] = persona.name
            recorded["handler_keys"] = set(handlers)
            return []

        with (
            mock.patch("src.cli.CorpusProfile.load", return_value=prof),
            mock.patch("src.cli.persona_executor.run_persona", side_effect=fake_run_persona),
        ):
            rc = cli.main(["run", "--profile", "p.json", "--persona", "llm-none"])
        self.assertEqual(rc, 0)
        self.assertEqual(recorded["persona"], "llm-none")
        prof.save.assert_called_once_with("p.json")


class TestWizardVerb(unittest.TestCase):
    def test_wizard_defaults_to_the_tui(self):
        with mock.patch("src.tui.app.run_tui", return_value=0) as rt:
            rc = cli.main(["wizard", "--profile", "p.json", "--model", "m"])
        self.assertEqual(rc, 0)
        rt.assert_called_once_with("p.json", model="m", limit=None)

    def test_wizard_classic_routes_to_run_wizard(self):
        with mock.patch("src.cli.persona_wizard.run_wizard", return_value=0) as rw:
            rc = cli.main(["wizard", "--profile", "p.json", "--model", "m", "--classic"])
        self.assertEqual(rc, 0)
        rw.assert_called_once_with("p.json", model="m", limit=None)

    def test_wizard_passes_limit(self):
        with mock.patch("src.tui.app.run_tui", return_value=0) as rt:
            cli.main(["wizard", "--profile", "p.json", "--model", "m", "--limit", "20"])
        self.assertEqual(rt.call_args.kwargs["limit"], 20)

    def test_wizard_classic_passes_limit(self):
        with mock.patch("src.cli.persona_wizard.run_wizard", return_value=0) as rw:
            cli.main(
                ["wizard", "--profile", "p.json", "--model", "m", "--limit", "20", "--classic"]
            )
        self.assertEqual(rw.call_args.kwargs["limit"], 20)


class TestRunLimit(unittest.TestCase):
    def test_run_passes_limit_to_build_handlers(self):
        prof = mock.Mock()
        handlers = {
            v: (lambda p, **k: None) for v in ("scope", "measure", "scan", "tag", "prune", "index")
        }
        with (
            mock.patch("src.cli.CorpusProfile.load", return_value=prof),
            mock.patch("src.cli.persona_runner.build_handlers", return_value=handlers) as bh,
            mock.patch("src.cli.persona_executor.run_persona", return_value=[]),
        ):
            rc = cli.main(["run", "--profile", "p.json", "--persona", "llm-none", "--limit", "15"])
        self.assertEqual(rc, 0)
        self.assertEqual(bh.call_args.kwargs["limit"], 15)


class TestBuildHandlers(unittest.TestCase):
    def test_llm_step_without_a_model_fails_with_an_actionable_error(self):
        """An LLM verb reached without a model must say so, here and clearly.

        `model` is optional because non-LLM personas do not need one, and both
        callers (wizard and TUI) prompt before an LLM verb — so the invariant
        held by convention across two files with nothing enforcing it. Without
        this guard a caller that forgot passed None all the way into the LLM
        client, which fails far from the cause.
        """
        from src.persona.runner import build_handlers

        handlers = build_handlers(profile_path="p.json", model=None)
        for verb in ("calibrate", "summarize", "judge"):
            with self.subTest(verb=verb):
                with self.assertRaises(ValueError) as ctx:
                    handlers[verb](mock.Mock())
                msg = str(ctx.exception)
                self.assertIn("needs an LLM model", msg)
                # The message must name a way out, not just state the problem.
                self.assertIn("RAG_LLM_MODEL", msg)

    def test_non_llm_steps_do_not_require_a_model(self):
        """The guard must not spread to verbs that never call an LLM.

        Pins the other direction: `model=None` is a legitimate configuration,
        so building the handler map and reaching a non-LLM verb stays valid.
        """
        from src.persona.runner import build_handlers

        handlers = build_handlers(profile_path="p.json", model=None)
        with mock.patch("src.pipeline.select.run", return_value={"kept": 1}) as run:
            out = handlers["scope"](mock.Mock())
        run.assert_called_once()
        self.assertEqual(out, {"kept": 1})

    def test_model_reaches_the_llm_stage_when_supplied(self):
        """The happy path still forwards the model rather than swallowing it."""
        from src.persona.runner import build_handlers

        handlers = build_handlers(profile_path="p.json", model="qwen3-8b")
        with mock.patch("src.pipeline.pass2.run", return_value={"done": 2}) as run:
            handlers["summarize"](mock.Mock())
        self.assertEqual(run.call_args.kwargs["model"], "qwen3-8b")

    def test_exposes_implemented_verbs_only(self):
        from src.persona.runner import build_handlers

        keys = set(build_handlers(profile_path="p.json", model="m"))
        self.assertEqual(
            keys,
            {
                "scope",
                "measure",
                "tag",
                "scan",
                "calibrate",
                "summarize",
                "index",
                "judge",
                "prune",
            },
        )


if __name__ == "__main__":
    unittest.main()
