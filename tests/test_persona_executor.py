import unittest

from src.persona.registry import load_registry
from src.persona.executor import run_persona, missing_handlers, StepResult


class TestPersonaExecutor(unittest.TestCase):
    def setUp(self):
        self.reg = load_registry()

    def test_runs_steps_in_order_with_params(self):
        calls = []

        def make(verb):
            def handler(profile, **params):
                calls.append((verb, params))
                return f"{verb}-ok"
            return handler

        persona = self.reg.get("llm-all")
        handlers = {v: make(v) for v in
                    ("scope", "measure", "calibrate", "summarize", "index")}
        results = run_persona("PROFILE", persona, handlers)

        self.assertEqual([v for v, _ in calls],
                         ["scope", "measure", "calibrate", "summarize", "index"])
        self.assertEqual(dict(calls)["summarize"], {"target": "all"})
        self.assertTrue(all(isinstance(r, StepResult) for r in results))
        self.assertEqual(results[3].result, "summarize-ok")

    def test_missing_required_handler_is_reported(self):
        persona = self.reg.get("llm-verify")  # needs judge (engine, not shipped)
        handlers = {v: (lambda profile, **k: None) for v in
                    ("scope", "measure", "scan", "calibrate", "summarize", "index")}
        miss = missing_handlers(persona, handlers)
        self.assertIn("judge", miss)
        with self.assertRaises(ValueError):
            run_persona("P", persona, handlers)

    def test_optional_step_skipped_when_handler_absent(self):
        calls = []
        persona = self.reg.get("llm-none")  # scan is {optional: true}
        handlers = {v: (lambda profile, _v=v, **k: calls.append(_v)) for v in
                    ("scope", "measure", "tag", "index")}  # no scan
        self.assertEqual(missing_handlers(persona, handlers), [])  # optional ignored
        run_persona("P", persona, handlers)
        self.assertNotIn("scan", calls)
        self.assertEqual(calls, ["scope", "measure", "tag", "index"])

    def test_on_step_callback_fires(self):
        seen = []
        persona = self.reg.get("llm-none")
        handlers = {v: (lambda profile, **k: None) for v in
                    ("scope", "measure", "scan", "tag", "index")}
        run_persona("P", persona, handlers, on_step=lambda step: seen.append(step.verb))
        self.assertEqual(seen[0], "scope")
        self.assertIn("scan", seen)


if __name__ == "__main__":
    unittest.main()
