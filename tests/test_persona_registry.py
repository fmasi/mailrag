import textwrap
import unittest

from src.persona.registry import Persona, load_registry


class TestPersonaRegistry(unittest.TestCase):
    def test_loads_shipped_registry(self):
        reg = load_registry()
        self.assertIn("llm-none", reg.names())
        self.assertIn("llm-verify", reg.names())
        self.assertIn("llm-all", reg.names())
        p = reg.get("llm-all")
        self.assertIsInstance(p, Persona)
        self.assertEqual(p.label, "Full (LLM on everything)")
        verbs = [s.verb for s in p.steps]
        self.assertEqual(verbs, ["scope", "measure", "calibrate", "summarize", "prune", "index"])
        # params parsed from {verb: {..}} steps
        summarize = next(s for s in p.steps if s.verb == "summarize")
        self.assertEqual(summarize.params, {"target": "all"})
        prune = next(s for s in p.steps if s.verb == "prune")
        self.assertEqual(prune.params, {"from": "summarize"})

    def test_string_and_dict_steps_both_parse(self):
        reg = load_registry()
        none = reg.get("llm-none")
        scan = next(s for s in none.steps if s.verb == "scan")
        self.assertEqual(scan.params, {"optional": True})
        scope = next(s for s in none.steps if s.verb == "scope")
        self.assertEqual(scope.params, {})

    def test_unknown_verb_in_step_raises(self):
        bad = textwrap.dedent("""
            verbs:
              scope: {does: x, cost: free}
            personas:
              broken:
                label: Broken
                steps: [scope, nonexistent_verb]
        """)
        with self.assertRaises(ValueError):
            load_registry(text=bad)

    def test_get_unknown_persona_raises(self):
        with self.assertRaises(ValueError):
            load_registry().get("does-not-exist")

    def test_verb_metadata_available(self):
        reg = load_registry()
        self.assertEqual(reg.verb_info("summarize").cost, "big-llm")
        self.assertIn("Qdrant", reg.verb_info("index").does)


if __name__ == "__main__":
    unittest.main()
