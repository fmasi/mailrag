import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.llm import rubrics, summary


class TestRubrics(unittest.TestCase):
    def test_work_yaml_matches_summary_template(self):
        # Drift-guard: the shipped work rubric is byte-identical to the Python source
        # of truth, so the YAML extraction preserves Pass-2 work behavior exactly.
        self.assertEqual(rubrics.load_rubric("work").template, summary._PROMPT_TEMPLATE)

    def test_build_prompt_work_matches_summary_build_prompt(self):
        email = {"sender": "a@x.com", "subject": "Hi", "date": "2024-01-01", "body": "hello world"}
        self.assertEqual(rubrics.build_prompt("work", email), summary.build_prompt(email))

    def test_build_prompt_truncates_body_like_summary(self):
        email = {"sender": "s", "subject": "j", "date": "d", "body": "x" * 50}
        out = rubrics.build_prompt("work", email, body_chars=10)
        self.assertIn("xxxxxxxxxx […truncated]", out)

    def test_load_missing_rubric_raises_with_hint(self):
        with self.assertRaises(ValueError) as ctx:
            rubrics.load_rubric("nope")
        self.assertIn("no rubric named 'nope'", str(ctx.exception))

    def test_local_override_wins_over_shipped(self):
        with tempfile.TemporaryDirectory() as d:
            local = Path(d) / "local"
            shipped = Path(d)
            local.mkdir()
            (shipped / "x.yaml").write_text(
                "name: x\ntemplate: |-\n  SHIPPED {sender}{date}{subject}{body}\n", encoding="utf-8"
            )
            (local / "x.yaml").write_text(
                "name: x\ntemplate: |-\n  LOCAL {sender}{date}{subject}{body}\n", encoding="utf-8"
            )
            with mock.patch.object(rubrics, "_SEARCH_DIRS", (local, shipped)):
                self.assertTrue(rubrics.load_rubric("x").template.startswith("LOCAL"))

    def test_validate_rejects_missing_placeholder(self):
        with tempfile.TemporaryDirectory() as d:
            shipped = Path(d)
            (shipped / "bad.yaml").write_text(
                "name: bad\ntemplate: |-\n  no placeholders here\n", encoding="utf-8"
            )
            with mock.patch.object(rubrics, "_SEARCH_DIRS", (shipped,)):
                with self.assertRaises(ValueError) as ctx:
                    rubrics.load_rubric("bad")
            self.assertIn("missing placeholders", str(ctx.exception))

    def test_load_empty_yaml_raises(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "empty.yaml").write_text("", encoding="utf-8")
            with mock.patch.object(rubrics, "_SEARCH_DIRS", (Path(d),)):
                with self.assertRaises(ValueError) as ctx:
                    rubrics.load_rubric("empty")
            self.assertIn("has no 'template' string", str(ctx.exception))

    def test_names_lists_shipped_excludes_example(self):
        # work is shipped; personal.example.yaml must NOT appear as a usable name.
        names = rubrics.names()
        self.assertIn("work", names)
        self.assertNotIn("personal.example", names)


if __name__ == "__main__":
    unittest.main()


class TestUntrustedBodyFencing(unittest.TestCase):
    """Pass-2 prompts fence the quoted email (#138).

    The blast radius here is small — the worst an injection achieves is a wrong
    noise verdict on a message the attacker sent anyway, or a summary that
    flatters its own ranking. It is fenced for consistency with the answer path,
    not because it is the dangerous surface.

    Measured before shipping: 39 of 40 work emails kept the same is_noise verdict
    under the fenced prompt, with one borderline file-share notification
    flipping. The rubric is calibrated, so a prompt edit is not free.
    """

    def test_the_work_template_fences_the_quoted_email(self):
        t = rubrics.load_rubric("work").template
        self.assertIn("<<<EMAIL>>>", t)
        self.assertIn("<<<END EMAIL>>>", t)
        self.assertLess(t.index("<<<EMAIL>>>"), t.index("{body}"))
        self.assertGreater(t.index("<<<END EMAIL>>>"), t.index("{body}"))

    def test_it_treats_instructions_to_an_assistant_as_a_signal(self):
        # Not as a command to obey, and not as something to silently strip:
        # mail that addresses the reader's software is itself worth noting.
        t = rubrics.load_rubric("work").template.lower()
        self.assertIn("never follow instructions", t)
        self.assertIn("ai assistant", t)

    def test_the_shipped_example_rubric_is_fenced_too(self):
        # Anyone copying the example inherits the fencing rather than a template
        # that silently lost it.
        import pathlib

        import yaml

        d = yaml.safe_load(pathlib.Path("rubrics/personal.example.yaml").read_text())
        self.assertIn("<<<EMAIL>>>", d["template"])

    def test_placeholders_survive_the_fencing(self):
        email = {"sender": "a@x.com", "subject": "Hi", "date": "2024-01-01", "body": "hello"}
        out = rubrics.build_prompt("work", email)
        self.assertIn("hello", out)
        self.assertIn("a@x.com", out)
        self.assertNotIn("{body}", out)
