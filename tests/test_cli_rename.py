import argparse
import io
import unittest
from contextlib import redirect_stdout

from src import cli

# (new canonical name, old hidden alias)
PAIRS = [
    ("scope", "select"),
    ("measure", "profile"),
    ("tag", "pass1"),
    ("scan", "explore"),
    ("summarize", "pass2"),
    ("index", "build"),
    ("ask", "query"),
]


def _subparsers(parser):
    return [a for a in parser._actions if isinstance(a, argparse._SubParsersAction)][0]


class TestVerbRename(unittest.TestCase):
    def test_new_and_old_names_resolve_to_same_handler(self):
        choices = _subparsers(cli.build_parser()).choices
        for new, old in PAIRS:
            self.assertIn(new, choices, f"missing canonical verb {new}")
            self.assertIn(old, choices, f"missing alias {old}")
            self.assertIs(
                choices[new].get_default("func"),
                choices[old].get_default("func"),
                f"{old} must dispatch to the same handler as {new}",
            )

    def test_old_names_hidden_from_help(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            try:
                cli.build_parser().parse_args(["--help"])
            except SystemExit:
                pass
        help_text = buf.getvalue()
        self.assertIn("scan", help_text)
        self.assertNotIn("explore", help_text)  # alias suppressed
        self.assertNotIn("pass2", help_text)
        self.assertIn("summarize", help_text)

    def test_unchanged_verbs_present(self):
        choices = _subparsers(cli.build_parser()).choices
        for verb in ("onboard", "calibrate"):
            self.assertIn(verb, choices)


if __name__ == "__main__":
    unittest.main()
