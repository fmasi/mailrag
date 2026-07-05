"""Smoke test for the TUI screenshot generator (``scripts/gen_tui_screenshots.py``).

Runs the generator headlessly into a temp directory (it fakes every handler and
uses a synthetic mailbox, so no Qdrant/LLM/real mail is touched) and asserts it
produces the six expected, non-empty SVGs. This keeps the committed docs
screenshots reproducible: if the TUI drifts enough that the generator can no
longer drive it, this test fails loudly.
"""

import asyncio
import importlib.util
import os
import tempfile
import unittest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_GEN_PATH = os.path.join(_REPO_ROOT, "scripts", "gen_tui_screenshots.py")


def _load_generator():
    spec = importlib.util.spec_from_file_location("gen_tui_screenshots", _GEN_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestGenTuiScreenshots(unittest.TestCase):
    def test_generates_six_non_empty_svgs(self):
        gen = _load_generator()
        with tempfile.TemporaryDirectory() as out:
            written = asyncio.run(gen.generate(out))
            # One SVG per stage, in flow order.
            self.assertEqual(
                [os.path.basename(p) for p in written],
                [f"{stage}.svg" for stage in gen.STAGES],
            )
            for stage in gen.STAGES:
                path = os.path.join(out, f"{stage}.svg")
                self.assertTrue(os.path.exists(path), f"missing {stage}.svg")
                self.assertGreater(os.path.getsize(path), 0, f"empty {stage}.svg")
                with open(path, encoding="utf-8") as fh:
                    self.assertIn("<svg", fh.read(200))


if __name__ == "__main__":
    unittest.main()
