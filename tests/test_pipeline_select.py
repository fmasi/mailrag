import unittest
from unittest import mock

from src.profile import CorpusProfile


class TestSelectStage(unittest.TestCase):
    def test_writes_rules_to_profile(self):
        from src.pipeline import select

        prof = CorpusProfile(root="/r")
        rules = [{"type": "prefix", "value": "Inbox/"}]
        with (
            mock.patch("src.pipeline.select.list_eml_relpaths", return_value=["Inbox/a.eml"]),
            mock.patch("src.pipeline.select.discover_structure", return_value=({}, False)),
            mock.patch("src.pipeline.select.prompt_guided_selection", return_value=rules),
        ):
            out = select.run(prof, questionary=mock.Mock())
        self.assertEqual(out, rules)
        self.assertEqual(prof.selection_rules, rules)


if __name__ == "__main__":
    unittest.main()
