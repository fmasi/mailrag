"""Tests for the source-agnostic folder selection + local .eml source.

Stdlib-only so the TDD loop runs under plain `python3` on the host (the Azure
batch script pulls heavy deps and only runs in the devcontainer).
"""
import os
import tempfile
import unittest

from src.ingest import selection


def _make_tree(root, rel_files):
    for rel in rel_files:
        p = os.path.join(root, *rel.split("/"))
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as fh:
            fh.write("x")


class TestListEmlRelpaths(unittest.TestCase):
    def test_lists_only_eml_as_sorted_forward_slash_relpaths(self):
        with tempfile.TemporaryDirectory() as root:
            _make_tree(root, [
                "top.eml",
                "Inbox/Acme Corp/a.eml",
                "Inbox/Google/b.eml",
                "Acme Corp/Archive/c.eml",
                "Inbox/Acme Corp/notes.txt",  # non-eml ignored
            ])
            self.assertEqual(
                selection.list_eml_relpaths(root),
                [
                    "Acme Corp/Archive/c.eml",
                    "Inbox/Acme Corp/a.eml",
                    "Inbox/Google/b.eml",
                    "top.eml",
                ],
            )

    def test_extension_match_is_case_insensitive(self):
        with tempfile.TemporaryDirectory() as root:
            _make_tree(root, ["A.EML", "b.eml"])
            self.assertEqual(selection.list_eml_relpaths(root), ["A.EML", "b.eml"])


class TestNormalizePrefix(unittest.TestCase):
    def test_appends_trailing_slash(self):
        self.assertEqual(selection.normalize_prefix("Inbox"), "Inbox/")

    def test_keeps_empty_prefix(self):
        self.assertEqual(selection.normalize_prefix(""), "")

    def test_keeps_existing_trailing_slash(self):
        self.assertEqual(selection.normalize_prefix("Inbox/"), "Inbox/")

    def test_strips_surrounding_whitespace(self):
        self.assertEqual(selection.normalize_prefix("  Inbox  "), "Inbox/")


class TestMatchesRule(unittest.TestCase):
    def test_prefix_rule_matches_by_path_prefix(self):
        rule = {"type": "prefix", "value": "Inbox/Acme Corp/"}
        self.assertTrue(selection.matches_rule("Inbox/Acme Corp/a.eml", rule))
        self.assertFalse(selection.matches_rule("Inbox/Google/b.eml", rule))

    def test_direct_root_files_rule_excludes_nested(self):
        rule = {"type": "direct-root-files", "root": "Inbox/"}
        self.assertTrue(selection.matches_rule("Inbox/a.eml", rule))
        self.assertFalse(selection.matches_rule("Inbox/Sub/a.eml", rule))

    def test_container_root_rule_matches_only_top_level(self):
        rule = {"type": "container-root"}
        self.assertTrue(selection.matches_rule("top.eml", rule))
        self.assertFalse(selection.matches_rule("Inbox/a.eml", rule))

    def test_unknown_rule_type_raises(self):
        with self.assertRaises(ValueError):
            selection.matches_rule("x.eml", {"type": "bogus"})


class TestDiscoverStructure(unittest.TestCase):
    def test_tracks_roots_children_and_root_files(self):
        tree, has_root = selection.discover_structure([
            "top.eml",
            "Inbox/message.eml",
            "Google/direct.eml",
            "Google/All Mail/a.eml",
            "Google/Sent/b.eml",
        ])
        self.assertTrue(has_root)
        self.assertEqual(tree["Inbox/"]["children"], set())
        self.assertTrue(tree["Inbox/"]["has_direct_files"])
        self.assertTrue(tree["Google/"]["has_direct_files"])
        self.assertEqual(
            tree["Google/"]["children"],
            {"Google/All Mail/", "Google/Sent/"},
        )


class TestSelectEmlPaths(unittest.TestCase):
    def test_returns_absolute_paths_matching_any_rule(self):
        with tempfile.TemporaryDirectory() as root:
            _make_tree(root, [
                "Inbox/Acme Corp/a.eml",
                "Inbox/Google/b.eml",
                "Acme Corp/Archive/c.eml",
                "top.eml",
            ])
            rules = [
                {"type": "prefix", "value": "Inbox/Acme Corp/"},
                {"type": "prefix", "value": "Acme Corp/"},
            ]
            selected = selection.select_eml_paths(root, rules)
            self.assertTrue(selected and all(os.path.isabs(p) for p in selected))
            self.assertEqual(
                [os.path.relpath(p, root).replace(os.sep, "/") for p in selected],
                ["Acme Corp/Archive/c.eml", "Inbox/Acme Corp/a.eml"],
            )


class _FakePrompt:
    def __init__(self, value):
        self.value = value

    def ask(self):
        return self.value


class _FakeQuestionary:
    """Returns canned answers in order for confirm()/select() calls."""

    def __init__(self, answers):
        self.answers = iter(answers)

    def confirm(self, *args, **kwargs):
        return _FakePrompt(next(self.answers))

    def select(self, *args, **kwargs):
        return _FakePrompt(next(self.answers))


class TestPromptGuidedSelection(unittest.TestCase):
    def test_collects_specific_level2_folders_and_direct_files(self):
        folder_tree = {
            "Inbox/": {"children": {"Inbox/A/", "Inbox/B/", "Inbox/C/"}, "has_direct_files": True},
            "Spam/": {"children": set(), "has_direct_files": True},
        }
        # Inbox: "children" -> include direct files True -> A True, B False, C True; Spam: "skip"
        fake = _FakeQuestionary(["children", True, True, False, True, "skip"])
        rules = selection.prompt_guided_selection(
            folder_tree, has_container_root_files=False, questionary=fake
        )
        self.assertEqual(
            rules,
            [
                {"type": "direct-root-files", "root": "Inbox/"},
                {"type": "prefix", "value": "Inbox/A/"},
                {"type": "prefix", "value": "Inbox/C/"},
            ],
        )

    def test_include_whole_folder_uses_prefix_rule(self):
        folder_tree = {"Acme Corp/": {"children": {"Acme Corp/Archive/"}, "has_direct_files": False}}
        fake = _FakeQuestionary(["all"])
        rules = selection.prompt_guided_selection(
            folder_tree, has_container_root_files=False, questionary=fake
        )
        self.assertEqual(rules, [{"type": "prefix", "value": "Acme Corp/"}])

    def test_container_root_confirmation_adds_rule(self):
        fake = _FakeQuestionary([True])  # include container-root files
        rules = selection.prompt_guided_selection(
            {}, has_container_root_files=True, questionary=fake
        )
        self.assertEqual(rules, [{"type": "container-root"}])


if __name__ == "__main__":
    unittest.main()
