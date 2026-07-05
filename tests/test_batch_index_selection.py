"""Tests for guided selection and persisted checkpoint state helpers."""

import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

# The script lives in scripts/ which is not a package; add it to sys.path.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from batch_index_to_vector_store import (  # noqa: E402
    _build_checkpoint_state,
    _discover_blob_structure,
    _filter_blobs_by_selection,
    _normalize_prefix,
    _prompt_guided_selection,
    _read_checkpoint_state,
    _remove_checkpoint,
    _render_selection_tree,
    _write_checkpoint_state,
)


class TestSelectionHelpers(unittest.TestCase):
    """Selection helpers should preserve the user's intended blob scope."""

    def test_normalize_prefix_appends_trailing_slash(self):
        self.assertEqual(_normalize_prefix("Inbox"), "Inbox/")

    def test_normalize_prefix_keeps_empty_prefix(self):
        self.assertEqual(_normalize_prefix(""), "")

    def test_discover_blob_structure_tracks_roots_and_children(self):
        tree, has_container_root_files = _discover_blob_structure(
            [
                "top-level.eml",
                "Inbox/message.eml",
                "Google/direct.eml",
                "Google/All Mail/a.eml",
                "Google/Sent/b.eml",
            ]
        )

        self.assertTrue(has_container_root_files)
        self.assertEqual(tree["Inbox/"]["children"], set())
        self.assertTrue(tree["Inbox/"]["has_direct_files"])
        self.assertTrue(tree["Google/"]["has_direct_files"])
        self.assertEqual(
            tree["Google/"]["children"],
            {"Google/All Mail/", "Google/Sent/"},
        )

    def test_filter_blobs_by_selection_supports_multiple_rule_types(self):
        blobs = [
            SimpleNamespace(name="top-level.eml"),
            SimpleNamespace(name="Inbox/direct.eml"),
            SimpleNamespace(name="Inbox/Sub/nested.eml"),
            SimpleNamespace(name="Google/All Mail/a.eml"),
            SimpleNamespace(name="Google/Sent/b.eml"),
        ]
        selection_rules = [
            {"type": "container-root"},
            {"type": "direct-root-files", "root": "Inbox/"},
            {"type": "prefix", "value": "Google/All Mail/"},
        ]

        matched = _filter_blobs_by_selection(blobs, selection_rules)

        self.assertEqual(
            [blob.name for blob in matched],
            [
                "top-level.eml",
                "Inbox/direct.eml",
                "Google/All Mail/a.eml",
            ],
        )

    def test_render_selection_tree_shows_partial_root_and_children(self):
        folder_tree = {
            "Google/": {
                "children": {"Google/All Mail/", "Google/Sent/"},
                "has_direct_files": True,
            },
            "Inbox/": {
                "children": set(),
                "has_direct_files": True,
            },
        }

        lines = _render_selection_tree(
            folder_tree,
            has_container_root_files=True,
            selection_rules=[
                {"type": "container-root"},
                {"type": "prefix", "value": "Google/All Mail/"},
                {"type": "direct-root-files", "root": "Inbox/"},
            ],
        )

        self.assertEqual(
            lines,
            [
                "  [x] selected  [~] partial  [ ] skipped",
                "  [x] [container root files]",
                "  [~] Google/",
                "      |-- [ ] Google/ [direct files only]",
                "      |-- [x] Google/All Mail/",
                "      `-- [ ] Google/Sent/",
                "  [x] Inbox/",
                "      `-- [x] Inbox/ [direct files only]",
            ],
        )


class _FakePrompt:
    def __init__(self, value):
        self.value = value

    def ask(self):
        return self.value


class _FakeQuestionary:
    def __init__(self, answers):
        self.answers = iter(answers)

    def confirm(self, *args, **kwargs):
        return _FakePrompt(next(self.answers))

    def select(self, *args, **kwargs):
        return _FakePrompt(next(self.answers))


class TestGuidedPromptFlow(unittest.TestCase):
    def test_prompt_guided_selection_can_collect_multiple_level2_folders(self):
        folder_tree = {
            "Inbox/": {
                "children": {"Inbox/A/", "Inbox/B/", "Inbox/C/"},
                "has_direct_files": True,
            },
            "Spam/": {"children": set(), "has_direct_files": True},
        }
        fake_questionary = _FakeQuestionary(
            [
                "children",
                True,
                True,
                False,
                True,
                "skip",
            ]
        )

        rules = _prompt_guided_selection(
            folder_tree,
            has_container_root_files=False,
            questionary=fake_questionary,
        )

        self.assertEqual(
            rules,
            [
                {"type": "direct-root-files", "root": "Inbox/"},
                {"type": "prefix", "value": "Inbox/A/"},
                {"type": "prefix", "value": "Inbox/C/"},
            ],
        )


class TestCheckpointState(unittest.TestCase):
    """Checkpoint state should preserve selection and resume position."""

    def test_checkpoint_state_round_trip(self):
        state = _build_checkpoint_state(
            "guided",
            [{"type": "prefix", "value": "Google/All Mail/"}],
        )
        state["last_blob_name"] = "Google/All Mail/a.eml"

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = os.path.join(temp_dir, "checkpoint.json")
            with patch("batch_index_to_vector_store.CHECKPOINT_FILE", checkpoint_path):
                _write_checkpoint_state(state)
                self.assertEqual(_read_checkpoint_state(), state)
                _remove_checkpoint()
                self.assertIsNone(_read_checkpoint_state())

    def test_read_checkpoint_state_supports_legacy_plain_text(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = os.path.join(temp_dir, "checkpoint.txt")
            with open(checkpoint_path, "w") as handle:
                handle.write("Google/All Mail/example.eml")

            with patch("batch_index_to_vector_store.CHECKPOINT_FILE", checkpoint_path):
                self.assertEqual(
                    _read_checkpoint_state(),
                    {
                        "version": 1,
                        "last_blob_name": "Google/All Mail/example.eml",
                    },
                )


if __name__ == "__main__":
    unittest.main()
