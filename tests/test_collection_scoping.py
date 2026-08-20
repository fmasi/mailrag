"""Scoping: a session researching one corpus must not be shown another.

Not access control — the caller may ask for two corpora deliberately. What must
not happen is being handed the wrong one silently. Three mechanisms:

* ambiguity refuses instead of picking a default,
* grep walks only the files its collection's profile selects,
* every response says which corpus answered it.
"""

import json
import os
import unittest
from unittest import mock

from src.mcp_server import scoping, server


def _write_profile(dirpath, name, collection, root):
    path = os.path.join(dirpath, f"{name}.profile.json")
    with open(path, "w") as fh:
        json.dump({"collection": collection, "root": root, "selection_rules": []}, fh)
    return path


class TestAmbiguityRefuses(unittest.TestCase):
    def setUp(self):
        os.environ.pop("MAILRAG_COLLECTION", None)
        scoping.clear_cache()

    def test_two_corpora_and_no_choice_is_an_error_naming_both(self):
        d = os.environ["MAILRAG_PROFILE_DIR"]
        _write_profile(d, "a", "work-rag", "/tmp/root")
        _write_profile(d, "b", "personal-rag", "/tmp/root")
        scoping.clear_cache()
        with self.assertRaises(ValueError) as ctx:
            server.resolve_collection()
        msg = str(ctx.exception)
        self.assertIn("work-rag", msg)
        self.assertIn("personal-rag", msg)

    def test_a_single_corpus_still_resolves_without_being_named(self):
        d = os.environ["MAILRAG_PROFILE_DIR"]
        _write_profile(d, "only", "solo-rag", "/tmp/root")
        scoping.clear_cache()
        self.assertEqual(server.resolve_collection(), "solo-rag")

    def test_an_explicit_choice_always_wins(self):
        d = os.environ["MAILRAG_PROFILE_DIR"]
        _write_profile(d, "a", "work-rag", "/tmp/root")
        _write_profile(d, "b", "personal-rag", "/tmp/root")
        scoping.clear_cache()
        self.assertEqual(server.resolve_collection("personal-rag"), "personal-rag")

    def test_the_env_default_still_wins(self):
        d = os.environ["MAILRAG_PROFILE_DIR"]
        _write_profile(d, "a", "work-rag", "/tmp/root")
        _write_profile(d, "b", "personal-rag", "/tmp/root")
        scoping.clear_cache()
        with mock.patch.dict(os.environ, {"MAILRAG_COLLECTION": "work-rag"}):
            self.assertEqual(server.resolve_collection(), "work-rag")


class TestProfileDiscovery(unittest.TestCase):
    def setUp(self):
        scoping.clear_cache()

    def test_profiles_are_indexed_by_the_collection_they_name(self):
        d = os.environ["MAILRAG_PROFILE_DIR"]
        _write_profile(d, "w", "work-rag", "/tmp/root")
        scoping.clear_cache()
        self.assertEqual(sorted(scoping.collection_profiles()), ["work-rag"])

    def test_an_unreadable_profile_does_not_break_the_others(self):
        d = os.environ["MAILRAG_PROFILE_DIR"]
        _write_profile(d, "good", "work-rag", "/tmp/root")
        with open(os.path.join(d, "broken.profile.json"), "w") as fh:
            fh.write("{ not json")
        scoping.clear_cache()
        self.assertIn("work-rag", scoping.collection_profiles())

    def test_an_unknown_collection_has_no_file_list(self):
        scoping.clear_cache()
        self.assertIsNone(scoping.files_for_collection("nope"))

    def test_the_cache_notices_a_new_profile(self):
        d = os.environ["MAILRAG_PROFILE_DIR"]
        scoping.clear_cache()
        self.assertEqual(scoping.collection_profiles(), {})
        _write_profile(d, "late", "late-rag", "/tmp/root")
        self.assertIn("late-rag", scoping.collection_profiles())


class TestGrepRefusesWhenItCannotScope(unittest.TestCase):
    """Falling back to the whole root would do the exact thing scoping prevents."""

    def test_naming_an_unscopable_collection_raises(self):
        from src.mcp_server.grep import grep_email

        scoping.clear_cache()
        with self.assertRaises(ValueError) as ctx:
            grep_email("anything", collection="not-a-known-corpus")
        msg = str(ctx.exception)
        self.assertIn("not-a-known-corpus", msg)
        self.assertIn("whole raw corpus", msg)

    def test_an_explicit_root_bypasses_scoping(self):
        # The escape hatch: a caller naming a directory has said what it wants.
        from src.mcp_server.grep import _scoped_files

        self.assertIsNone(_scoped_files("not-a-known-corpus", "/tmp/somewhere"))

    def test_no_collection_means_no_scoping_attempt(self):
        from src.mcp_server.grep import _scoped_files

        self.assertIsNone(_scoped_files(None, None))


if __name__ == "__main__":
    unittest.main()
