"""Scoping: a session researching one corpus must not be shown another.

Not access control — the caller may ask for two corpora deliberately. What must
not happen is being handed the wrong one silently. Three mechanisms:

* ambiguity refuses instead of picking a default,
* grep walks only the files its collection's profile selects,
* every response says which corpus answered it.
"""

import json
import os
import tempfile
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

    def test_the_refusal_survives_a_missing_default_corpus_root(self):
        """A scoping refusal must not be masked by an unrelated config error.

        ``$MAILRAG_EML_ROOT`` is irrelevant to a scoped grep — the files come
        from the profile — but grep used to resolve it first, so on any machine
        without the default root (CI, a fresh checkout) naming an unknown
        collection reported "corpus not found" and the caller never learned that
        scoping was what actually failed.
        """
        from src.mcp_server.grep import grep_email

        scoping.clear_cache()
        with mock.patch.dict(os.environ, {"MAILRAG_EML_ROOT": "/nonexistent/corpus/root"}):
            with self.assertRaises(ValueError) as ctx:
                grep_email("anything", collection="not-a-known-corpus")
        msg = str(ctx.exception)
        self.assertIn("not-a-known-corpus", msg)
        self.assertIn("whole raw corpus", msg)
        self.assertNotIn("corpus not found", msg)

    def test_a_scoped_grep_ignores_a_missing_default_root(self):
        """Scoping supplies the files, so the default root need not exist.

        Profiles may point anywhere; requiring ``$MAILRAG_EML_ROOT`` to be a real
        directory broke scoped greps that were never going to read it.
        """
        from src.mcp_server.grep import grep_email

        d = os.environ["MAILRAG_PROFILE_DIR"]
        with tempfile.TemporaryDirectory() as corpus:
            with open(os.path.join(corpus, "m1.eml"), "w") as fh:
                fh.write("Subject: Greeting\nFrom: a@b.c\n\nthe needle is here\n")
            path = os.path.join(d, "scoped.profile.json")
            with open(path, "w") as fh:
                json.dump(
                    {
                        "collection": "scoped-rag",
                        "root": corpus,
                        "selection_rules": [{"type": "container-root"}],
                    },
                    fh,
                )
            scoping.clear_cache()
            with mock.patch.dict(os.environ, {"MAILRAG_EML_ROOT": "/nonexistent/corpus/root"}):
                res = grep_email("needle", collection="scoped-rag")
            self.assertTrue(res["scoped"])
            self.assertEqual(len(res["matches"]), 1)
            self.assertEqual(res["matches"][0]["subject"], "Greeting")
            # The reported root is the one actually walked, not the default.
            self.assertEqual(os.path.realpath(res["root"]), os.path.realpath(corpus))

    def test_an_unscoped_grep_still_reports_a_missing_root(self):
        """Removing the eager check must not swallow the genuine config error."""
        from src.mcp_server.grep import grep_email

        with mock.patch.dict(os.environ, {"MAILRAG_EML_ROOT": "/nonexistent/corpus/root"}):
            with self.assertRaises(ValueError) as ctx:
                grep_email("anything")
        self.assertIn("MAILRAG_EML_ROOT", str(ctx.exception))

    def test_an_explicit_root_bypasses_scoping(self):
        # The escape hatch: a caller naming a directory has said what it wants.
        from src.mcp_server.grep import _scoped_files

        self.assertIsNone(_scoped_files("not-a-known-corpus", "/tmp/somewhere"))

    def test_no_collection_means_no_scoping_attempt(self):
        from src.mcp_server.grep import _scoped_files

        self.assertIsNone(_scoped_files(None, None))


if __name__ == "__main__":
    unittest.main()


class TestManifestProfileMapping(unittest.TestCase):
    """The collection→profile mapping should be recorded, not re-derived.

    Scoping a grep means walking the files a collection's profile selects, so
    that link is load-bearing. Inferring it by scanning a directory for
    `*.profile.json` works until a profile moves, is renamed, or two of them
    name the same collection — so the collection records its own provenance.
    """

    def setUp(self):
        scoping.clear_cache()

    def test_a_recorded_mapping_is_found(self):
        from src.onboard import manifest_profile_paths, record_profile_for_collection

        d = os.environ["MAILRAG_PROFILE_DIR"]
        path = _write_profile(d, "w", "work-rag", "/tmp/root")
        record_profile_for_collection("work-rag", path)
        self.assertEqual(manifest_profile_paths().get("work-rag"), path)

    def test_a_recorded_mapping_wins_over_directory_scanning(self):
        from src.onboard import record_profile_for_collection

        d = os.environ["MAILRAG_PROFILE_DIR"]
        scanned = _write_profile(d, "scanned", "dual-rag", "/tmp/root")
        recorded = _write_profile(d, "recorded", "dual-rag", "/tmp/root")
        record_profile_for_collection("dual-rag", recorded)
        scoping.clear_cache()
        self.assertEqual(scoping.collection_profiles()["dual-rag"], recorded)
        self.assertNotEqual(scoping.collection_profiles()["dual-rag"], scanned)

    def test_recording_merges_rather_than_clobbering_a_manifest(self):
        import json as _json

        from src.onboard import manifest_dir, record_profile_for_collection

        d = os.environ["MAILRAG_PROFILE_DIR"]
        path = _write_profile(d, "w", "keep-rag", "/tmp/root")
        m = manifest_dir()
        m.mkdir(parents=True, exist_ok=True)
        (m / "keep-rag.json").write_text(_json.dumps({"collection": "keep-rag", "chunks": 42}))
        record_profile_for_collection("keep-rag", path)
        data = _json.loads((m / "keep-rag.json").read_text())
        self.assertEqual(data["chunks"], 42)  # reproducibility record survives
        self.assertEqual(data["profile_path"], path)

    def test_a_stale_mapping_to_a_missing_profile_is_ignored(self):
        from src.onboard import manifest_profile_paths, record_profile_for_collection

        d = os.environ["MAILRAG_PROFILE_DIR"]
        path = _write_profile(d, "gone", "gone-rag", "/tmp/root")
        record_profile_for_collection("gone-rag", path)
        os.unlink(path)
        self.assertNotIn("gone-rag", manifest_profile_paths())

    def test_non_string_inputs_are_ignored(self):
        from src.onboard import manifest_profile_paths, record_profile_for_collection

        record_profile_for_collection(object(), object())
        self.assertEqual(manifest_profile_paths(), {})
