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

    def test_an_unknown_collection_has_no_scope(self):
        scoping.clear_cache()
        self.assertIsNone(scoping.scope_for_collection("nope"))

    def test_the_cache_notices_a_new_profile(self):
        d = os.environ["MAILRAG_PROFILE_DIR"]
        scoping.clear_cache()
        self.assertEqual(scoping.collection_profiles(), {})
        _write_profile(d, "late", "late-rag", "/tmp/root")
        self.assertIn("late-rag", scoping.collection_profiles())

    def test_the_cache_notices_a_recorded_manifest_mapping(self):
        """A manifest write must invalidate the cache even though it lands
        outside ``profile_dir()``.

        ``record_profile_for_collection`` (called by ``index`` and
        ``attachments build`` for a profile that need not live under
        ``$MAILRAG_PROFILE_DIR``) writes into ``$MAILRAG_HOME``, a directory
        the old cache key never looked at. A long-running server that warmed
        the cache before the write used to keep serving the pre-write mapping
        forever, because nothing under ``profile_dir()`` had changed.
        """
        scoping.clear_cache()
        self.assertEqual(scoping.collection_profiles(), {})  # warm the cache

        from src.onboard import record_profile_for_collection

        with tempfile.TemporaryDirectory() as elsewhere:
            profile_path = os.path.join(elsewhere, "outside.profile.json")
            with open(profile_path, "w") as fh:
                json.dump(
                    {"collection": "late-manifest-rag", "root": "/tmp/root", "selection_rules": []},
                    fh,
                )
            record_profile_for_collection("late-manifest-rag", profile_path)

            self.assertIn("late-manifest-rag", scoping.collection_profiles())

    def test_a_missing_manifest_dir_does_not_break_profile_discovery(self):
        """The defensive path in ``_manifest_paths()`` — mirrors
        ``test_an_unreadable_profile_does_not_break_the_others`` for the
        analogous profile-dir case.

        A server can start before ``$MAILRAG_HOME`` has ever been created
        (nothing has been onboarded yet), and the manifest signature must
        degrade to empty rather than crash discovery of the profiles that
        do exist under ``profile_dir()``.
        """
        d = os.environ["MAILRAG_PROFILE_DIR"]
        _write_profile(d, "good", "work-rag", "/tmp/root")
        with mock.patch.dict(os.environ, {"MAILRAG_HOME": "/nonexistent/mailrag-home"}):
            scoping.clear_cache()
            self.assertIn("work-rag", scoping.collection_profiles())


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
        from src.mcp_server.grep import _scoped_walk

        self.assertIsNone(_scoped_walk("not-a-known-corpus", "/tmp/somewhere"))

    def test_no_collection_means_no_scoping_attempt(self):
        from src.mcp_server.grep import _scoped_walk

        self.assertIsNone(_scoped_walk(None, None))


class TestScopeResolution(unittest.TestCase):
    """``scope_for_collection`` is the single answer to "which files, which root".

    Files and root used to be two lookups against the same profile — one cached
    with no invalidation, one re-read every call — so they could describe
    different corpora at the same moment.
    """

    def setUp(self):
        scoping.clear_cache()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def _corpus(self, name, needle_file):
        """A corpus directory holding one ``.eml`` that mentions "needle"."""
        path = os.path.join(self.tmp.name, name)
        os.makedirs(path)
        with open(os.path.join(path, needle_file), "w") as fh:
            fh.write(f"Subject: From {name}\nFrom: a@b.c\n\nthe needle is here\n")
        return path

    def _profile(self, collection, root):
        """Write ``collection``'s profile pointing at ``root``."""
        d = os.environ["MAILRAG_PROFILE_DIR"]
        path = os.path.join(d, f"{collection}.profile.json")
        with open(path, "w") as fh:
            json.dump(
                {
                    "collection": collection,
                    "root": root,
                    "selection_rules": [{"type": "container-root"}],
                },
                fh,
            )
        return path

    def test_the_scope_carries_both_the_files_and_the_root_they_came_from(self):
        corpus = self._corpus("old", "m1.eml")
        self._profile("scoped-rag", corpus)
        scoping.clear_cache()
        scope = scoping.scope_for_collection("scoped-rag")
        self.assertEqual(os.path.realpath(scope.root), os.path.realpath(corpus))
        self.assertEqual(len(scope.files), 1)
        for f in scope.files:
            self.assertTrue(os.path.realpath(f).startswith(os.path.realpath(scope.root)))

    def test_an_unchanged_profile_is_answered_from_the_cache(self):
        corpus = self._corpus("old", "m1.eml")
        self._profile("scoped-rag", corpus)
        scoping.clear_cache()
        first = scoping.scope_for_collection("scoped-rag")
        second = scoping.scope_for_collection("scoped-rag")
        # Same list object: resolution was memoised, not repeated.
        self.assertIs(first.files, second.files)

    def test_no_collection_named_means_nothing_to_scope(self):
        self.assertIsNone(scoping.scope_for_collection(None))
        self.assertIsNone(scoping.scope_for_collection(""))

    def test_an_unknown_collection_has_no_scope(self):
        self.assertIsNone(scoping.scope_for_collection("no-such-corpus"))

    def test_a_profile_that_cannot_be_stat_ed_gets_a_neutral_timestamp(self):
        """The mtime is a cache key, and a missing file is not a crash.

        A profile can vanish between the directory listing and the resolution
        that reads it; the stat must degrade to a value that simply never
        matches a real one, so the entry is re-resolved rather than trusted.
        """
        self.assertEqual(scoping._mtime(os.path.join(self.tmp.name, "gone.profile.json")), 0.0)

    def test_an_unreadable_profile_is_a_scoping_error_not_a_silent_default(self):
        """A recorded mapping can outlive a readable profile.

        Directory scanning skips a corrupt profile, but a manifest names one
        explicitly — and then the collection *is* known, its profile just
        cannot be read. Answering ``None`` there would report "unknown
        collection" for a profile sitting right where the manifest says.
        """
        from src.onboard import record_profile_for_collection

        d = os.environ["MAILRAG_PROFILE_DIR"]
        path = os.path.join(d, "broken-rag.profile.json")
        with open(path, "w") as fh:
            fh.write("{ not json")
        record_profile_for_collection("broken-rag", path)
        scoping.clear_cache()
        with self.assertRaises(ValueError) as ctx:
            scoping.scope_for_collection("broken-rag")
        msg = str(ctx.exception)
        self.assertIn("broken-rag", msg)
        self.assertIn(path, msg)

    def test_a_grep_reports_the_unreadable_profile_rather_than_scanning_everything(self):
        from src.mcp_server.grep import grep_email
        from src.onboard import record_profile_for_collection

        d = os.environ["MAILRAG_PROFILE_DIR"]
        path = os.path.join(d, "broken-rag.profile.json")
        with open(path, "w") as fh:
            fh.write("{ not json")
        record_profile_for_collection("broken-rag", path)
        scoping.clear_cache()
        with self.assertRaises(ValueError) as ctx:
            grep_email("needle", collection="broken-rag")
        self.assertIn("could not be read", str(ctx.exception))


class TestReOnboardingInvalidatesAWarmCache(unittest.TestCase):
    """The MCP server is long-lived; a re-onboard happens underneath it.

    Nothing in the onboarding path calls ``clear_cache``, so a warm process
    must notice the profile changing by itself. When it did not, the file list
    stayed on the previous corpus while the reported root moved to the new one
    — grep answered from the old corpus under the new corpus's name, which is
    the precise failure scoping exists to prevent.
    """

    def setUp(self):
        scoping.clear_cache()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def _corpus(self, name, filename):
        path = os.path.join(self.tmp.name, name)
        os.makedirs(path)
        with open(os.path.join(path, filename), "w") as fh:
            fh.write(f"Subject: From {name}\nFrom: a@b.c\n\nthe needle is here\n")
        return path

    def _write_scoped_profile(self, root, stamp):
        d = os.environ["MAILRAG_PROFILE_DIR"]
        path = os.path.join(d, "moving-rag.profile.json")
        with open(path, "w") as fh:
            json.dump(
                {
                    "collection": "moving-rag",
                    "root": root,
                    "selection_rules": [{"type": "container-root"}],
                },
                fh,
            )
        # Stamp the mtime explicitly: two writes inside one filesystem timestamp
        # tick would make this test pass or fail on clock resolution.
        os.utime(path, (stamp, stamp))
        return path

    def test_a_re_onboarded_collection_is_grepped_in_its_new_corpus(self):
        from src.mcp_server.grep import grep_email

        old = self._corpus("old_corpus", "old.eml")
        new = self._corpus("new_corpus", "new.eml")

        self._write_scoped_profile(old, stamp=1_000_000)
        scoping.clear_cache()
        first = grep_email("needle", collection="moving-rag")
        self.assertEqual(os.path.realpath(first["root"]), os.path.realpath(old))
        self.assertEqual([m["subject"] for m in first["matches"]], ["From old_corpus"])

        # Re-onboard: same collection, same profile path, new corpus root. The
        # cache is warm and nobody clears it.
        self._write_scoped_profile(new, stamp=2_000_000)
        second = grep_email("needle", collection="moving-rag")

        self.assertEqual(os.path.realpath(second["root"]), os.path.realpath(new))
        self.assertEqual([m["subject"] for m in second["matches"]], ["From new_corpus"])
        # The invariant either lookup could break on its own: every file the
        # scan actually read lives under the root the result names.
        self.assertTrue(second["matches"])
        for m in second["matches"]:
            self.assertTrue(
                os.path.realpath(m["path"]).startswith(os.path.realpath(second["root"])),
                f"{m['path']} is not under the reported root {second['root']}",
            )


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
