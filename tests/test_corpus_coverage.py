"""Coverage: mail that no profile claims must be reportable, not discovered by luck.

Selection rules are a snapshot of an interactive choice, and nothing used to
report what the choice left out — so a deliberate exclusion and an oversight
looked identical afterwards. On a real corpus that hid 11,832 messages (16%),
including most of an account's sent mail, found by accident while investigating
something else.
"""

import os
import shutil
import tempfile
import unittest
from unittest import mock

from src.ingest.coverage import coverage, render


class _Profile:
    def __init__(self, root, rules, collection):
        self.root = root
        self.selection_rules = rules
        self.collection = collection
        self.blacklist = None

    def resolved_root(self):
        return self.root


class TestCoverage(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="cov_")
        for folder, n in (("Work/Inbox", 3), ("Personal/iCloud", 2), ("Personal/Ignored", 4)):
            d = os.path.join(self.root, folder)
            os.makedirs(d, exist_ok=True)
            for i in range(n):
                open(os.path.join(d, f"m{i}.eml"), "wb").close()

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def _profiles(self):
        return [
            _Profile(self.root, [{"type": "prefix", "value": "Work/"}], "work"),
            _Profile(self.root, [{"type": "prefix", "value": "Personal/iCloud/"}], "personal"),
        ]

    def test_counts_claimed_and_unclaimed(self):
        r = coverage(self._profiles(), self.root)
        self.assertEqual(r["total"], 9)
        self.assertEqual(r["claimed"], 5)
        self.assertEqual(r["unclaimed"], 4)

    def test_attributes_selection_to_each_profile(self):
        r = coverage(self._profiles(), self.root)
        self.assertEqual(r["per_profile"], {"work": 3, "personal": 2})

    def test_names_the_folders_nobody_selected(self):
        r = coverage(self._profiles(), self.root)
        self.assertEqual(r["unclaimed_folders"].most_common(1)[0], ("Personal/Ignored", 4))

    def test_files_directly_in_a_top_level_folder_group_under_that_folder(self):
        """A message sitting straight in ``Trash/`` belongs to the folder ``Trash``.
        Grouping on the first two path components took its file name for a folder,
        so 3,000 trashed messages printed as 3,000 one-message "folders", each
        named after a subject line."""
        d = os.path.join(self.root, "Trash")
        os.makedirs(d)
        for name in ("Quarterly numbers.eml", "Re- lunch.eml", "Invoice 7.eml"):
            open(os.path.join(d, name), "wb").close()
        r = coverage(self._profiles(), self.root)
        self.assertEqual(r["unclaimed_folders"]["Trash/ (direct files)"], 3)
        self.assertFalse([f for f in r["unclaimed_folders"] if f.endswith(".eml")])
        self.assertNotIn("Quarterly numbers", render(r, limit=50))

    def test_a_message_at_the_corpus_root_is_reported_as_root(self):
        open(os.path.join(self.root, "loose.eml"), "wb").close()
        r = coverage(self._profiles(), self.root)
        self.assertEqual(r["unclaimed_folders"]["(root)"], 1)

    def test_deeper_folders_still_collapse_to_two_levels(self):
        d = os.path.join(self.root, "Personal", "Ignored", "2019", "Q1")
        os.makedirs(d)
        open(os.path.join(d, "old.eml"), "wb").close()
        r = coverage(self._profiles(), self.root)
        self.assertEqual(r["unclaimed_folders"]["Personal/Ignored"], 5)

    def test_loose_files_are_not_mistaken_for_the_whole_folder(self):
        """A two-level row counts everything beneath it, a one-level row only the
        files sitting directly in the folder. Printed alike, ``1 Personal`` next
        to ``4 Personal/Ignored`` reads as "Personal holds one message"."""
        open(os.path.join(self.root, "Personal", "loose.eml"), "wb").close()
        r = coverage(self._profiles(), self.root)
        self.assertEqual(r["unclaimed_folders"]["Personal/ (direct files)"], 1)
        self.assertEqual(r["unclaimed_folders"]["Personal/Ignored"], 4)
        self.assertNotIn("Personal", r["unclaimed_folders"])
        self.assertEqual(sum(r["unclaimed_folders"].values()), r["unclaimed"])

    def test_a_relative_or_tilde_root_gives_the_same_numbers(self):
        """Profiles resolve their files to absolute paths. A root passed as typed
        made the two sets disjoint: every claimed file also counted as unclaimed,
        and ``~/mail`` was never expanded so the walk found nothing."""
        want = coverage(self._profiles(), self.root)
        parent, name = os.path.split(self.root)
        cwd = os.getcwd()
        os.chdir(parent)
        try:
            got = coverage(self._profiles(), name)
        finally:
            os.chdir(cwd)
        self.assertEqual(got["total"], 9)
        self.assertEqual(got["unclaimed"], want["unclaimed"])
        self.assertEqual(got["claimed"] + got["unclaimed"], got["total"])
        with mock.patch.dict(os.environ, {"HOME": parent}):
            self.assertEqual(coverage(self._profiles(), "~/" + name)["total"], 9)

    def test_a_profile_rooted_through_a_symlink_claims_the_same_files(self):
        """Profiles record the root as it was typed. Reached through a symlink,
        none of a profile's paths matched the report's, and it claimed nothing."""
        link = self.root + "-link"
        os.symlink(self.root, link)
        self.addCleanup(os.remove, link)
        profiles = [
            _Profile(link, [{"type": "prefix", "value": "Work/"}], "work"),
            _Profile(link, [{"type": "prefix", "value": "Personal/iCloud/"}], "personal"),
        ]
        r = coverage(profiles, self.root)
        self.assertEqual(r["unclaimed"], 4)
        self.assertEqual(r["claimed"] + r["unclaimed"], r["total"])

    def test_claimed_counts_only_files_under_the_report_root(self):
        """A profile also selects mail outside the root being reported on. Counted,
        it made "claimed" larger than the total it is printed under."""
        r = coverage(self._profiles(), os.path.join(self.root, "Work"))
        self.assertEqual((r["total"], r["claimed"], r["unclaimed"]), (3, 3, 0))

    def test_a_message_claimed_by_any_profile_is_not_unclaimed(self):
        # Corpora share a root, so "unclaimed" means no profile selects it —
        # not "this particular profile skipped it".
        profiles = self._profiles() + [
            _Profile(self.root, [{"type": "prefix", "value": "Personal/Ignored/"}], "third")
        ]
        self.assertEqual(coverage(profiles, self.root)["unclaimed"], 0)

    def test_full_coverage_reports_nothing_missing(self):
        p = [_Profile(self.root, [{"type": "prefix", "value": ""}], "all")]
        r = coverage(p, self.root)
        self.assertEqual(r["unclaimed"], 0)
        self.assertNotIn("selects", render(r).split("per profile:")[1])


class TestRender(unittest.TestCase):
    def test_leads_with_the_unclaimed_number(self):
        r = {
            "total": 100,
            "claimed": 84,
            "unclaimed": 16,
            "per_profile": {"work": 84},
            "unclaimed_folders": __import__("collections").Counter({"X/Y": 16}),
        }
        text = render(r)
        self.assertIn("16%", text)
        self.assertIn("X/Y", text)

    def test_says_nothing_alarming_when_coverage_is_complete(self):
        r = {
            "total": 10,
            "claimed": 10,
            "unclaimed": 0,
            "per_profile": {"work": 10},
            "unclaimed_folders": __import__("collections").Counter(),
        }
        self.assertNotIn("searchable nowhere", render(r))


class TestProfileStampsUpdatedAt(unittest.TestCase):
    def test_saving_records_when_the_choice_was_made(self):
        """The field existed but nothing ever set it, so every profile read None."""
        from src.profile import CorpusProfile

        d = tempfile.mkdtemp(prefix="prof_")
        try:
            path = os.path.join(d, "p.profile.json")
            prof = CorpusProfile(root=d, collection="c")
            self.assertIsNone(prof.updated_at)
            prof.save(path)
            self.assertIsNotNone(CorpusProfile.load(path).updated_at)
        finally:
            shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
