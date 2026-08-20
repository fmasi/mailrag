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
