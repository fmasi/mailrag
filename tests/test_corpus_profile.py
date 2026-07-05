# tests/test_corpus_profile.py
import json
import os
import tempfile
import unittest

from src.profile import CorpusProfile


class TestCorpusProfile(unittest.TestCase):
    def test_round_trip(self):
        p = CorpusProfile(
            root="~/m",
            selection_rules=[{"type": "prefix", "value": "a/"}],
            chunk_size=1024,
            collection="c",
        )
        with tempfile.TemporaryDirectory() as d:
            fp = os.path.join(d, "prof.json")
            p.save(fp)
            q = CorpusProfile.load(fp)
        self.assertEqual(q.root, "~/m")
        self.assertEqual(q.chunk_size, 1024)
        self.assertEqual(q.selection_rules, p.selection_rules)

    def test_load_migrates_old_selection_json(self):
        # Old selector output: extra keys (n_selected/n_total/generated_at) must be ignored,
        # missing new keys (chunk_size, collection) must default in.
        old = {
            "root": "/r",
            "selection_rules": [{"type": "prefix", "value": "x/"}],
            "n_selected": 100,
            "n_total": 200,
            "generated_at": "2026-06-03T20:08:43",
        }
        with tempfile.TemporaryDirectory() as d:
            fp = os.path.join(d, "sel.json")
            with open(fp, "w") as fh:
                json.dump(old, fh)
            p = CorpusProfile.load(fp)
        self.assertEqual(p.root, "/r")
        self.assertEqual(p.selection_rules, old["selection_rules"])
        self.assertEqual(p.chunk_size, 512)  # default
        self.assertEqual(p.collection, "email-rag")  # default

    def test_resolved_root_expands(self):
        p = CorpusProfile(root="~/x")
        self.assertTrue(p.resolved_root().endswith("/x"))
        self.assertFalse(p.resolved_root().startswith("~"))


if __name__ == "__main__":
    unittest.main()
