"""A command must not write back a profile someone else changed meanwhile.

Observed: an index build loaded the profile, ran for ~40 minutes, and saved it
at the end — reverting a calibration that had finished in between. The loss was
silent, because the file simply held older numbers and nothing said so.
"""

import json
import os
import shutil
import tempfile
import time
import unittest

from src.profile import CorpusProfile, ProfileChangedError


class _Tmp(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="prof_")
        self.path = os.path.join(self.dir, "corpus.profile.json")
        CorpusProfile(root=self.dir, collection="c").save(self.path)

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)


class TestStaleWriteRefused(_Tmp):
    def test_saving_over_someone_elses_write_raises(self):
        long_running = CorpusProfile.load(self.path)
        other = CorpusProfile.load(self.path)
        time.sleep(0.02)
        other.calibration = {"noise_rate": 0.49}
        other.save(self.path)

        long_running.collection = "stale"
        with self.assertRaises(ProfileChangedError):
            long_running.save(self.path)

    def test_the_other_commands_result_survives(self):
        long_running = CorpusProfile.load(self.path)
        other = CorpusProfile.load(self.path)
        time.sleep(0.02)
        other.calibration = {"noise_rate": 0.49}
        other.save(self.path)
        try:
            long_running.save(self.path)
        except ProfileChangedError:
            pass
        self.assertEqual(json.load(open(self.path))["calibration"], {"noise_rate": 0.49})

    def test_force_overrides_when_the_caller_means_it(self):
        held = CorpusProfile.load(self.path)
        time.sleep(0.02)
        CorpusProfile.load(self.path).save(self.path)
        held.collection = "deliberate"
        held.save(self.path, force=True)
        self.assertEqual(json.load(open(self.path))["collection"], "deliberate")


class TestOrdinaryUseUnaffected(_Tmp):
    def test_repeated_saves_from_one_instance_are_fine(self):
        # The guard must not fire on a command that saves more than once.
        prof = CorpusProfile.load(self.path)
        prof.collection = "a"
        prof.save(self.path)
        prof.collection = "b"
        prof.save(self.path)
        self.assertEqual(json.load(open(self.path))["collection"], "b")

    def test_a_profile_never_loaded_from_disk_can_be_saved(self):
        fresh = CorpusProfile(root=self.dir, collection="new")
        fresh.save(os.path.join(self.dir, "other.json"))
        self.assertTrue(os.path.exists(os.path.join(self.dir, "other.json")))

    def test_the_bookkeeping_field_is_not_serialised(self):
        CorpusProfile.load(self.path).save(self.path)
        self.assertNotIn("_loaded_mtime", json.load(open(self.path)))

    def test_round_trip_preserves_the_real_fields(self):
        prof = CorpusProfile.load(self.path)
        prof.selection_rules = [{"type": "prefix", "value": "Inbox/"}]
        prof.save(self.path)
        again = CorpusProfile.load(self.path)
        self.assertEqual(again.selection_rules, [{"type": "prefix", "value": "Inbox/"}])
        self.assertIsNotNone(again.updated_at)


class TestIndexDoesNotWriteTheProfile(unittest.TestCase):
    """`build` only reads the profile, so it must not save it at all.

    Removing the write is the actual fix; the staleness guard above is the
    backstop for every other command that legitimately does save.
    """

    def test_cmd_index_has_no_save_call(self):
        import inspect

        import src.cli as cli

        self.assertNotIn("prof.save", inspect.getsource(cli._cmd_index))

    def test_commands_that_record_a_result_still_save(self):
        import inspect

        import src.cli as cli

        for name in ("_cmd_calibrate", "_cmd_scope", "_cmd_measure"):
            with self.subTest(cmd=name):
                self.assertIn("prof.save", inspect.getsource(getattr(cli, name)))


if __name__ == "__main__":
    unittest.main()
