"""Two commands sharing one profile must not lose each other's work.

Observed: an index build loaded the profile, ran for ~40 minutes, and saved it
at the end — reverting a calibration that had finished in between. The loss was
silent, because the file simply held older numbers and nothing said so.

Refusing the late save is not enough. It turns a silent revert into a crash
that throws away the result the late command just spent its run computing. So
a save writes only the fields this instance changed and keeps whatever anyone
else changed meanwhile. It refuses only when both changed the same field.
"""

import io
import json
import os
import shutil
import tempfile
import unittest
from contextlib import redirect_stderr
from unittest import mock

from src.profile import CorpusProfile, ProfileChangedError

CALIBRATION = {"rubric": "work", "passed": True, "noise_rate": 0.42}


class _Tmp(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="prof_")
        self.path = os.path.join(self.dir, "corpus.profile.json")
        CorpusProfile(root=self.dir, collection="c", chunk_size=512).save(self.path)

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def _on_disk(self):
        with open(self.path, encoding="utf-8") as fh:
            return json.load(fh)

    def _other_command_records(self, **changes):
        """A second command loads, changes fields and saves while ours runs."""
        other = CorpusProfile.load(self.path)
        for name, value in changes.items():
            setattr(other, name, value)
        other.save(self.path)


class TestBothResultsSurvive(_Tmp):
    def test_a_late_save_keeps_the_other_commands_field_and_writes_its_own(self):
        """measure is running; calibrate finishes; measure saves its chunk size."""
        measure = CorpusProfile.load(self.path)
        self._other_command_records(calibration=CALIBRATION)
        measure.chunk_size = 384
        measure.save(self.path)
        self.assertEqual(self._on_disk()["calibration"], CALIBRATION)
        self.assertEqual(self._on_disk()["chunk_size"], 384)

    def test_a_holder_that_changed_nothing_reverts_nothing(self):
        """The original bug: a long build saving the copy it loaded at the start."""
        build = CorpusProfile.load(self.path)
        self._other_command_records(calibration=CALIBRATION, chunk_size=256)
        build.save(self.path)
        self.assertEqual(self._on_disk()["calibration"], CALIBRATION)
        self.assertEqual(self._on_disk()["chunk_size"], 256)

    def test_the_instance_itself_picks_up_what_it_kept(self):
        mine = CorpusProfile.load(self.path)
        self._other_command_records(calibration=CALIBRATION)
        mine.save(self.path)
        self.assertEqual(mine.calibration, CALIBRATION)

    def test_a_list_changed_in_place_counts_as_this_commands_change(self):
        """scope appends to selection_rules rather than assigning a new list. A
        snapshot sharing that list would see no change and drop the rules."""
        scope = CorpusProfile.load(self.path)
        self._other_command_records(calibration=CALIBRATION)
        scope.selection_rules.append({"type": "prefix", "value": "Inbox/"})
        scope.save(self.path)
        self.assertEqual(
            self._on_disk()["selection_rules"], [{"type": "prefix", "value": "Inbox/"}]
        )
        self.assertEqual(self._on_disk()["calibration"], CALIBRATION)

    def test_a_change_within_the_same_timestamp_is_still_seen(self):
        """Detection reads the file's content. An mtime comparison missed a
        rewrite inside one clock tick, or a file restored with an older stamp."""
        mine = CorpusProfile.load(self.path)
        before = os.stat(self.path)
        self._other_command_records(calibration=CALIBRATION)
        os.utime(self.path, ns=(before.st_atime_ns, before.st_mtime_ns))
        mine.chunk_size = 384
        mine.save(self.path)
        self.assertEqual(self._on_disk()["calibration"], CALIBRATION)


class TestARealConflictIsRefused(_Tmp):
    def _conflict(self):
        mine = CorpusProfile.load(self.path)
        self._other_command_records(chunk_size=256)
        mine.chunk_size = 384
        return mine

    def test_both_changing_one_field_raises(self):
        with self.assertRaises(ProfileChangedError):
            self._conflict().save(self.path)

    def test_the_refusal_leaves_the_other_commands_value_on_disk(self):
        with self.assertRaises(ProfileChangedError):
            self._conflict().save(self.path)
        self.assertEqual(self._on_disk()["chunk_size"], 256)

    def test_the_error_names_the_field_and_no_value(self):
        mine = CorpusProfile.load(self.path)
        self._other_command_records(selection_rules=[{"type": "prefix", "value": "Theirs/"}])
        mine.selection_rules = [{"type": "prefix", "value": "Mine/"}]
        with self.assertRaises(ProfileChangedError) as ctx:
            mine.save(self.path)
        self.assertIn("selection_rules", str(ctx.exception))
        self.assertNotIn("Theirs", str(ctx.exception))
        self.assertNotIn("Mine", str(ctx.exception))

    def test_a_conflict_on_one_field_still_saves_the_others(self):
        """A persona run changes several fields and saves once, after the index
        is built. Refusing the whole save over one contested field threw away
        the rest, and "re-run" then meant rebuilding the index."""
        run = CorpusProfile.load(self.path)
        self._other_command_records(chunk_size=256)
        run.selection_rules = [{"type": "prefix", "value": "Inbox/"}]
        run.calibration = CALIBRATION
        run.chunk_size = 384
        with self.assertRaises(ProfileChangedError) as ctx:
            run.save(self.path)
        disk = self._on_disk()
        self.assertEqual(disk["selection_rules"], [{"type": "prefix", "value": "Inbox/"}])
        self.assertEqual(disk["calibration"], CALIBRATION)
        self.assertEqual(disk["chunk_size"], 256, "the contested field keeps the other value")
        self.assertIn("chunk_size", str(ctx.exception))
        self.assertNotIn("calibration", str(ctx.exception))

    def test_after_a_conflict_a_second_save_does_not_sneak_the_field_in(self):
        mine = self._conflict()
        with self.assertRaises(ProfileChangedError):
            mine.save(self.path)
        with self.assertRaises(ProfileChangedError):
            mine.save(self.path)
        self.assertEqual(self._on_disk()["chunk_size"], 256)

    def test_both_arriving_at_the_same_value_is_not_a_conflict(self):
        mine = CorpusProfile.load(self.path)
        self._other_command_records(chunk_size=256)
        mine.chunk_size = 256
        mine.save(self.path)
        self.assertEqual(self._on_disk()["chunk_size"], 256)

    def test_force_writes_this_copy_as_it_stands(self):
        mine = self._conflict()
        mine.save(self.path, force=True)
        self.assertEqual(self._on_disk()["chunk_size"], 384)

    def test_the_cli_reports_it_as_an_error_not_a_traceback(self):
        """A RuntimeError escaping main() printed a stack trace at the end of a
        command that had otherwise finished its work."""
        from src import cli

        def boom(_args):
            raise ProfileChangedError("profile changed: chunk_size")

        err = io.StringIO()
        with mock.patch.object(cli, "_cmd_measure", boom), redirect_stderr(err):
            rc = cli.main(["measure", "--profile", self.path])
        self.assertEqual(rc, 1)
        self.assertIn("error: profile changed: chunk_size", err.getvalue())


class TestOrdinaryUseUnaffected(_Tmp):
    def test_repeated_saves_from_one_instance_are_fine(self):
        prof = CorpusProfile.load(self.path)
        for size in (300, 400, 500):
            prof.chunk_size = size
            prof.save(self.path)
        self.assertEqual(self._on_disk()["chunk_size"], 500)

    def test_a_profile_never_loaded_from_disk_overwrites(self):
        """onboarding builds a profile from scratch; there is nothing to merge."""
        CorpusProfile(root=self.dir, collection="new", chunk_size=128).save(self.path)
        self.assertEqual(self._on_disk()["collection"], "new")
        self.assertEqual(self._on_disk()["chunk_size"], 128)

    def test_saving_under_another_name_does_not_merge_with_that_file(self):
        """The snapshot belongs to the file it was read from. Saving a copy over a
        different, existing profile must not keep that profile's fields."""
        other_path = os.path.join(self.dir, "other.json")
        CorpusProfile(root="/elsewhere", collection="other", chunk_size=999).save(other_path)
        CorpusProfile.load(self.path).save(other_path)
        with open(other_path, encoding="utf-8") as fh:
            data = json.load(fh)
        self.assertEqual((data["collection"], data["chunk_size"]), ("c", 512))

    def test_a_file_deleted_since_load_is_written_again(self):
        prof = CorpusProfile.load(self.path)
        os.remove(self.path)
        prof.chunk_size = 384
        prof.save(self.path)
        self.assertEqual(self._on_disk()["chunk_size"], 384)

    def test_a_symlinked_profile_stays_a_link_and_still_merges(self):
        """Renaming over the link replaced it with a plain file, left the real
        profile stale, and made the next save skip the merge and write whole."""
        link = os.path.join(self.dir, "link.profile.json")
        os.symlink(self.path, link)
        holder = CorpusProfile.load(link)
        other = CorpusProfile.load(link)
        other.calibration = CALIBRATION
        other.save(link)
        holder.chunk_size = 384
        holder.save(link)
        self.assertTrue(os.path.islink(link))
        self.assertEqual(self._on_disk()["calibration"], CALIBRATION)
        self.assertEqual(self._on_disk()["chunk_size"], 384)

    def test_a_save_keeps_the_files_permissions(self):
        os.chmod(self.path, 0o600)
        prof = CorpusProfile.load(self.path)
        prof.chunk_size = 384
        prof.save(self.path)
        self.assertEqual(os.stat(self.path).st_mode & 0o777, 0o600)

    @unittest.skipIf(os.geteuid() == 0, "root can write a read-only file")
    def test_a_read_only_profile_is_still_refused(self):
        """A rename only needs the directory to be writable, so it would have
        gone straight through a file its owner had locked."""
        prof = CorpusProfile.load(self.path)
        os.chmod(self.path, 0o444)
        prof.chunk_size = 384
        try:
            with self.assertRaises(PermissionError):
                prof.save(self.path)
        finally:
            os.chmod(self.path, 0o644)
        self.assertEqual(self._on_disk()["chunk_size"], 512)

    def test_a_file_that_is_no_longer_a_profile_is_overwritten(self):
        prof = CorpusProfile.load(self.path)
        for junk in ("[]", "null", "", "{not json"):
            with self.subTest(junk=junk):
                with open(self.path, "w", encoding="utf-8") as fh:
                    fh.write(junk)
                prof.chunk_size = 384
                prof.save(self.path)
                self.assertEqual(self._on_disk()["chunk_size"], 384)

    def test_bookkeeping_never_reaches_the_file(self):
        CorpusProfile.load(self.path).save(self.path)
        self.assertFalse([k for k in self._on_disk() if k.startswith("_")])

    def test_bookkeeping_cannot_be_planted_in_the_file(self):
        data = self._on_disk()
        data["_loaded"] = {"chunk_size": "1"}
        data["_loaded_path"] = "/nowhere"
        with open(self.path, "w", encoding="utf-8") as fh:
            json.dump(data, fh)
        prof = CorpusProfile.load(self.path)
        self.assertEqual(prof._loaded_path, os.path.realpath(self.path))
        self._other_command_records(calibration=CALIBRATION)
        prof.save(self.path)
        self.assertEqual(self._on_disk()["calibration"], CALIBRATION)

    def test_round_trip_preserves_the_real_fields(self):
        prof = CorpusProfile.load(self.path)
        prof.selection_rules = [{"type": "prefix", "value": "Inbox/"}]
        prof.calibration = CALIBRATION
        prof.save(self.path)
        again = CorpusProfile.load(self.path)
        self.assertEqual(again.selection_rules, prof.selection_rules)
        self.assertEqual(again.calibration, CALIBRATION)
        self.assertIsNotNone(again.updated_at)

    def test_a_save_never_leaves_a_half_written_file_or_a_stray(self):
        """Written to a temp file and renamed, so a reader (another command's
        merge, the MCP server) never sees a truncated profile."""
        prof = CorpusProfile.load(self.path)
        seen = []
        real_replace = os.replace

        def spy(src, dst):
            with open(src, encoding="utf-8") as fh:
                seen.append(json.load(fh)["chunk_size"])
            return real_replace(src, dst)

        prof.chunk_size = 384
        with mock.patch("src.profile.os.replace", side_effect=spy):
            prof.save(self.path)
        self.assertEqual(seen, [384])
        self.assertEqual(os.listdir(self.dir), ["corpus.profile.json"])


class TestIndexDoesNotWriteTheProfile(unittest.TestCase):
    """`build` only reads the profile, so it must not rewrite it at all."""

    def test_a_build_leaves_the_profile_file_byte_for_byte(self):
        from src import cli

        with tempfile.TemporaryDirectory() as d:
            fp = os.path.join(d, "p.json")
            CorpusProfile(root="/r", collection="c").save(fp)
            with open(fp, "rb") as fh:
                before = fh.read()
            with mock.patch("src.cli.build_stage") as bs, mock.patch("src.cli.BgeM3Embedder"):
                bs.run.return_value = mock.Mock(chunks=10, collection="c")
                rc = cli.main(["build", "--profile", fp, "--limit", "1"])
            with open(fp, "rb") as fh:
                after = fh.read()
        self.assertEqual(rc, 0)
        self.assertEqual(before, after)


class TestTheTuiSurvivesAConflict(_Tmp):
    """execute_plan saves in a ``finally``. A conflict raised there replaced the
    run's own outcome with a stack trace in the middle of the screen."""

    def test_a_conflict_at_the_final_save_is_logged_and_the_run_still_returns(self):
        from src.tui import flow

        prof = CorpusProfile.load(self.path)
        self._other_command_records(chunk_size=256)

        def measure(p):
            p.chunk_size = 384

        step = mock.Mock(verb="measure", skipped=False, params={})
        ui = mock.Mock()
        rc = flow.execute_plan(prof, self.path, [step], {"measure": measure}, ui)
        self.assertEqual(rc, 0)
        logged = " ".join(str(c.args[0]) for c in ui.log.call_args_list)
        self.assertIn("chunk_size", logged)
        self.assertEqual(self._on_disk()["chunk_size"], 256)

    def test_a_handlers_own_exception_is_not_hidden_by_it(self):
        from src.tui import flow

        prof = CorpusProfile.load(self.path)
        self._other_command_records(chunk_size=256)

        def measure(p):
            p.chunk_size = 384
            raise RuntimeError("handler failed")

        step = mock.Mock(verb="measure", skipped=False, params={})
        with self.assertRaisesRegex(RuntimeError, "handler failed"):
            flow.execute_plan(prof, self.path, [step], {"measure": measure}, mock.Mock())


if __name__ == "__main__":
    unittest.main()
