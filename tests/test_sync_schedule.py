"""Scheduler unit rendering — the way this feature most often fails silently (#101)."""

from __future__ import annotations

import plistlib
import unittest
from unittest import mock

from src.sync.schedule import (
    LAUNCHD_LABEL,
    SYSTEMD_UNIT,
    install_hint,
    render_launchd_plist,
    render_systemd_units,
    sync_command,
)


class TestSyncCommand(unittest.TestCase):
    def test_uses_absolute_paths(self):
        """A scheduler inherits almost no environment; a relative path is the classic
        way to get a job that has silently failed since it was installed."""
        argv = sync_command(repo_root="/repo")
        self.assertTrue(argv[0].startswith("/"))

    def test_wraps_in_conda_run_when_an_env_is_given(self):
        with mock.patch("src.sync.schedule.shutil.which", return_value="/opt/conda/bin/conda"):
            argv = sync_command(repo_root="/repo", conda_env="mailrag")
        self.assertEqual(
            argv[:5], ["/opt/conda/bin/conda", "run", "--no-capture-output", "-n", "mailrag"]
        )
        self.assertIn("sync", argv)

    def test_conda_run_does_not_capture_output(self):
        """Without --no-capture-output the log file stays empty and a failing job
        looks identical to a working one."""
        with mock.patch("src.sync.schedule.shutil.which", return_value="/c/conda"):
            self.assertIn("--no-capture-output", sync_command(conda_env="mailrag"))

    def test_passes_the_account_and_model_through(self):
        argv = sync_command(repo_root="/repo", account="personal", model="qwen")
        self.assertIn("--account", argv)
        self.assertIn("personal", argv)
        self.assertIn("--model", argv)
        self.assertIn("qwen", argv)

    def test_omits_optional_flags_when_unset(self):
        argv = sync_command(repo_root="/repo")
        self.assertNotIn("--account", argv)
        self.assertNotIn("--model", argv)


class TestLaunchdPlist(unittest.TestCase):
    def _plist(self, **kw):
        kw.setdefault("interval_seconds", 43200)
        kw.setdefault("repo_root", "/repo")
        kw.setdefault("log_path", "/tmp/sync.log")
        return render_launchd_plist(**kw)

    def test_renders_valid_plist_xml(self):
        parsed = plistlib.loads(self._plist().encode("utf-8"))
        self.assertEqual(parsed["Label"], LAUNCHD_LABEL)
        self.assertEqual(parsed["StartInterval"], 43200)

    def test_uses_start_interval_so_a_sleeping_laptop_catches_up(self):
        """StartCalendarInterval silently loses a window that passes while asleep."""
        parsed = plistlib.loads(self._plist().encode("utf-8"))
        self.assertIn("StartInterval", parsed)
        self.assertNotIn("StartCalendarInterval", parsed)

    def test_always_writes_a_log(self):
        parsed = plistlib.loads(self._plist().encode("utf-8"))
        self.assertEqual(parsed["StandardOutPath"], "/tmp/sync.log")
        self.assertEqual(parsed["StandardErrorPath"], "/tmp/sync.log")

    def test_does_not_run_at_load(self):
        """Loading the agent should not kick off an unexpected sync mid-install."""
        self.assertFalse(plistlib.loads(self._plist().encode("utf-8"))["RunAtLoad"])

    def test_escapes_xml_metacharacters_in_paths(self):
        parsed = plistlib.loads(self._plist(log_path="/tmp/a&b<c>.log").encode("utf-8"))
        self.assertEqual(parsed["StandardOutPath"], "/tmp/a&b<c>.log")

    def test_includes_the_conda_env_in_the_program_arguments(self):
        with mock.patch("src.sync.schedule.shutil.which", return_value="/c/conda"):
            parsed = plistlib.loads(self._plist(conda_env="mailrag").encode("utf-8"))
        self.assertIn("mailrag", parsed["ProgramArguments"])


class TestSystemdUnits(unittest.TestCase):
    def _units(self, **kw):
        kw.setdefault("interval_seconds", 43200)
        kw.setdefault("repo_root", "/repo")
        kw.setdefault("log_path", "/tmp/sync.log")
        return render_systemd_units(**kw)

    def test_service_is_a_oneshot_with_logging(self):
        service, _timer = self._units()
        self.assertIn("Type=oneshot", service)
        self.assertIn("append:/tmp/sync.log", service)

    def test_timer_is_persistent_so_a_missed_window_still_fires(self):
        _service, timer = self._units()
        self.assertIn("Persistent=true", timer)
        self.assertIn("OnUnitActiveSec=43200s", timer)

    def test_service_waits_for_the_network(self):
        service, _timer = self._units()
        self.assertIn("network-online.target", service)


class TestInstallHint(unittest.TestCase):
    def test_macos_hint_uses_launchctl(self):
        hint = install_hint("darwin")
        self.assertIn("launchctl load", hint)
        self.assertIn(LAUNCHD_LABEL, hint)

    def test_linux_hint_uses_systemctl_user(self):
        hint = install_hint("linux")
        self.assertIn("systemctl --user", hint)
        self.assertIn(SYSTEMD_UNIT, hint)

    def test_both_hints_include_a_verification_step(self):
        self.assertIn("verify", install_hint("darwin"))
        self.assertIn("verify", install_hint("linux"))


if __name__ == "__main__":
    unittest.main()
