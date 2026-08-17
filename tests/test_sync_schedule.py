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

    def test_runs_at_load_so_a_boot_does_not_lose_the_window(self):
        """A LaunchAgent loads at login, so RunAtLoad is the boot catch-up.

        With this false, the first sync after every restart is one whole interval
        away — a measured 4 h blind window on 2026-08-17, when a reboot at 11:09
        left `launchctl print` reporting zero runs at 11:35. The systemd sibling
        never had this gap (``OnBootSec=5min``), so macOS was the one platform
        that lost a window on boot.

        The cost is a sync at install time, which is harmless: ``sync`` is
        one-shot and idempotent over a resumable queue, so the worst case is an
        idle tick that indexes nothing.
        """
        self.assertTrue(plistlib.loads(self._plist().encode("utf-8"))["RunAtLoad"])

    def test_the_unit_always_gates_on_per_account_cadence(self):
        """Without --due-only the unit ticks every account at the SHORTEST cadence.

        The interval is the minimum across accounts, so a unit missing this flag
        would sync a 24 h account every 4 h and make `cadence:` in accounts.yaml
        decorative — the exact class of silent drift this replaced.
        """
        parsed = plistlib.loads(self._plist().encode("utf-8"))
        self.assertIn("--due-only", parsed["ProgramArguments"])

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

    def test_the_service_also_gates_on_per_account_cadence(self):
        service, _timer = self._units()
        self.assertIn("--due-only", service)


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


class TestSchedulerEnvironment(unittest.TestCase):
    """A scheduled job inherits almost no environment — and this project's .env
    is written for the DEVCONTAINER, where `host.docker.internal` resolves. On the
    host it does not, so a unit without an explicit environment would fail every
    tick forever while blaming the LLM and the vector store."""

    def _plist(self, **kw):
        kw.setdefault("interval_seconds", 43200)
        kw.setdefault("repo_root", "/repo")
        kw.setdefault("log_path", "/tmp/sync.log")
        return plistlib.loads(render_launchd_plist(**kw).encode("utf-8"))

    def test_the_launchd_unit_pins_both_endpoints_to_localhost(self):
        env = self._plist()["EnvironmentVariables"]
        self.assertEqual(env["RAG_LLM_API_BASE"], "http://localhost:1234/v1")
        self.assertEqual(env["QDRANT_URL"], "http://localhost:6333")

    def test_no_endpoint_is_left_pointing_at_a_container_hostname(self):
        for v in self._plist()["EnvironmentVariables"].values():
            self.assertNotIn("host.docker.internal", v)

    def test_the_environment_can_be_overridden(self):
        env = self._plist(environment={"QDRANT_URL": "http://otherbox:6333"})[
            "EnvironmentVariables"
        ]
        self.assertEqual(env["QDRANT_URL"], "http://otherbox:6333")
        # the un-overridden default survives
        self.assertEqual(env["RAG_LLM_API_BASE"], "http://localhost:1234/v1")

    def test_extra_variables_can_be_added(self):
        env = self._plist(environment={"RAG_LLM_MAX_RETRIES": "1"})["EnvironmentVariables"]
        self.assertEqual(env["RAG_LLM_MAX_RETRIES"], "1")
        self.assertIn("QDRANT_URL", env)

    def test_the_systemd_service_carries_the_same_environment(self):
        service, _timer = render_systemd_units(
            interval_seconds=43200, repo_root="/repo", log_path="/tmp/sync.log"
        )
        self.assertIn("Environment=RAG_LLM_API_BASE=http://localhost:1234/v1", service)
        self.assertIn("Environment=QDRANT_URL=http://localhost:6333", service)

    def test_environment_lines_precede_execstart(self):
        """systemd applies Environment= in order; after ExecStart it would be
        parsed but is conventionally wrong and easy to misread."""
        service, _ = render_systemd_units(
            interval_seconds=43200, repo_root="/repo", log_path="/tmp/sync.log"
        )
        lines = service.splitlines()
        self.assertLess(
            max(i for i, l in enumerate(lines) if l.startswith("Environment=")),
            next(i for i, l in enumerate(lines) if l.startswith("ExecStart=")),
        )

    def test_the_unit_carries_a_PATH_that_includes_homebrew(self):
        """Attachment OCR shells out to `tesseract`, which a scheduled job cannot
        find without PATH — and the resulting `ocr_unavailable` used to be cached,
        so a later supervised run never healed it (GH #37 via launchd)."""
        path = self._plist()["EnvironmentVariables"]["PATH"]
        self.assertIn("/opt/homebrew/bin", path)
        self.assertIn("/usr/bin", path)
