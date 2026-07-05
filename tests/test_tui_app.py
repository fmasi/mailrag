"""Pilot tests for the full-screen Textual wizard (src/tui/app.py).

The whole app runs headlessly via ``App.run_test()``; the pipeline handlers are
faked (patched at ``src.tui.flow.build_handlers``), so tests drive real screens,
keybindings, and the worker/modal bridge without touching Qdrant or an LLM.
Uses ``unittest.IsolatedAsyncioTestCase`` (stdlib), which pytest runs natively.
"""

import json
import os
import tempfile
import time
import unittest
from unittest import mock

from src.tui.app import (
    CalibrateGateModal,
    ConfirmModal,
    MailragWizardApp,
    ModelScreen,
    PersonaScreen,
    ReviewScreen,
    RubricPickModal,
    RunScreen,
    ScopeScreen,
    WelcomeScreen,
    _describe_rule,
    run_tui,
)

_SIZE = (110, 40)


def _recording_handlers(calls, prune_confirm=None):
    """The fake verb handlers; ``prune`` exercises the injected confirm gate."""

    def make(verb):
        def handler(prof, **params):
            calls.append(verb)
            return f"{verb}-report"

        return handler

    handlers = {
        v: make(v)
        for v in (
            "scope",
            "measure",
            "tag",
            "scan",
            "calibrate",
            "summarize",
            "index",
            "judge",
            "prune",
        )
    }
    if prune_confirm is not None:

        def prune(prof, **params):
            calls.append("prune")
            # bracketed subject exercises markup-escaping in the prune dialog
            return 2 if prune_confirm(["0.91  [Receipt] promo blast", "0.88  newsletter"]) else 0

        handlers["prune"] = prune
    return handlers


class _TuiCase(unittest.IsolatedAsyncioTestCase):
    """Shared fixture: a temp profile + mailbox, and fake pipeline handlers."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = os.path.join(self.tmp.name, "mailbox")
        # "[Lists]" exercises markup-escaping of folder names in the scope tree.
        for rel in (
            "Inbox/a.eml",
            "Inbox/Acme/b.eml",
            "Archive/c.eml",
            "root.eml",
            "[Lists]/l.eml",
        ):
            path = os.path.join(root, *rel.split("/"))
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as fh:
                fh.write("Subject: hi\n\nbody\n")
        self.profile_path = os.path.join(self.tmp.name, "box.profile.json")
        with open(self.profile_path, "w") as fh:
            json.dump({"root": root, "rubric": "personal"}, fh)

        self.calls = []

        def fake_build_handlers(**kwargs):
            self.build_kwargs = kwargs
            return _recording_handlers(self.calls, prune_confirm=kwargs.get("prune_confirm"))

        patcher = mock.patch("src.tui.flow.build_handlers", side_effect=fake_build_handlers)
        patcher.start()
        self.addCleanup(patcher.stop)

    def app(self, **kwargs):
        return MailragWizardApp(self.profile_path, **kwargs)

    async def wait_for(self, pilot, predicate, timeout=8.0):
        """Poll *predicate* while the app processes events (worker + modals)."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return
            await pilot.pause(0.05)
        self.fail("timed out waiting for app state")

    def _screen_is(self, app, cls):
        return lambda: isinstance(app.screen, cls)

    async def run_done(self, app, pilot):
        """Wait for the run worker to finish, then dismiss the run screen."""
        await self.wait_for(
            pilot, lambda: isinstance(app.screen, RunScreen) and app.screen._exit_code is not None
        )
        await pilot.press("enter")
        await self.wait_for(pilot, lambda: app.return_value is not None)

    async def to_scope(self, app, pilot, persona_index=0):
        """Welcome -> persona pick (by list position) -> land wherever comes next."""
        await self.wait_for(pilot, self._screen_is(app, WelcomeScreen))
        await pilot.press("enter")
        await self.wait_for(pilot, self._screen_is(app, PersonaScreen))
        for _ in range(persona_index):
            await pilot.press("down")
        await pilot.press("enter")

    async def answer_gate(self, app, pilot, keys, wanted):
        """Wait for a gate modal of type *wanted*, answer it, wait for dismissal.

        Waiting for the answered instance to leave the screen keeps two
        consecutive modals of the same class (spend confirm, then prune
        confirm) from being conflated."""
        await self.wait_for(pilot, self._screen_is(app, wanted))
        modal = app.screen
        await pilot.press(*((keys,) if isinstance(keys, str) else keys))
        await self.wait_for(pilot, lambda: app.screen is not modal)


class TestHappyPathNoLLM(_TuiCase):
    async def test_llm_none_end_to_end(self):
        app = self.app()
        async with app.run_test(size=_SIZE) as pilot:
            await self.to_scope(app, pilot, persona_index=0)  # llm-none is first
            await self.wait_for(pilot, self._screen_is(app, ScopeScreen))
            # llm-none has no LLM steps: no Model stage in the breadcrumb
            self.assertNotIn("Model", app.stage_names)
            # rows: root-files leaf, Archive/, Inbox/ (+child). Toggle Inbox/.
            await pilot.press("down", "down", "space")
            await pilot.press("c")
            await self.wait_for(pilot, self._screen_is(app, ReviewScreen))
            await pilot.press("enter")
            await self.answer_gate(app, pilot, "y", ConfirmModal)  # prune confirm
            await self.run_done(app, pilot)
        self.assertEqual(app.return_value, 0)
        # 'scope' is not in calls: the TUI replaces that handler with the rules
        # picked on the scope screen (flow.make_scope_handler).
        self.assertEqual(self.calls, ["measure", "scan", "tag", "prune", "index"])
        # the scope handler wrote the picked rules through to the saved profile
        with open(self.profile_path) as fh:
            saved = json.load(fh)
        self.assertEqual(saved["selection_rules"], [{"type": "prefix", "value": "Inbox/"}])

    async def test_scope_requires_a_selection(self):
        app = self.app()
        async with app.run_test(size=_SIZE) as pilot:
            await self.to_scope(app, pilot)
            await self.wait_for(pilot, self._screen_is(app, ScopeScreen))
            await pilot.press("c")  # nothing selected -> stays with a warning
            await pilot.pause()
            self.assertIsInstance(app.screen, ScopeScreen)
            await pilot.press("space", "c")  # cursor starts on the root-files row
            await self.wait_for(pilot, self._screen_is(app, ReviewScreen))
        self.assertEqual(app.state.scope_rules, [{"type": "container-root"}])


class TestLLMPathAndGates(_TuiCase):
    def setUp(self):
        super().setUp()
        for target, value in (
            # bracketed subject exercises markup-escaping in the calibrate dialog
            ("src.llm.calibration.format_report", "BUCKETS: [Receipt] Your order :: 50% off"),
            ("src.llm.rubrics.names", ["personal", "work"]),
        ):
            p = mock.patch(target, return_value=value)
            p.start()
            self.addCleanup(p.stop)

    async def to_run(self, app, pilot, gate_keys):
        """Drive llm-all to the run screen, answering the gates with *gate_keys*."""
        await self.to_scope(app, pilot, persona_index=2)  # llm-all is third
        await self.wait_for(pilot, self._screen_is(app, ModelScreen))
        await pilot.press("enter")  # empty submit -> stays, shows the error
        await pilot.pause()
        self.assertIsInstance(app.screen, ModelScreen)
        app.screen.query_one("#model-input").value = "my-model"
        await pilot.press("enter")
        await self.wait_for(pilot, self._screen_is(app, ScopeScreen))
        await pilot.press("space", "c")
        await self.wait_for(pilot, self._screen_is(app, ReviewScreen))
        await pilot.press("enter")
        for key, wanted in gate_keys:
            await self.answer_gate(app, pilot, key, wanted)

    async def test_llm_all_proceed_spend_and_prune_confirm(self):
        app = self.app()
        async with app.run_test(size=_SIZE) as pilot:
            await self.to_run(
                app,
                pilot,
                gate_keys=[
                    ("p", CalibrateGateModal),  # calibrate gate -> proceed
                    ("y", ConfirmModal),  # confirm-before-spend -> yes
                    ("y", ConfirmModal),  # prune blacklist confirm -> yes
                ],
            )
            await self.run_done(app, pilot)
        self.assertEqual(app.return_value, 0)
        self.assertEqual(app.state.model, "my-model")
        self.assertEqual(self.calls, ["measure", "calibrate", "summarize", "prune", "index"])

    async def test_calibrate_gate_abort_stops_run(self):
        app = self.app()
        async with app.run_test(size=_SIZE) as pilot:
            await self.to_run(app, pilot, gate_keys=[("a", CalibrateGateModal)])
            await self.run_done(app, pilot)
        self.assertEqual(app.return_value, 1)
        self.assertNotIn("summarize", self.calls)

    async def test_calibrate_gate_retune_picks_rubric_and_recalibrates(self):
        app = self.app()
        async with app.run_test(size=_SIZE) as pilot:
            await self.to_run(app, pilot, gate_keys=[("r", CalibrateGateModal)])
            await self.answer_gate(app, pilot, ("down", "enter"), RubricPickModal)  # pick 'work'
            await self.answer_gate(app, pilot, "p", CalibrateGateModal)
            await self.answer_gate(app, pilot, "y", ConfirmModal)  # spend
            await self.answer_gate(app, pilot, "y", ConfirmModal)  # prune
            await self.run_done(app, pilot)
        self.assertEqual(app.return_value, 0)
        self.assertEqual(app.state.profile.rubric, "work")
        self.assertEqual(self.calls.count("calibrate"), 2)

    async def test_spend_declined_stops_run(self):
        app = self.app()
        async with app.run_test(size=_SIZE) as pilot:
            await self.to_run(
                app,
                pilot,
                gate_keys=[("p", CalibrateGateModal), ("n", ConfirmModal)],
            )
            await self.run_done(app, pilot)
        self.assertEqual(app.return_value, 1)
        self.assertNotIn("summarize", self.calls)

    async def test_cli_model_skips_model_screen(self):
        app = self.app(model="cli-model")
        async with app.run_test(size=_SIZE) as pilot:
            await self.to_scope(app, pilot, persona_index=2)  # llm-all
            await self.wait_for(pilot, self._screen_is(app, ScopeScreen))
            # the skipped Model stage must not appear in the breadcrumb at all
            self.assertNotIn("Model", app.stage_names)
        self.assertEqual(app.state.model, "cli-model")

    async def test_model_back_returns_to_persona(self):
        app = self.app()
        async with app.run_test(size=_SIZE) as pilot:
            await self.to_scope(app, pilot, persona_index=2)  # llm-all
            await self.wait_for(pilot, self._screen_is(app, ModelScreen))
            await pilot.press("escape")
            await self.wait_for(pilot, self._screen_is(app, PersonaScreen))
            await pilot.press("enter")  # the chosen persona is still highlighted
            await self.wait_for(pilot, self._screen_is(app, ModelScreen))
        self.assertEqual(app.state.persona_name, "llm-all")


class TestNavigationAndEdges(_TuiCase):
    async def test_quit_from_welcome(self):
        app = self.app()
        async with app.run_test(size=_SIZE) as pilot:
            await self.wait_for(pilot, self._screen_is(app, WelcomeScreen))
            await pilot.press("q")
            await self.wait_for(pilot, lambda: app.return_value is not None)
        self.assertEqual(app.return_value, 1)

    async def test_back_navigation_persona_to_welcome_and_scope_to_persona(self):
        app = self.app()
        async with app.run_test(size=_SIZE) as pilot:
            await self.to_scope(app, pilot)
            await self.wait_for(pilot, self._screen_is(app, ScopeScreen))
            await pilot.press("escape")
            await self.wait_for(pilot, self._screen_is(app, PersonaScreen))
            await pilot.press("escape")
            await self.wait_for(pilot, self._screen_is(app, WelcomeScreen))
            await pilot.press("q")
            await self.wait_for(pilot, lambda: app.return_value is not None)
        self.assertEqual(app.return_value, 1)

    async def test_review_blocks_on_missing_required_verbs(self):
        # Drop judge+prune: llm-verify becomes unrunnable and review must block.
        def crippled_build_handlers(**kwargs):
            handlers = _recording_handlers(self.calls)
            del handlers["judge"]
            del handlers["prune"]
            return handlers

        with mock.patch("src.tui.flow.build_handlers", side_effect=crippled_build_handlers):
            app = self.app(model="m")
            async with app.run_test(size=_SIZE) as pilot:
                await self.to_scope(app, pilot, persona_index=1)  # llm-verify
                await self.wait_for(pilot, self._screen_is(app, ScopeScreen))
                await pilot.press("space", "c")
                await self.wait_for(pilot, self._screen_is(app, ReviewScreen))
                await pilot.press("enter")  # blocked -> notified, stays on review
                await pilot.pause()
                self.assertIsInstance(app.screen, ReviewScreen)
                await pilot.press("q")
                await self.wait_for(pilot, lambda: app.return_value is not None)
        self.assertEqual(app.return_value, 2)  # classic 'not implemented' exit code
        self.assertNotIn("summarize", self.calls)

    async def test_scope_empty_mailbox_shows_error_and_allows_back(self):
        empty_root = os.path.join(self.tmp.name, "empty")
        os.makedirs(empty_root)
        with open(self.profile_path, "w") as fh:
            json.dump({"root": empty_root, "rubric": "personal"}, fh)
        app = self.app()
        async with app.run_test(size=_SIZE) as pilot:
            await self.to_scope(app, pilot)
            await self.wait_for(pilot, self._screen_is(app, ScopeScreen))
            self.assertTrue(app.screen.query("#scope-empty"))
            await pilot.press("c")  # cannot continue with nothing to select
            await pilot.pause()
            self.assertIsInstance(app.screen, ScopeScreen)
            await pilot.press("escape")
            await self.wait_for(pilot, self._screen_is(app, PersonaScreen))

    async def test_welcome_shows_scan_recommendation(self):
        with open(os.path.join(self.tmp.name, "box.profile.scan.json"), "w") as fh:
            json.dump({"recommended_persona": "llm-verify"}, fh)
        app = self.app()
        async with app.run_test(size=_SIZE) as pilot:
            await self.wait_for(pilot, self._screen_is(app, WelcomeScreen))
            self.assertEqual(app.state.recommendation, "llm-verify")
            await pilot.press("enter")
            await self.wait_for(pilot, self._screen_is(app, PersonaScreen))
            # the recommended persona is pre-highlighted; enter accepts it
            await pilot.press("enter")
            await pilot.pause()
        self.assertEqual(app.state.persona_name, "llm-verify")


class TestRunTuiGuards(unittest.TestCase):
    def test_run_tui_needs_a_tty(self):
        # Under pytest stdin/stdout are not ttys, so the guard trips.
        rc = run_tui("does-not-matter.json")
        self.assertEqual(rc, 2)

    def test_run_tui_reports_unloadable_profile(self):
        with (
            mock.patch("sys.stdin.isatty", return_value=True),
            mock.patch("sys.stdout.isatty", return_value=True),
        ):
            rc = run_tui(os.path.join(tempfile.gettempdir(), "no-such-profile.json"))
        self.assertEqual(rc, 2)

    def test_run_tui_reports_profile_of_wrong_shape(self):
        # Valid JSON that isn't an object -> friendly exit 2, not a traceback.
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
            fh.write("[1, 2, 3]")
            path = fh.name
        self.addCleanup(os.unlink, path)
        with (
            mock.patch("sys.stdin.isatty", return_value=True),
            mock.patch("sys.stdout.isatty", return_value=True),
        ):
            rc = run_tui(path)
        self.assertEqual(rc, 2)


class TestDescribeRule(unittest.TestCase):
    def test_prefix_rule(self):
        self.assertEqual(
            _describe_rule({"type": "prefix", "value": "Inbox/"}), "Inbox/ (and subfolders)"
        )

    def test_direct_root_files_rule(self):
        self.assertEqual(
            _describe_rule({"type": "direct-root-files", "root": "Inbox/"}),
            "messages directly in Inbox/",
        )

    def test_container_root_rule(self):
        self.assertEqual(_describe_rule({"type": "container-root"}), "messages at the mailbox root")

    def test_unknown_rule_type_shown_raw(self):
        self.assertEqual(_describe_rule({"type": "weird"}), "{'type': 'weird'}")


if __name__ == "__main__":
    unittest.main()
