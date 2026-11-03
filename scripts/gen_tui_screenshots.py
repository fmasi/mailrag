#!/usr/bin/env python3
"""Regenerate the full-screen TUI screenshots embedded in the docs.

Boots :class:`src.tui.app.MailragWizardApp` headlessly via Textual's
``App.run_test`` (no display needed — Textual renders straight to SVG) against a
*synthetic* demo mailbox and *faked* pipeline handlers. Nothing here touches a
real inbox, Qdrant, or an LLM: every folder name, persona, and log line is
representative sample data, so no private content can leak into the images.

It reuses the exact harness the pilot tests use (``tests/test_tui_app.py``):
``src.tui.flow.build_handlers`` is patched with recording fakes, and the LLM
calibrate gate's report/rubric lookups are stubbed. The app is driven through
all six stages with ``pilot.press`` and one SVG is written per screen:

    welcome → persona → model → scope → review → run

Usage::

    conda run -n mailrag python scripts/gen_tui_screenshots.py [OUT_DIR]

``OUT_DIR`` defaults to ``docs/images/tui``. The six SVGs are committed so the
docs render without a build step; rerun this script whenever the TUI changes.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import tempfile
import time
from typing import Any, Callable, Dict, List
from unittest import mock

# Make ``src`` importable when run as a plain script from the repo root.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from textual.widgets import Input  # noqa: E402

from src.tui.app import (  # noqa: E402
    CalibrateGateModal,
    ConfirmModal,
    MailragWizardApp,
    ModelScreen,
    PersonaScreen,
    ReviewScreen,
    RunScreen,
    ScopeScreen,
    WelcomeScreen,
)

#: A terminal size wide enough for the two-panel screens to breathe.
_SIZE = (120, 40)

#: The screens captured, in flow order — used for logging and the return value.
STAGES = ("welcome", "persona", "model", "scope", "review", "run")

#: A synthetic mailbox: representative folder names only, no real mail. The
#: shape (a folder with both direct files and subfolders, flat folders, and a
#: root-level message) exercises every kind of row the scope tree can render.
_DEMO_EMLS = (
    "readme.eml",  # a message sitting at the mailbox root
    "Inbox/welcome.eml",  # direct files under a folder that also has subfolders
    "Inbox/Family/reunion.eml",
    "Inbox/Receipts/amazon-order.eml",
    "Newsletters/weekly-digest.eml",  # a flat folder (no subfolders)
    "Projects/Acme/kickoff.eml",
    "Projects/Zephyr/roadmap.eml",
    "Archive/2019-taxes.eml",
)


#: Sample step results, typed like the real handlers' returns so the run log
#: shows exactly the sentences the app writes (flow.describe_result).
def _fake_results() -> Dict[str, Any]:
    from src.indexing.contextual_index import BuildResult
    from src.pipeline.profile import ProfileReport

    return {
        "scope": [{"type": "prefix", "value": "Inbox/"}, {"type": "prefix", "value": "Projects/"}],
        "measure": ProfileReport(
            percentiles={50: 212}, mean=290.0, bodies=5120, suggested_chunk_size=512
        ),
        "summarize": {"cached": 0, "done": 5120, "error": 0, "unavailable": 0},
        "prune": 1204,
        "index": BuildResult(collection="acme-mail", kept_emails=3916, chunks=8840),
    }


def _fake_calibration_text() -> str:
    """A representative report through the real formatter (synthetic rows)."""
    from src.llm.calibration import CalibrationReport, format_report

    nag = {
        "sender": "Vault <noreply@example.com>",
        "subject": "Improve your password",
        "summary": "security reminder",
    }
    return format_report(
        CalibrationReport(
            rubric="personal",
            sample=200,
            noise_rate=0.38,
            false_noise=[
                {"sender": "Shop <orders@example.com>", "subject": "Your order", "reason": "promo"}
            ],
            false_keep=[nag] * 6,
        )
    )


def _fake_handlers(calls: List[str]) -> Dict[str, Callable[..., Any]]:
    """Recording fakes for every verb; ``prune`` never opens a confirm modal.

    Mirrors ``tests/test_tui_app.py``'s recorder, but deliberately ignores the
    ``prune_confirm`` callback so the run reaches completion through only the
    two unavoidable gates (calibrate, confirm-before-spend) — keeping the run
    screenshot a clean, finished ladder."""

    def make(verb: str) -> Callable[..., Any]:
        def handler(prof: Any, **params: Any) -> Any:
            calls.append(verb)
            return _fake_results().get(verb, f"{verb} ok")

        return handler

    verbs = (
        "scope",
        "measure",
        "tag",
        "scan",
        "calibrate",
        "summarize",
        "index",
        "judge",
        "prune",
        "attach",
    )
    return {v: make(v) for v in verbs}


def _write_demo_mailbox(root: str) -> None:
    for rel in _DEMO_EMLS:
        path = os.path.join(root, *rel.split("/"))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("Subject: sample\nFrom: someone@example.com\n\nbody\n")


async def _wait_for(pilot: Any, predicate: Callable[[], bool], timeout: float = 10.0) -> None:
    """Poll *predicate* while the app processes events (worker + modals)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        await pilot.pause(0.05)
    raise TimeoutError("timed out waiting for the app to reach the expected state")


async def generate(out_dir: str) -> List[str]:
    """Drive the wizard through all six stages, writing one SVG per screen.

    Returns the list of written file paths. Runs entirely in a temporary demo
    directory (restored on exit) so the on-screen profile/root paths read as
    clean relative names rather than a random temp path."""
    os.makedirs(out_dir, exist_ok=True)
    out_dir = os.path.abspath(out_dir)
    written: List[str] = []

    demo = tempfile.mkdtemp(prefix="mailrag-tui-demo-")
    prev_cwd = os.getcwd()
    calls: List[str] = []
    try:
        # Build the demo mailbox + profile with clean, relative-looking names.
        _write_demo_mailbox(os.path.join(demo, "acme.mbox"))
        profile_path = "acme.profile.json"
        with open(os.path.join(demo, profile_path), "w", encoding="utf-8") as fh:
            json.dump({"root": "acme.mbox", "rubric": "personal", "collection": "acme-mail"}, fh)
        # A scan artifact so Welcome shows a recommendation and Persona pre-picks it.
        with open(os.path.join(demo, "acme.profile.scan.json"), "w", encoding="utf-8") as fh:
            json.dump({"recommended_persona": "llm-all"}, fh)
        os.chdir(demo)

        with (
            mock.patch(
                "src.tui.flow.build_handlers", side_effect=lambda **_: _fake_handlers(calls)
            ),
            mock.patch("src.llm.calibration.format_report", return_value=_fake_calibration_text()),
            mock.patch("src.llm.rubrics.names", return_value=["personal", "work", "legal"]),
        ):
            app = MailragWizardApp(profile_path)

            def shot(name: str) -> None:
                app.save_screenshot(filename=f"{name}.svg", path=out_dir)
                written.append(os.path.join(out_dir, f"{name}.svg"))

            async with app.run_test(size=_SIZE) as pilot:
                # 1. Welcome ---------------------------------------------------
                await _wait_for(pilot, lambda: isinstance(app.screen, WelcomeScreen))
                await pilot.pause()
                shot("welcome")

                # 2. Persona (llm-all is recommended -> pre-highlighted, starred)
                await pilot.press("enter")
                await _wait_for(pilot, lambda: isinstance(app.screen, PersonaScreen))
                await pilot.pause()
                shot("persona")

                # 3. Model — accept the highlighted llm-all persona ------------
                await pilot.press("enter")
                await _wait_for(pilot, lambda: isinstance(app.screen, ModelScreen))
                app.screen.query_one("#model-input", Input).value = "qwen/qwen3-4b-2507"
                await pilot.pause()
                shot("model")

                # 4. Scope — check Inbox/ (covers its children) + Projects/Acme/
                await pilot.press("enter")
                await _wait_for(pilot, lambda: isinstance(app.screen, ScopeScreen))
                # tree rows: 0 root-files, 1 Archive/, 2 Inbox/, 3 (direct),
                # 4 Inbox/Family/, 5 Inbox/Receipts/, 6 Newsletters/,
                # 7 Projects/, 8 Projects/Acme/, 9 Projects/Zephyr/
                await pilot.press("down", "down", "space")  # check Inbox/
                for _ in range(6):
                    await pilot.press("down")  # -> Projects/Acme/
                await pilot.press("space")  # check Projects/Acme/
                await pilot.pause()
                shot("scope")

                # 5. Review ----------------------------------------------------
                await pilot.press("c")
                await _wait_for(pilot, lambda: isinstance(app.screen, ReviewScreen))
                await pilot.pause()
                shot("review")

                # 6. Run — answer the two gates, then capture the finished ladder
                await pilot.press("enter")
                await _wait_for(pilot, lambda: isinstance(app.screen, CalibrateGateModal))
                gate = app.screen
                await pilot.press("p")  # proceed past the calibrate gate
                await _wait_for(pilot, lambda: app.screen is not gate)
                await _wait_for(pilot, lambda: isinstance(app.screen, ConfirmModal))
                spend = app.screen
                await pilot.press("y")  # confirm-before-spend
                await _wait_for(pilot, lambda: app.screen is not spend)
                await _wait_for(
                    pilot,
                    lambda: isinstance(app.screen, RunScreen) and app.screen._exit_code is not None,
                )
                await pilot.pause()
                shot("run")
    finally:
        os.chdir(prev_cwd)
        shutil.rmtree(demo, ignore_errors=True)

    return written


def main() -> int:
    out_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_REPO_ROOT, "docs/images/tui")
    written = asyncio.run(generate(out_dir))
    for path in written:
        size = os.path.getsize(path)
        status = "ok" if size > 0 else "EMPTY"
        print(f"  {status:>5}  {path}  ({size} bytes)")
    missing = [s for s in STAGES if not os.path.exists(os.path.join(out_dir, f"{s}.svg"))]
    if missing:
        print(f"error: missing screenshots: {', '.join(missing)}", file=sys.stderr)
        return 1
    print(f"wrote {len(written)} screenshots to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
