"""Interactive persona wizard (the TUI) — a thin `questionary`+`rich` runner.

Asks "which persona?" first (surfacing `scan`'s recommendation if a scan ran),
then walks the recipe in order, rendering each step. Two interaction points the
plain `run` verb does not have:

* the **calibrate gate** — after `calibrate` shows its FALSE-NOISE/FALSE-KEEP
  buckets, the user may proceed, re-tune (pick another rubric and re-calibrate),
  or abort;
* a **confirm-before-spend** prompt before the big `summarize` LLM pass.

`questionary` is injected (like the `select` stage) so the flow is unit-testable.
Heavy work is delegated to the same handler map the headless `run` verb uses.
See docs/VERBS.md.
"""
from __future__ import annotations

import json
import os
from typing import Any, Optional

from src.profile import CorpusProfile
from src.persona.registry import load_registry
from src.persona.executor import missing_handlers
from src.persona.runner import build_handlers

# Verbs that consume the LLM (so the wizard knows to ask for a model first).
_LLM_STEPS = {"calibrate", "summarize", "judge"}


def _read_recommendation(profile_path: str) -> Optional[str]:
    """Return the persona `scan` recommended, if a scan artifact sits beside the profile."""
    path = profile_path.rsplit(".", 1)[0] + ".scan.json"
    if not os.path.exists(path):
        return None
    try:
        with open(path) as fh:
            return json.load(fh).get("recommended_persona")
    except (OSError, ValueError):
        return None


def _calibrate_gate(prof, handler, reg, q, console) -> str:
    """Run calibrate, show the buckets, and loop on the user's choice.

    Returns "proceed" or "abort"."""
    from src.llm import calibration as calibration_lib
    from src.llm import rubrics
    while True:
        report = handler(prof)                      # runs calibrate, records on prof
        console.print(calibration_lib.format_report(report))
        choice = q.select(
            "Calibration done. What next?",
            choices=["proceed to the LLM pass",
                     "re-tune (pick another rubric and re-calibrate)",
                     "abort"]).ask()
        if choice is None or choice.startswith("abort"):
            return "abort"
        if choice.startswith("proceed"):
            return "proceed"
        # re-tune: choose a different rubric and loop
        names = rubrics.names()
        new = q.select("Pick a rubric to try:", choices=names).ask()
        if new:
            prof.rubric = new
            console.print(f"rubric -> {new}; re-calibrating")


def run_wizard(profile_path: str, *, questionary: Any = None, registry: Any = None,
               model: Optional[str] = None, console: Any = None) -> int:
    """Drive the interactive persona flow. Returns a process exit code."""
    if questionary is None:
        import questionary as questionary  # noqa: PLC0414
    q = questionary
    reg = registry or load_registry()
    if console is None:
        from rich.console import Console
        console = Console()

    rec = _read_recommendation(profile_path)
    if rec:
        console.print(f"scan recommends: [bold]{rec}[/bold]")
    for name in reg.names():
        p = reg.get(name)
        console.print(f"  [bold]{name}[/bold] — {p.label}: {p.advisor_hint}")

    chosen = q.select("Choose a persona:", choices=reg.names(),
                      default=rec if rec in reg.names() else None).ask()
    if not chosen:
        return 1
    persona = reg.get(chosen)

    needs_model = any(s.verb in _LLM_STEPS for s in persona.steps)
    if needs_model and not model:
        model = q.text("LLM model id for the LLM steps:").ask()
        if not model:
            console.print("no model given; aborting")
            return 2

    handlers = build_handlers(profile_path=profile_path, model=model)
    missing = missing_handlers(persona, handlers)
    if missing:
        console.print(f"persona '{persona.name}' needs verb(s) not yet implemented: "
                      f"{', '.join(missing)} — try 'llm-none' or 'llm-all'.")
        return 2

    prof = CorpusProfile.load(profile_path)
    for step in persona.steps:
        if step.verb not in handlers:           # optional + unimplemented -> skip
            console.print(f"  – skip {step.verb} (optional)")
            continue
        if step.verb == "calibrate":
            if _calibrate_gate(prof, handlers["calibrate"], reg, q, console) == "abort":
                console.print("aborted at the calibration gate")
                prof.save(profile_path)
                return 1
            continue
        if step.verb == "summarize":
            if not q.confirm("Run the LLM summary pass over the keep set?").ask():
                console.print("stopped before the LLM summary pass")
                prof.save(profile_path)
                return 1
        params = {k: v for k, v in step.params.items() if k != "optional"}
        console.print(f"  ▶ {step.verb} — {reg.verb_info(step.verb).does}")
        handlers[step.verb](prof, **params)

    prof.save(profile_path)
    console.print(f"persona '{persona.name}' complete -> {profile_path}")
    return 0
