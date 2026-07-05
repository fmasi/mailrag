"""Presentation-free logic behind the full-screen wizard (``src/tui/app.py``).

The Textual app is a thin view: everything it renders or decides — persona
cards, the scope tree and its rule conversion, the planned step list, the
calibrate / confirm-before-spend / prune gates — is computed here so the flow
can be unit-tested headlessly (``tests/test_tui_flow.py``), exactly like the
classic questionary wizard before it (``src/persona/wizard.py``, kept as
``mailrag wizard --classic``).

Heavy work still goes through the same verb->handler map the headless ``run``
verb uses (:func:`src.persona.runner.build_handlers`); the only override is
``scope``, whose folder picking happens on a dedicated screen *before* the run
instead of prompt-by-prompt inside the handler. See docs/VERBS.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence, Tuple

from src.ingest.selection import discover_structure, list_eml_relpaths
from src.persona.registry import Persona, Registry
from src.persona.runner import build_handlers
from src.persona.wizard import read_recommendation

__all__ = [
    "LLM_STEPS",
    "PersonaCard",
    "PlannedStep",
    "ScopeNode",
    "WizardUI",
    "build_scope_tree",
    "execute_plan",
    "make_scope_handler",
    "needs_model",
    "persona_cards",
    "plan_steps",
    "prepare_handlers",
    "read_recommendation",
    "run_calibrate_gate",
    "selection_to_rules",
    "short_result",
    "validate_model",
]

#: Verbs that consume the LLM (the wizard asks for a model before running these).
LLM_STEPS = {"calibrate", "summarize", "judge"}

Handler = Callable[..., Any]


# --- persona picking ---------------------------------------------------------


@dataclass(frozen=True)
class PersonaCard:
    """Display-ready persona summary for the picker screen."""

    name: str
    label: str
    hint: str
    verbs: List[str]
    costs: List[str]  # aligned with ``verbs``
    recommended: bool = False


def persona_cards(registry: Registry, recommended: Optional[str] = None) -> List[PersonaCard]:
    """One card per persona, flagging the one `scan` recommended (if any)."""
    cards: List[PersonaCard] = []
    for name in registry.names():
        p = registry.get(name)
        cards.append(
            PersonaCard(
                name=name,
                label=p.label,
                hint=p.advisor_hint,
                verbs=[s.verb for s in p.steps],
                costs=[registry.verb_info(s.verb).cost for s in p.steps],
                recommended=(name == recommended),
            )
        )
    return cards


def needs_model(persona: Persona) -> bool:
    """True when the persona has at least one LLM-consuming step."""
    return any(s.verb in LLM_STEPS for s in persona.steps)


def validate_model(text: Optional[str]) -> Optional[str]:
    """Return the cleaned model id, or ``None`` when blank / whitespace-only."""
    cleaned = (text or "").strip()
    return cleaned or None


# --- scope picking -----------------------------------------------------------


@dataclass
class ScopeNode:
    """One selectable row of the scope tree; ``rule`` is the rule checking it grants."""

    node_id: str
    label: str
    rule: Dict[str, Any]
    children: List["ScopeNode"] = field(default_factory=list)


def build_scope_tree(root: str) -> Tuple[List[ScopeNode], bool]:
    """``(top_level_nodes, has_root_files)`` for the mailbox under *root*.

    Mirrors the classic guided picker's shape (top-level folders, their level-2
    children, plus a pseudo-node for messages sitting directly in a folder that
    also has subfolders). An empty list means no ``.eml`` files were found."""
    folder_tree, has_root = discover_structure(list_eml_relpaths(root))
    nodes: List[ScopeNode] = []
    for prefix in sorted(folder_tree):
        entry = folder_tree[prefix]
        children: List[ScopeNode] = []
        if entry["children"] and entry["has_direct_files"]:
            children.append(
                ScopeNode(
                    node_id=f"{prefix}::direct",
                    label="(messages directly in this folder)",
                    rule={"type": "direct-root-files", "root": prefix},
                )
            )
        for child in sorted(entry["children"]):
            children.append(
                ScopeNode(node_id=child, label=child, rule={"type": "prefix", "value": child})
            )
        nodes.append(
            ScopeNode(
                node_id=prefix,
                label=prefix,
                rule={"type": "prefix", "value": prefix},
                children=children,
            )
        )
    return nodes, has_root


def selection_to_rules(
    nodes: Sequence[ScopeNode],
    checked: Sequence[str],
    *,
    include_root_files: bool = False,
) -> List[Dict[str, Any]]:
    """Convert the checked node-id set into selection rules (see ingest/selection.py).

    A checked top-level folder wins over its children (one whole-prefix rule);
    otherwise each checked child contributes its own rule."""
    chosen = set(checked)
    rules: List[Dict[str, Any]] = []
    if include_root_files:
        rules.append({"type": "container-root"})
    for node in nodes:
        if node.node_id in chosen:
            rules.append(dict(node.rule))
            continue
        rules.extend(dict(c.rule) for c in node.children if c.node_id in chosen)
    return rules


# --- run plan ----------------------------------------------------------------


@dataclass(frozen=True)
class PlannedStep:
    """A persona step resolved against the handler map, ready to render/run."""

    verb: str
    params: Dict[str, Any]
    does: str
    cost: str
    skipped: bool = False  # optional step whose verb has no handler yet


def plan_steps(
    persona: Persona, handlers: Dict[str, Handler], registry: Registry
) -> List[PlannedStep]:
    """Resolve *persona*'s recipe into display/run-ready steps.

    A step with no handler is marked ``skipped``; required-but-missing verbs are
    the caller's problem (check :func:`src.persona.executor.missing_handlers`
    before running — the review screen blocks on them)."""
    planned: List[PlannedStep] = []
    for step in persona.steps:
        info = registry.verb_info(step.verb)
        planned.append(
            PlannedStep(
                verb=step.verb,
                params={k: v for k, v in step.params.items() if k != "optional"},
                does=info.does,
                cost=info.cost,
                skipped=step.verb not in handlers,
            )
        )
    return planned


def make_scope_handler(rules: Sequence[Dict[str, Any]]) -> Handler:
    """A ``scope`` handler that applies rules already collected by the scope screen.

    (The classic flow prompts folder-by-folder *inside* the handler; the TUI
    picks everything up front on a tree, so the handler just writes the rules.)"""

    def _scope(prof: Any, **_: Any) -> List[Dict[str, Any]]:
        prof.selection_rules = [dict(r) for r in rules]
        return prof.selection_rules

    return _scope


def prepare_handlers(
    *,
    profile_path: str,
    model: Optional[str] = None,
    limit: Optional[int] = None,
    scope_rules: Optional[Sequence[Dict[str, Any]]] = None,
    confirm_prune: Optional[Callable[[List[str]], bool]] = None,
) -> Dict[str, Handler]:
    """The shared handler map, with ``scope`` overridden when rules were pre-picked."""
    handlers = build_handlers(
        profile_path=profile_path, model=model, prune_confirm=confirm_prune, limit=limit
    )
    if scope_rules is not None:
        handlers["scope"] = make_scope_handler(scope_rules)
    return handlers


# --- execution (the run screen's engine) --------------------------------------


class WizardUI(Protocol):
    """What the run loop needs from a front-end (the Textual bridge, or a test fake).

    The three gates block until the user answers; everything else is
    fire-and-forget progress reporting."""

    def on_step_start(self, index: int, step: PlannedStep) -> None: ...

    def on_step_done(self, index: int, step: PlannedStep, result: Any) -> None: ...

    def on_step_skip(self, index: int, step: PlannedStep) -> None: ...

    def log(self, message: str) -> None: ...

    def calibrate_gate(self, report_text: str) -> Optional[str]:
        """Show the calibration buckets; return 'proceed', 'retune' or 'abort'."""
        ...

    def pick_rubric(self, names: Sequence[str], current: str) -> Optional[str]: ...

    def confirm_spend(self) -> bool: ...


def run_calibrate_gate(prof: Any, handler: Handler, ui: WizardUI) -> str:
    """Calibrate, show the buckets, and loop on re-tune (same as the classic gate).

    Returns ``"proceed"`` or ``"abort"``."""
    from src.llm import calibration as calibration_lib
    from src.llm import rubrics

    while True:
        report = handler(prof)  # runs calibrate, records on prof
        decision = ui.calibrate_gate(calibration_lib.format_report(report))
        if decision == "proceed":
            return "proceed"
        if decision != "retune":  # abort, or the gate was dismissed
            return "abort"
        new = ui.pick_rubric(rubrics.names(), prof.rubric)
        if new:
            prof.rubric = new
            ui.log(f"rubric -> {new}; re-calibrating")


def execute_plan(
    prof: Any,
    profile_path: str,
    planned: Sequence[PlannedStep],
    handlers: Dict[str, Handler],
    ui: WizardUI,
) -> int:
    """Run the planned steps with the two human gates; mirrors the classic wizard.

    Returns a process exit code (0 done, 1 user-aborted). The profile is saved
    on every exit path so partial progress (scope rules, calibration) is kept."""
    for index, step in enumerate(planned):
        if step.skipped:
            ui.on_step_skip(index, step)
            ui.log(f"skip {step.verb} (optional, not implemented yet)")
            continue
        if step.verb == "calibrate":
            ui.on_step_start(index, step)
            if run_calibrate_gate(prof, handlers["calibrate"], ui) == "abort":
                ui.log("aborted at the calibration gate")
                prof.save(profile_path)
                return 1
            ui.on_step_done(index, step, None)
            continue
        if step.verb == "summarize" and not ui.confirm_spend():
            ui.log("stopped before the LLM summary pass")
            prof.save(profile_path)
            return 1
        ui.on_step_start(index, step)
        result = handlers[step.verb](prof, **step.params)
        ui.on_step_done(index, step, result)
    prof.save(profile_path)
    return 0


def short_result(result: Any, limit: int = 140) -> str:
    """One whitespace-collapsed line describing a step result, for the run log."""
    text = " ".join(str(result).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"
