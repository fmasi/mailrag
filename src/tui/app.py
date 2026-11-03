"""Full-screen Textual wizard — the interactive face of the persona pipeline.

Replaces the scrolling questionary flow (still available as ``mailrag wizard
--classic``) with a guided app::

    Welcome -> Persona -> Model (LLM personas) -> Scope -> Review -> Run

Every decision is computed in :mod:`src.tui.flow`; this module is view code
only. The run screen executes the shared verb handlers in a worker thread and
keeps the UI live; the three human gates (calibrate, confirm-before-spend,
prune) surface as modal dialogs, marshalled across the thread boundary by
:class:`_WorkerBridge`. Pilot tests drive the whole app headlessly in
``tests/test_tui_app.py``. See docs/GUIDE.md for the user-facing tour.
"""

from __future__ import annotations

import shlex
import sys
import threading
import traceback
from dataclasses import dataclass
from typing import Any, Callable, ClassVar, Dict, List, Optional, Sequence

from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Center, Horizontal, Vertical, VerticalScroll
from textual.markup import escape
from textual.screen import ModalScreen, Screen
from textual.widgets import Footer, Input, OptionList, ProgressBar, RichLog, Static, Tree
from textual.widgets.option_list import Option

from src.persona.executor import missing_handlers
from src.persona.registry import Persona, Registry, load_registry
from src.profile import CorpusProfile
from src.tui import flow


def make_progress_bars_safe() -> None:
    """Give tqdm a thread lock before any step draws a progress bar.

    While the app runs, Textual replaces ``sys.stderr`` with a capture whose
    ``fileno()`` is invalid. tqdm's first bar in a process creates a
    multiprocessing lock, which starts a helper process and passes it stderr's
    file descriptor, and that fails with ``ValueError: bad value(s) in
    fds_to_keep``. Calibrate crashed that way, and summarize and index create
    bars too, ours and the embedding library's. The steps run in threads, never
    in processes, so a thread lock is all tqdm needs. It is process-wide, which
    is why this runs before the first step rather than around each one.
    """
    from tqdm import tqdm

    tqdm.set_lock(threading.RLock())


class _Nav:
    """Navigation sentinels returned by screens (identity-compared)."""

    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:  # pragma: no cover - debugging nicety
        return f"<nav {self.name}>"


BACK = _Nav("back")
QUIT = _Nav("quit")
#: Quit from a review that cannot run (missing verbs) — maps to exit code 2,
#: matching the classic wizard's "not yet implemented" exit.
BLOCKED = _Nav("blocked")

#: Stage names shown in the breadcrumb bar, in order.
STAGES = ["Welcome", "Persona", "Model", "Scope", "Review", "Run"]

_COST_STYLE = {
    "free": "green",
    "cheap": "cyan",
    "small-llm": "yellow",
    "big-llm": "orange1",
    "gpu": "magenta",
}


#: What each cost tier means to someone who has never seen the pipeline.
_COST_LABEL = {
    "free": "free",
    "cheap": "no AI, GPU minutes",
    "small-llm": "AI, some emails",
    "big-llm": "AI, every email",
    "gpu": "GPU, minutes to hours",
}


def _cost_badge(cost: str) -> str:
    return f"[{_COST_STYLE.get(cost, 'white')}]{_COST_LABEL.get(cost, cost or '?')}[/]"


@dataclass
class WizardState:
    """Everything the guided flow collects before (and while) running."""

    profile_path: str
    profile: CorpusProfile
    model: Optional[str] = None
    limit: Optional[int] = None
    recommendation: Optional[str] = None
    persona_name: Optional[str] = None
    scope_rules: Optional[List[Dict[str, Any]]] = None


class StageBar(Static):
    """Breadcrumb of wizard stages with the current one highlighted.

    Reads the app's ``stage_names`` — the stages that actually apply to the
    chosen persona — so a skipped stage (e.g. Model for a no-LLM persona)
    never shows up as visited."""

    def __init__(self, current: str) -> None:
        super().__init__("", id="stagebar")
        self._current = current

    def on_mount(self) -> None:
        names = getattr(self.app, "stage_names", None) or list(STAGES)
        if self._current not in names:  # pragma: no cover - defensive
            names = list(STAGES)
        parts: List[str] = []
        seen_current = False
        for name in names:
            if name == self._current:
                parts.append(f"[bold reverse] {name} [/]")
                seen_current = True
            elif not seen_current:
                parts.append(f"[green]✓[/] [dim]{name}[/]")
            else:
                parts.append(f"[dim]{name}[/]")
        self.update("  ".join(parts))


class WizardScreen(Screen[Any]):
    """Base chrome for the guided screens: stage bar on top, footer below."""

    stage: ClassVar[str] = "Welcome"

    def compose(self) -> ComposeResult:
        yield StageBar(self.stage)
        yield from self.compose_body()
        yield Footer()

    def compose_body(self) -> ComposeResult:  # pragma: no cover - overridden
        yield from ()


# --- 1. Welcome ---------------------------------------------------------------

_LOGO = """\
┌┬┐┌─┐┬┬  ┬─┐┌─┐┌─┐
│││├─┤││  ├┬┘├─┤│ ┬
┴ ┴┴ ┴┴┴─┘┴└─┴ ┴└─┘"""


class WelcomeScreen(WizardScreen):
    """Home screen: what we're about to onboard, and how to move around."""

    stage = "Welcome"
    BINDINGS = [
        Binding("enter", "begin", "Begin", priority=True),
        Binding("q", "quit_wizard", "Quit"),
    ]

    def __init__(self, state: WizardState, registry: Optional[Registry] = None) -> None:
        super().__init__()
        self._state = state
        self._registry = registry

    def compose_body(self) -> ComposeResult:
        st = self._state
        facts = [
            ("settings file", st.profile_path),
            ("mail folder", st.profile.root or "(not set)"),
            ("junk rules", st.profile.rubric),
            ("search index", st.profile.collection),
            ("database", st.profile.qdrant_url),
        ]
        if st.limit:
            facts.append(("limit", f"first {st.limit} messages (test run)"))
        rows = "\n".join(f"[dim]{k:>14}[/]  {escape(str(v))}" for k, v in facts)
        rec = (
            f"\n\n[green]★ a scan of this mailbox recommends the "
            f"[bold]{escape(self._recommended_label())}[/bold] recipe[/]"
            if st.recommendation
            else ""
        )
        with Center(id="welcome-center"):
            with Vertical(id="welcome-card"):
                yield Static(f"[bold $primary]{_LOGO}[/]", id="logo")
                yield Static("[dim]guided mailbox onboarding[/]", id="tagline")
                yield Static(rows + rec, id="welcome-facts")
                yield Static(
                    "This makes a mailbox searchable, for you and for AI agents. You pick a "
                    "recipe, the folders to include and, for the AI recipes, the model. Then "
                    "it runs, and stops to ask you at the decisions that matter.\n"
                    "[dim]Nothing in your mailbox is changed or deleted.[/]",
                    id="welcome-blurb",
                )
                yield Static("[b]enter[/] begin   [b]q[/] quit", id="welcome-keys")

    def _recommended_label(self) -> str:
        """The recommended persona by the name the picker shows, not its id."""
        name = self._state.recommendation or ""
        if self._registry is None:
            return name
        try:
            return self._registry.get(name).label
        except ValueError:
            return name

    def action_begin(self) -> None:
        self.dismiss(True)

    def action_quit_wizard(self) -> None:
        self.dismiss(QUIT)


# --- 2. Persona ----------------------------------------------------------------


class PersonaScreen(WizardScreen):
    """Pick a persona; the right panel previews its recipe and costs."""

    stage = "Persona"
    BINDINGS = [
        Binding("escape", "back", "Back"),
        Binding("q", "quit_wizard", "Quit"),
    ]

    def __init__(self, state: WizardState, registry: Registry) -> None:
        super().__init__()
        self._state = state
        self._cards = flow.persona_cards(registry, recommended=state.recommendation)

    def compose_body(self) -> ComposeResult:
        options = [
            Option(
                f"{escape(c.label)}  [green]★[/]" if c.recommended else escape(c.label),
                id=c.name,
            )
            for c in self._cards
        ]
        with Horizontal(id="persona-body"):
            with Vertical(id="persona-list-panel"):
                yield Static("[b]Choose a recipe[/]", classes="panel-title")
                yield OptionList(*options, id="persona-list")
            with Vertical(id="persona-detail-panel"):
                yield Static("", id="persona-detail")

    def on_mount(self) -> None:
        picker = self.query_one("#persona-list", OptionList)
        names = [c.name for c in self._cards]
        wanted = self._state.persona_name or self._state.recommendation
        picker.highlighted = names.index(wanted) if wanted in names else 0
        picker.focus()

    def _card(self, name: str) -> Optional[flow.PersonaCard]:
        return next((c for c in self._cards if c.name == name), None)

    def _render_detail(self, name: str) -> None:
        card = self._card(name)
        if card is None:  # pragma: no cover - defensive
            return
        lines = [f"[bold]{escape(card.label)}[/]", escape(card.hint), ""]
        if card.recommended:
            lines.insert(2, "[green]★ a scan of this mailbox recommends this one[/]")
        lines.append("[b]Steps[/] [dim](free ones first, so the AI only sees what is left)[/]")
        for i, (verb, cost, does) in enumerate(zip(card.verbs, card.costs, card.does), start=1):
            lines.append(f"  {i}. {escape(does)}  {_cost_badge(cost)}")
        lines += [
            "",
            "[dim]No AI model running? Choose the first recipe.[/]",
            "[dim]enter[/] choose   [dim]↑/↓[/] browse   [dim]esc[/] back",
        ]
        self.query_one("#persona-detail", Static).update("\n".join(lines))

    @on(OptionList.OptionHighlighted, "#persona-list")
    def _on_highlight(self, event: OptionList.OptionHighlighted) -> None:
        if event.option.id:
            self._render_detail(event.option.id)

    @on(OptionList.OptionSelected, "#persona-list")
    def _on_select(self, event: OptionList.OptionSelected) -> None:
        self.dismiss(event.option.id)

    def action_back(self) -> None:
        self.dismiss(BACK)

    def action_quit_wizard(self) -> None:
        self.dismiss(QUIT)


# --- 3. Model -------------------------------------------------------------------


class ModelScreen(WizardScreen):
    """Ask for the LLM model id (only when the persona has LLM steps)."""

    stage = "Model"
    BINDINGS = [
        Binding("escape", "back", "Back", priority=True),
        Binding("ctrl+q", "quit_wizard", "Quit"),
    ]

    def __init__(self, state: WizardState, persona: Persona, registry: Registry) -> None:
        super().__init__()
        self._state = state
        self._persona = persona
        self._registry = registry

    def compose_body(self) -> ComposeResult:
        llm_steps = [
            self._registry.verb_info(s.verb).does
            for s in self._persona.steps
            if s.verb in flow.LLM_STEPS
        ]
        with Center(id="model-center"):
            with Vertical(id="model-card"):
                yield Static("[b]Which AI model reads your mail?[/]", classes="panel-title")
                yield Static(
                    f"The [bold]{escape(self._persona.label)}[/] recipe uses an AI model to:\n"
                    + "".join(f"  • {escape(d)}\n" for d in llm_steps)
                    + f"Your mail is sent to the AI server at [b]{escape(self._endpoint())}[/]. "
                    "localhost means your own machine; anything else, make sure you "
                    "trust it. To use another server (Ollama, for example), quit and set "
                    "RAG_LLM_API_BASE.\n"
                    "Type the model's name exactly as that server lists it, and keep it "
                    "running while this works.\n"
                    "[dim]LM Studio: the name is under My Models, or run `lms ls`.[/]",
                    id="model-blurb",
                )
                yield Input(
                    value=self._state.model or "",
                    placeholder="model name, e.g. qwen/qwen3-4b-2507",
                    id="model-input",
                )
                yield Static("", id="model-error")

    @staticmethod
    def _endpoint() -> str:
        from src.llm.client import resolve_llm_api_base

        return resolve_llm_api_base()

    def on_mount(self) -> None:
        self.query_one("#model-input", Input).focus()

    @on(Input.Submitted, "#model-input")
    def _on_submit(self, event: Input.Submitted) -> None:
        model = flow.validate_model(event.value)
        if model is None:
            self.query_one("#model-error", Static).update(
                "[red]Type a model name, or press esc and choose the recipe that needs no AI.[/]"
            )
            return
        self.dismiss(model)

    def action_back(self) -> None:
        self.dismiss(BACK)

    def action_quit_wizard(self) -> None:
        self.dismiss(QUIT)


# --- 4. Scope --------------------------------------------------------------------


class ScopeScreen(WizardScreen):
    """Navigable folder tree: check what to index (replaces per-folder prompts)."""

    stage = "Scope"
    BINDINGS = [
        # Named toggle_include (not "toggle") — DOMNode.action_toggle already exists.
        Binding("space", "toggle_include", "Include/exclude", priority=True),
        Binding("c", "continue", "Continue"),
        Binding("escape", "back", "Back"),
        Binding("q", "quit_wizard", "Quit"),
    ]

    ROOT_FILES_ID = "::container-root"

    def __init__(self, state: WizardState) -> None:
        super().__init__()
        self._state = state
        self._scope_nodes, self._has_root = flow.build_scope_tree(state.profile.resolved_root())
        self._checked: set[str] = set()
        # Re-entering the screen restores the previous selection.
        for rule in state.scope_rules or []:
            if rule["type"] == "container-root":
                self._checked.add(self.ROOT_FILES_ID)
            elif rule["type"] == "prefix":
                self._checked.add(rule["value"])
            elif rule["type"] == "direct-root-files":
                self._checked.add(f"{rule['root']}::direct")

    def compose_body(self) -> ComposeResult:
        with Vertical(id="scope-body"):
            yield Static("[b]Which folders should be searchable?[/]", classes="panel-title")
            yield Static(
                "Tick the folders to include. A ticked folder includes everything inside it. "
                "Unticked folders are left out of search.\n"
                "[dim]space[/] tick/untick   [dim]c[/] continue   [dim]esc[/] back",
                id="scope-help",
            )
            if self._scope_nodes or self._has_root:
                yield Tree("mailbox", id="scope-tree")
            else:
                yield Static(
                    f"[red]No .eml files found under "
                    f"{self._state.profile.resolved_root()!r}.[/]\n"
                    "[dim]Point the settings file's `root` at your mail folder and start "
                    "again, or press esc to go back.[/]",
                    id="scope-empty",
                )
            yield Static("", id="scope-status")

    def on_mount(self) -> None:
        if not (self._scope_nodes or self._has_root):
            return
        tree = self.query_one("#scope-tree", Tree)
        tree.show_root = False
        tree.guide_depth = 3
        if self._has_root:
            tree.root.add_leaf("", data=self.ROOT_FILES_ID)
        for node in self._scope_nodes:
            branch = tree.root.add("", data=node.node_id, expand=True)
            for child in node.children:
                branch.add_leaf("", data=child.node_id)
        tree.root.expand()
        self._drop_missing_ticks()
        self._refresh_labels()
        tree.focus()

    def _drop_missing_ticks(self) -> None:
        """Forget restored ticks for folders this tree no longer shows.

        Last run's choice is restored from the profile, and the mailbox may have
        changed since. A tick with no row could never be unticked, yet it still
        became a rule, and a run made only of such rules indexes nothing."""
        shown = {self.ROOT_FILES_ID} if self._has_root else set()
        for node in self._scope_nodes:
            shown.add(node.node_id)
            shown.update(child.node_id for child in node.children)
        missing = self._checked - shown
        if missing:
            self._checked -= missing
            self.notify(
                f"{len(missing)} folder(s) ticked last time are not shown in this tree "
                "and were unticked.",
                severity="warning",
            )

    # -- rendering ------------------------------------------------------------

    def _label_for(self, node_id: str) -> str:
        if node_id == self.ROOT_FILES_ID:
            text = "(messages at the mailbox root)"
        else:
            scope_node = self._find(node_id)
            text = scope_node.label if scope_node else node_id
        text = escape(text)  # folder names may contain markup-significant brackets
        # A checked parent covers this node whatever its own check state is
        # (selection_to_rules gives the parent's whole-prefix rule precedence).
        parent = self._parent_of(node_id)
        if parent and parent in self._checked:
            return f"[dim]▣ {text} (covered by {escape(parent)})[/]"
        if node_id in self._checked:
            return f"[green]▣[/] {text}"
        return f"[dim]☐[/] {text}"

    def _find(self, node_id: str) -> Optional[flow.ScopeNode]:
        for node in self._scope_nodes:
            if node.node_id == node_id:
                return node
            for child in node.children:
                if child.node_id == node_id:
                    return child
        return None

    def _parent_of(self, node_id: str) -> Optional[str]:
        for node in self._scope_nodes:
            if any(c.node_id == node_id for c in node.children):
                return node.node_id
        return None

    def _refresh_labels(self) -> None:
        tree = self.query_one("#scope-tree", Tree)

        def _walk(tree_node: Any) -> None:
            for child in tree_node.children:
                if child.data:
                    child.set_label(self._label_for(str(child.data)))
                _walk(child)

        _walk(tree.root)
        rules = self._rules()
        n = len(rules)
        self.query_one("#scope-status", Static).update(
            f"[green]{n}[/] folder{'s' if n != 1 else ''} ticked. Press [b]c[/] to continue."
            if n
            else "[yellow]Nothing ticked yet.[/] Press [dim]space[/] on a folder to include it."
        )

    # -- selection --------------------------------------------------------------

    def _rules(self) -> List[Dict[str, Any]]:
        return flow.selection_to_rules(
            self._scope_nodes,
            [c for c in self._checked if c != self.ROOT_FILES_ID],
            include_root_files=self.ROOT_FILES_ID in self._checked,
        )

    def _toggle_current(self) -> None:
        tree = self.query_one("#scope-tree", Tree)
        node = tree.cursor_node
        if node is None or node.data is None:
            return
        node_id = str(node.data)
        # Child checks are kept (not cleared) when a parent gets checked:
        # selection_to_rules gives the parent precedence while it is checked,
        # and unchecking the parent restores the child selection unchanged.
        if node_id in self._checked:
            self._checked.discard(node_id)
        else:
            self._checked.add(node_id)
        self._refresh_labels()

    def action_toggle_include(self) -> None:
        if self._scope_nodes or self._has_root:
            self._toggle_current()

    @on(Tree.NodeSelected)
    def _on_node_selected(self, event: Tree.NodeSelected[Any]) -> None:
        self._toggle_current()

    def action_continue(self) -> None:
        if not (self._scope_nodes or self._has_root):
            self.notify("Empty mailbox — press escape to go back.", severity="warning")
            return
        rules = self._rules()
        if not rules:
            self.notify("Select at least one folder first.", severity="warning")
            return
        self.dismiss(rules)

    def action_back(self) -> None:
        self.dismiss(BACK)

    def action_quit_wizard(self) -> None:
        self.dismiss(QUIT)


# --- 5. Review --------------------------------------------------------------------


class ReviewScreen(WizardScreen):
    """Recap every choice and the exact recipe before anything runs."""

    stage = "Review"
    BINDINGS = [
        Binding("enter", "start", "Start run", priority=True),
        Binding("escape", "back", "Back"),
        Binding("q", "quit_wizard", "Quit"),
    ]

    def __init__(self, state: WizardState, registry: Registry) -> None:
        super().__init__()
        self._state = state
        self._personas = registry
        self._persona = registry.get(state.persona_name or "")
        # Building a handler map is cheap and side-effect-free by design (the
        # heavy imports live inside the handlers — see runner.build_handlers),
        # so review builds a throwaway copy just to plan; RunScreen rebuilds
        # one with the interactive prune confirm bound to its worker bridge.
        handlers = flow.prepare_handlers(
            profile_path=state.profile_path,
            model=state.model,
            limit=state.limit,
            scope_rules=state.scope_rules,
        )
        self._planned = flow.plan_steps(self._persona, handlers, registry)
        self._missing = missing_handlers(self._persona, handlers)

    def compose_body(self) -> ComposeResult:
        st = self._state
        facts = [
            ("recipe", escape(self._persona.label)),
            ("AI model", escape(st.model) if st.model else "[dim]none needed[/]"),
            ("folders", f"{len(st.scope_rules or [])} ticked"),
            ("junk rules", escape(st.profile.rubric)),
            ("search index", escape(st.profile.collection)),
            ("how much", f"first {st.limit} messages (a test)" if st.limit else "all of it"),
        ]
        with Horizontal(id="review-body"):
            with Vertical(id="review-config-panel"):
                yield Static("[b]Check before you start[/]", classes="panel-title")
                yield Static("\n".join(f"[dim]{k:>12}[/]  {v}" for k, v in facts))
                for rule in (st.scope_rules or [])[:8]:
                    yield Static(f"[dim]{'':>12}  · {escape(_describe_rule(rule))}[/]")
                extra = len(st.scope_rules or []) - 8
                if extra > 0:
                    yield Static(f"[dim]{'':>12}  · … and {extra} more[/]")
            with Vertical(id="review-plan-panel"):
                yield Static("[b]What will happen[/]", classes="panel-title")
                for i, step in enumerate(self._planned, start=1):
                    marker = "[dim](skipped, not built yet)[/]" if step.skipped else ""
                    yield Static(f"  {i}. {escape(step.does)}  {_cost_badge(step.cost)} {marker}")
                if self._missing:
                    yield Static(
                        f"\n[red]✗ This recipe needs steps that are not built yet "
                        f"({', '.join(self._missing)}). Press esc and choose another.[/]",
                        id="review-missing",
                    )
                else:
                    asks = flow.describe_stops(self._planned)
                    yield Static(f"\n[green]Ready.[/] Press [b]enter[/] to start.\n[dim]{asks}[/]")

    def action_start(self) -> None:
        if self._missing:
            self.notify(
                f"Steps not built yet: {', '.join(self._missing)}. Choose another recipe.",
                severity="error",
            )
            return
        self.dismiss(True)

    def action_back(self) -> None:
        self.dismiss(BACK)

    def action_quit_wizard(self) -> None:
        self.dismiss(BLOCKED if self._missing else QUIT)


def _describe_rule(rule: Dict[str, Any]) -> str:
    """Human-readable one-liner for a selection rule."""
    if rule["type"] == "prefix":
        return f"{rule['value']} (and everything inside)"
    if rule["type"] == "direct-root-files":
        return f"messages directly in {rule['root']}"
    if rule["type"] == "container-root":
        return "messages at the mailbox root"
    return str(rule)  # unknown rule type: show it raw rather than mislabel it


# --- gates (modal dialogs shown during the run) --------------------------------------


class ConfirmModal(ModalScreen[bool]):
    """Yes/no dialog (confirm-before-spend, prune confirmation)."""

    BINDINGS = [
        Binding("y", "yes", "Yes"),
        Binding("n", "no", "No"),
        Binding("escape", "no", "No"),
    ]

    def __init__(
        self, title: str, body: str, *, yes_hint: str = "yes", no_hint: str = "cancel"
    ) -> None:
        super().__init__()
        self._title = title
        self._body = body
        self._yes_hint = yes_hint
        self._no_hint = no_hint

    def compose(self) -> ComposeResult:
        with Vertical(classes="dialog"):
            yield Static(f"[b]{self._title}[/]", classes="dialog-title")
            with VerticalScroll(classes="dialog-scroll"):
                yield Static(self._body)
            yield Static(
                f"[b]y[/] {self._yes_hint}   [b]n[/] {self._no_hint}", classes="dialog-keys"
            )

    def action_yes(self) -> None:
        self.dismiss(True)

    def action_no(self) -> None:
        self.dismiss(False)


class CalibrateGateModal(ModalScreen[str]):
    """The calibrate gate: inspect the sample buckets, then proceed/re-tune/abort."""

    BINDINGS = [
        Binding("p", "proceed", "Proceed"),
        Binding("r", "retune", "Re-tune"),
        Binding("a", "abort", "Abort"),
        Binding("escape", "abort", "Abort"),
    ]

    def __init__(self, report_text: str) -> None:
        super().__init__()
        self._report = report_text

    def compose(self) -> ComposeResult:
        with Vertical(classes="dialog dialog-wide"):
            yield Static("[b]Check the AI's decisions on a sample[/]", classes="dialog-title")
            yield Static(
                "The AI applied the junk rules to a sample of your mail. What it is sure is "
                "junk is left out of search. Below are the decisions most likely to be "
                "wrong; scroll to see both lists.\n"
                "• [b]Marked junk, but might be a real record[/]: if real receipts, invoices "
                "or account notices are here, the rules are too strict.\n"
                "• [b]Kept, but might be junk[/]: if newsletters or ads are here, the rules "
                "are too loose. Real conversations here are fine.",
                classes="dialog-intro",
            )
            with VerticalScroll(classes="dialog-scroll"):
                # The report quotes email subjects/senders — escape their brackets.
                yield Static(escape(self._report))
            yield Static(
                "[b]p[/] looks right, carry on   [b]r[/] try other junk rules   [b]a[/] stop here",
                classes="dialog-keys",
            )

    def action_proceed(self) -> None:
        self.dismiss("proceed")

    def action_retune(self) -> None:
        self.dismiss("retune")

    def action_abort(self) -> None:
        self.dismiss("abort")


class RubricPickModal(ModalScreen[Optional[str]]):
    """Pick a rubric to re-calibrate with."""

    BINDINGS = [Binding("escape", "cancel", "Cancel")]

    def __init__(self, names: Sequence[str], current: str) -> None:
        super().__init__()
        self._names = list(names)
        self._current = current

    def compose(self) -> ComposeResult:
        options = [
            Option(f"{escape(n)}  [dim](current)[/]" if n == self._current else escape(n), id=n)
            for n in self._names
        ]
        with Vertical(classes="dialog"):
            yield Static("[b]Pick the junk rules to try[/]", classes="dialog-title")
            yield Static(
                "[dim]Each set of rules is a file in rubrics/ (yours in rubrics/local/). "
                "The sample is checked again with the one you pick.[/]",
                classes="dialog-intro",
            )
            yield OptionList(*options, id="rubric-list")
            yield Static("[b]enter[/] choose   [b]esc[/] keep current", classes="dialog-keys")

    def on_mount(self) -> None:
        self.query_one("#rubric-list", OptionList).focus()

    @on(OptionList.OptionSelected, "#rubric-list")
    def _on_select(self, event: OptionList.OptionSelected) -> None:
        self.dismiss(event.option.id)

    def action_cancel(self) -> None:
        self.dismiss(None)


# --- 6. Run ----------------------------------------------------------------------


class _WorkerBridge:
    """Thread-side implementation of :class:`src.tui.flow.WizardUI`.

    ``flow.execute_plan`` runs in a worker thread. Progress calls are marshalled
    onto the UI thread with ``call_from_thread``; the gate methods push a modal
    and block the worker on an Event until the user answers."""

    def __init__(self, screen: "RunScreen") -> None:
        self._screen = screen

    def _ui(self, fn: Callable[..., None], *args: Any) -> None:
        self._screen.app.call_from_thread(fn, *args)

    def _ask(self, make_modal: Callable[[], ModalScreen[Any]]) -> Any:
        answered = threading.Event()
        holder: Dict[str, Any] = {}

        def _push() -> None:
            def _done(value: Any) -> None:
                holder["value"] = value
                answered.set()

            self._screen.app.push_screen(make_modal(), _done)

        self._screen.app.call_from_thread(_push)
        # Poll-wait so the worker can't strand the process if the app is torn
        # down (Ctrl+Q, test shutdown) while a gate is still on screen.
        while not answered.wait(0.25):
            if not self._screen.app.is_running:
                return None
        return holder.get("value")

    # -- flow.WizardUI ---------------------------------------------------------

    def on_step_start(self, index: int, step: flow.PlannedStep) -> None:
        self._ui(self._screen.mark_step, index, "running")
        self._ui(self._screen.write_log, f"[bold cyan]▶ {escape(step.does)}[/]")

    def on_step_done(self, index: int, step: flow.PlannedStep, result: Any) -> None:
        self._ui(self._screen.mark_step, index, "done")
        if result is not None:
            self._ui(
                self._screen.write_log,
                f"[dim]  {escape(flow.describe_result(result, step.verb))}[/]",
            )

    def on_step_skip(self, index: int, step: flow.PlannedStep) -> None:
        self._ui(self._screen.mark_step, index, "skipped")

    def log(self, message: str) -> None:
        self._ui(self._screen.write_log, message)

    def calibrate_gate(self, report_text: str) -> Optional[str]:
        return self._ask(lambda: CalibrateGateModal(report_text))

    def pick_rubric(self, names: Sequence[str], current: str) -> Optional[str]:
        return self._ask(lambda: RubricPickModal(names, current))

    def confirm_spend(self) -> bool:
        return bool(
            self._ask(
                lambda: ConfirmModal(
                    "Let the AI read every email?",
                    "This is the slow step: one AI call for each email it hasn't read "
                    "yet. Emails it already read are skipped, so stopping and running "
                    "again later loses nothing.",
                    yes_hint="start",
                    no_hint="stop here",
                )
            )
        )

    def confirm_prune(self, preview: List[str]) -> bool:
        # Preview lines quote email senders/subjects — escape their brackets.
        if self._screen.ai_judges_everything():
            if_no = (
                "No changes nothing about this run: the AI's confident junk is left "
                "out of search either way."
            )
        else:
            if_no = "No keeps them: they will be searchable along with everything else."
        if self._screen.ai_judges_everything():
            who = "the AI judged"
            legend = "Each line: how sure the AI is (0 to 1), then its reason.\n"
            # build.py re-applies the cached verdicts whatever the list says.
            undo = (
                "Deleting the list does not bring these back: the AI's junk calls are "
                "kept in its cache and applied again at every rebuild."
            )
        else:
            who = "the header rules flagged"
            legend = ""
            undo = (
                "The list builds up over every run: deleting it brings back everything "
                "ever skipped, from the next rebuild."
            )
        skip_list = self._screen.skip_list_path()
        body = (
            f"These are emails {who} as junk (a sample is shown). Yes adds them to the "
            f"skip list ({escape(skip_list)}), so later runs and rebuilds pass over them "
            f"too. Nothing is deleted from your mailbox. {undo}\n[dim]{if_no}[/]\n\n"
            f"{legend}"
        ) + "\n".join(f"  [dim]{escape(line)}[/]" for line in preview)
        return bool(
            self._ask(
                lambda: ConfirmModal(
                    "Put the junk on the skip list?",
                    body,
                    yes_hint="add to skip list",
                    no_hint="skip this step",
                )
            )
        )


_STEP_ICON = {
    "pending": "[dim]○[/]",
    "running": "[cyan]▶[/]",
    "done": "[green]✓[/]",
    "skipped": "[dim]–[/]",
    "failed": "[red]✗[/]",
}


class RunScreen(Screen[int]):
    """Live run view: step ladder on the left, streaming log on the right."""

    BINDINGS = [
        Binding("enter", "finish", "Finish", priority=True),
        Binding("q", "finish", "Quit"),
    ]

    def __init__(self, state: WizardState, registry: Registry) -> None:
        super().__init__()
        self._state = state
        self._personas = registry
        self._persona = registry.get(state.persona_name or "")
        self._bridge = _WorkerBridge(self)
        self._handlers = flow.prepare_handlers(
            profile_path=state.profile_path,
            model=state.model,
            limit=state.limit,
            scope_rules=state.scope_rules,
            confirm_prune=self._bridge.confirm_prune,
        )
        self._planned = flow.plan_steps(self._persona, self._handlers, registry)
        self._exit_code: Optional[int] = None

    def compose(self) -> ComposeResult:
        yield StageBar("Run")
        with Horizontal(id="run-body"):
            with VerticalScroll(id="run-steps-panel"):
                yield Static(
                    f"[b]{self._persona.name}[/] — {escape(self._persona.label)}",
                    classes="panel-title",
                )
                for i, step in enumerate(self._planned):
                    yield Static("", id=f"step-{i}", classes="step-row")
            with Vertical(id="run-log-panel"):
                yield Static("[b]Log[/]", classes="panel-title")
                # min_width defaults to 78, wider than this panel on most screens,
                # so long lines were cut off at the edge instead of wrapping.
                yield RichLog(id="run-log", markup=True, wrap=True, min_width=20)
        yield ProgressBar(total=len(self._planned), show_eta=False, id="run-progress")
        yield Static("[dim]running… the log on the right shows each step.[/]", id="run-status")
        yield Footer()

    def on_mount(self) -> None:
        for i in range(len(self._planned)):
            self.mark_step(i, "pending")
        self.write_log(f"[dim]profile: {escape(self._state.profile_path)}[/]")
        self.run_worker(self._execute, thread=True, exclusive=True)

    # -- UI-thread helpers (called via call_from_thread) -------------------------

    def mark_step(self, index: int, status: str) -> None:
        step = self._planned[index]
        row = self.query_one(f"#step-{index}", Static)
        text = f" {_STEP_ICON[status]} {escape(step.does)}"
        if status == "skipped":
            text = f" {_STEP_ICON[status]} [dim]{escape(step.does)} (skipped)[/]"
        row.update(text)
        if status in ("done", "skipped"):
            self.query_one("#run-progress", ProgressBar).advance(1)

    def ai_judges_everything(self) -> bool:
        """Did the AI make the junk calls prune is about to apply? Then the
        index leaves that confident junk out whether or not prune writes a skip
        list: the verdicts sit in the cache either way. Decided by prune's own
        ``from`` setting, not by which steps the recipe has."""
        sources = [s.params.get("from", "judge") for s in self._planned if s.verb == "prune"]
        if not sources or not all(src in ("judge", "summarize") for src in sources):
            return False
        # The index leaves cached junk out only when it embeds summaries
        # (build.run applies the verdicts under embed_summary alone).
        return any(
            s.verb == "index" and s.params.get("embed", "summary") == "summary"
            for s in self._planned
        )

    def stopped_message(self) -> str:
        """What a stopped run keeps, for this recipe. The recipe and model are not
        saved anywhere, so a rerun asks for them again."""
        kept = ""
        if any(step.verb in flow.LLM_STEPS for step in self._planned):
            kept = " Summaries the AI already wrote are kept; the sample check runs again."
        return (
            "[yellow]Stopped.[/] Run the wizard again to carry on; it asks for the recipe "
            f"again.{kept} Press [b]enter[/] to exit."
        )

    def skip_list_path(self) -> str:
        """Where prune writes, as the runner will resolve it."""
        prof = self._state.profile
        # The runner's default when the profile names none (persona/runner.py).
        return prof.blacklist or self._state.profile_path.rsplit(".", 1)[0] + ".blacklist.txt"

    def write_log(self, message: str) -> None:
        self.query_one("#run-log", RichLog).write(message)

    def _finish(self, code: int) -> None:
        self._exit_code = code
        status = self.query_one("#run-status", Static)
        if code == 0:
            collection = escape(self._state.profile.collection)
            lines = [f"\n[green b]Done.[/] Your mail is searchable in [b]{collection}[/]."]
            if self._state.model:
                # `ask` writes its answer with an LLM it finds through RAG_LLM_MODEL,
                # not through anything the wizard saved.
                command = (
                    f"RAG_LLM_MODEL={shlex.quote(self._state.model)} ./mailrag ask "
                    f'"your question" --profile {shlex.quote(self._state.profile_path)}'
                )
                lines.append(f"Ask it a question:  {escape(command)}")
            lines += [
                "Let AI agents search it: connect them to the MCP server (docs/MCP_SERVER.md).",
                "[dim]To keep it up to date as new mail arrives, see docs/SYNC.md.[/]",
            ]
            self.write_log("\n".join(lines))
            status.update(
                "[green]✓ done[/]. Your folder choice is saved. Press [b]enter[/] to exit."
            )
        else:
            status.update(self.stopped_message())

    # -- the worker ---------------------------------------------------------------

    def _execute(self) -> None:
        """Worker-thread entry: run the plan; every UI touch goes through the bridge."""
        try:
            # Inside the try: if it ever raised, the worker would exit without
            # _finish and leave the screen on "running…" for good.
            make_progress_bars_safe()
            code = flow.execute_plan(
                self._state.profile,
                self._state.profile_path,
                self._planned,
                self._handlers,
                self._bridge,
            )
        except Exception:
            self.app.call_from_thread(self.write_log, f"[red]{traceback.format_exc().rstrip()}[/]")
            code = 1
        self.app.call_from_thread(self._finish, code)

    def action_finish(self) -> None:
        if self._exit_code is None:
            self.notify(
                "Still running. Ctrl+Q closes the screen, but the terminal stays busy "
                "until the current step finishes. Summaries already written are kept.",
                severity="warning",
            )
            return
        self.dismiss(self._exit_code)


# --- the app ----------------------------------------------------------------------


class MailragWizardApp(App[int]):
    """The guided onboarding app; screens are driven sequentially by `_drive`."""

    TITLE = "mailrag — guided setup"

    CSS = """
    #stagebar {
        dock: top;
        height: 1;
        padding: 0 2;
        background: $surface;
        color: $text-muted;
    }
    .panel-title {
        color: $text;
        padding: 0 0 1 0;
    }

    /* Welcome */
    #welcome-center { align: center middle; height: 1fr; }
    #welcome-card {
        width: 74;
        height: auto;
        border: round $primary;
        background: $surface;
        padding: 1 3;
    }
    #logo { width: 100%; content-align: center middle; }
    #tagline { width: 100%; content-align: center middle; padding-bottom: 1; }
    #welcome-facts { padding-bottom: 1; }
    #welcome-blurb { padding-bottom: 1; }
    #welcome-keys { width: 100%; content-align: center middle; color: $text-muted; }

    /* Persona */
    #persona-body { height: 1fr; padding: 1 2; }
    #persona-list-panel {
        width: 36;
        border: round $primary;
        background: $surface;
        padding: 1 2;
        margin-right: 2;
    }
    #persona-list { background: $surface; border: none; }
    #persona-list:focus { border: none; }
    #persona-detail-panel {
        width: 1fr;
        border: round $secondary;
        background: $surface;
        padding: 1 2;
    }

    /* Model */
    #model-center { align: center middle; height: 1fr; }
    #model-card {
        width: 74;
        height: auto;
        border: round $primary;
        background: $surface;
        padding: 1 3;
    }
    #model-blurb { padding-bottom: 1; }
    #model-error { padding-top: 1; }

    /* Scope */
    #scope-body { height: 1fr; padding: 1 2; }
    #scope-help { color: $text-muted; padding-bottom: 1; }
    #scope-tree {
        height: 1fr;
        border: round $primary;
        background: $surface;
        padding: 1 2;
    }
    #scope-empty {
        height: 1fr;
        border: round $error;
        background: $surface;
        padding: 1 2;
    }
    #scope-status { height: 1; padding-top: 1; }

    /* Review */
    #review-body { height: 1fr; padding: 1 2; }
    #review-config-panel {
        width: 46%;
        border: round $primary;
        background: $surface;
        padding: 1 2;
        margin-right: 2;
    }
    #review-plan-panel {
        width: 1fr;
        border: round $secondary;
        background: $surface;
        padding: 1 2;
    }

    /* Run */
    #run-body { height: 1fr; padding: 1 2; }
    #run-steps-panel {
        width: 52;
        border: round $primary;
        background: $surface;
        padding: 1 2;
        margin-right: 2;
    }
    .step-row { height: auto; }
    #run-log-panel {
        width: 1fr;
        border: round $secondary;
        background: $surface;
        padding: 1 2;
    }
    #run-log { background: $surface; }
    #run-progress { width: 100%; height: 1; padding: 0 2; }
    #run-progress Bar { width: 1fr; }
    #run-status { height: 1; padding: 0 2; }

    /* Modals */
    ModalScreen { align: center middle; }
    .dialog {
        width: 72;
        max-height: 80%;
        height: auto;
        border: round $accent;
        background: $surface;
        padding: 1 2;
    }
    .dialog-wide { width: 96; }
    .dialog-title { padding-bottom: 1; }
    .dialog-intro { padding-bottom: 1; }
    .dialog-scroll { max-height: 24; height: auto; }
    .dialog-keys { padding-top: 1; color: $text-muted; }
    #rubric-list { background: $surface; border: none; max-height: 12; }
    """

    def __init__(
        self,
        profile_path: str,
        *,
        model: Optional[str] = None,
        limit: Optional[int] = None,
        registry: Optional[Registry] = None,
    ) -> None:
        super().__init__()
        self._personas = registry or load_registry()
        profile = CorpusProfile.load(profile_path)
        self._cli_model = flow.validate_model(model)
        #: Stages the breadcrumb shows; narrowed to the chosen persona's actual
        #: path by _stage_factories (so a skipped Model never reads as visited).
        self.stage_names: List[str] = list(STAGES)
        self._state = WizardState(
            profile_path=profile_path,
            profile=profile,
            model=self._cli_model,
            limit=limit,
            recommendation=flow.read_recommendation(profile_path),
            # A rerun after a stop starts with last time's folders ticked, not
            # an empty tree.
            scope_rules=list(profile.selection_rules) or None,
        )

    @property
    def state(self) -> WizardState:
        """The collected wizard choices (exposed for tests)."""
        return self._state

    def on_mount(self) -> None:
        # Curated default look; a user-configured Textual theme wins. (getattr
        # keeps CI's stub-less mypy from a has-type cycle on the read-then-
        # assign of an attribute it only knows through Any.)
        if getattr(self, "theme", "") == "textual-dark":
            self.theme = "tokyo-night"
        self._drive()

    def _persona(self) -> Optional[Persona]:
        if self._state.persona_name is None:
            return None
        return self._personas.get(self._state.persona_name)

    def _stage_factories(self) -> List[tuple]:
        """The screen sequence, recomputed as choices land (model/scope are dynamic)."""
        st = self._state
        stages: List[tuple] = [
            ("welcome", lambda: WelcomeScreen(st, self._personas)),
            ("persona", lambda: PersonaScreen(st, self._personas)),
        ]
        persona = self._persona()
        if persona is not None:
            if flow.needs_model(persona) and self._cli_model is None:
                stages.append(("model", lambda: ModelScreen(st, persona, self._personas)))
            if any(s.verb == "scope" for s in persona.steps):
                stages.append(("scope", lambda: ScopeScreen(st)))
        stages.append(("review", lambda: ReviewScreen(st, self._personas)))
        # Until a persona is chosen the breadcrumb shows the full ladder; after
        # that, only the stages this persona actually walks.
        self.stage_names = (
            list(STAGES) if persona is None else [name.capitalize() for name, _ in stages] + ["Run"]
        )
        return stages

    @work(exclusive=True)
    async def _drive(self) -> None:
        """Walk the guided screens (with back-navigation), then hand off to the run."""
        st = self._state
        index = 0
        while True:
            stages = self._stage_factories()
            if index >= len(stages):
                break
            name, make_screen = stages[index]
            result = await self.push_screen_wait(make_screen())
            if result is BLOCKED:
                self.exit(2)
                return
            if result is QUIT or (name == "welcome" and result is not True):
                self.exit(1)
                return
            if result is BACK:
                index = max(0, index - 1)
                continue
            if name == "persona":
                st.persona_name = str(result)
            elif name == "model":
                st.model = str(result)
            elif name == "scope":
                st.scope_rules = list(result)
            index += 1
        code = await self.push_screen_wait(RunScreen(st, self._personas))
        self.exit(code)


def run_tui(
    profile_path: str,
    *,
    model: Optional[str] = None,
    limit: Optional[int] = None,
    registry: Optional[Registry] = None,
) -> int:
    """Launch the full-screen wizard; returns a process exit code.

    Needs a real terminal — the CLI falls back with a hint otherwise (headless
    runs use ``mailrag run --persona <name>``, scripted ones ``--classic``)."""
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        print(
            "error: the wizard needs an interactive terminal. For non-interactive "
            "runs use `mailrag run --profile ... --persona <name>` (or "
            "`mailrag wizard --classic` for the line-prompt flow).",
            file=sys.stderr,
        )
        return 2
    try:
        app = MailragWizardApp(profile_path, model=model, limit=limit, registry=registry)
    # OSError: unreadable file; ValueError: bad JSON; AttributeError: valid
    # JSON that isn't an object (CorpusProfile.load calls .items() on it).
    except (OSError, ValueError, AttributeError) as exc:
        print(f"error: cannot load profile {profile_path!r}: {exc}", file=sys.stderr)
        return 2
    result = app.run()
    return result if isinstance(result, int) else 1
