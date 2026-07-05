"""Execute a persona recipe by dispatching each step to a verb handler.

Pure orchestration: the caller supplies ``handlers`` (``verb -> callable``) and
this walks the persona's steps in order, calling ``handler(profile, **params)``.
Both the headless ``run`` verb and the interactive TUI build a handler map and
hand it here, so the run logic lives in one place. See docs/VERBS.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from src.persona.registry import Persona, Step

Handler = Callable[..., Any]


@dataclass
class StepResult:
    verb: str
    params: Dict[str, Any]
    result: Any


def _is_optional(step: Step) -> bool:
    return bool(step.params.get("optional"))


def missing_handlers(persona: Persona, handlers: Dict[str, Handler]) -> List[str]:
    """Required (non-optional) verbs in *persona* that have no handler, in order."""
    seen: List[str] = []
    for s in persona.steps:
        if _is_optional(s):
            continue
        if s.verb not in handlers and s.verb not in seen:
            seen.append(s.verb)
    return seen


def run_persona(
    profile: Any,
    persona: Persona,
    handlers: Dict[str, Handler],
    *,
    on_step: Optional[Callable[[Step], None]] = None,
    on_skip: Optional[Callable[[Step], None]] = None,
) -> List[StepResult]:
    """Run *persona*'s steps in order against *profile*.

    Raises ``ValueError`` if a required step has no handler (check first with
    :func:`missing_handlers` to fail friendly). Optional steps whose handler is
    absent are skipped (``on_skip`` notified)."""
    missing = missing_handlers(persona, handlers)
    if missing:
        raise ValueError(
            f"persona {persona.name!r} needs unimplemented verb(s): {', '.join(missing)}"
        )
    results: List[StepResult] = []
    for step in persona.steps:
        if step.verb not in handlers:
            if _is_optional(step):
                if on_skip:
                    on_skip(step)
                continue
            raise ValueError(f"no handler for required verb {step.verb!r}")
        if on_step:
            on_step(step)
        # Drop the orchestration-only `optional` flag before calling the handler.
        params = {k: v for k, v in step.params.items() if k != "optional"}
        res = handlers[step.verb](profile, **params)
        results.append(StepResult(verb=step.verb, params=params, result=res))
    return results
