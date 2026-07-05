"""Persona registry: load the named verb recipes from ``personas.yaml``.

A persona is a named, ordered recipe of verbs (with optional per-step settings).
Resolves two-tier like the rubric registry: a gitignored ``personas.local.yaml``
merges over the shipped ``personas.yaml`` (local personas win). See docs/VERBS.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# Repo root is two levels up from this file (src/persona/registry.py).
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_SHIPPED = _REPO_ROOT / "personas.yaml"
_LOCAL = _REPO_ROOT / "personas.local.yaml"


@dataclass(frozen=True)
class VerbInfo:
    name: str
    does: str
    cost: str


@dataclass(frozen=True)
class Step:
    verb: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Persona:
    name: str
    label: str
    advisor_hint: str
    steps: List[Step]


@dataclass(frozen=True)
class Registry:
    verbs: Dict[str, VerbInfo]
    personas: Dict[str, Persona]

    def names(self) -> List[str]:
        return list(self.personas)

    def get(self, name: str) -> Persona:
        if name not in self.personas:
            raise ValueError(f"no persona named {name!r}; known: {', '.join(self.names())}")
        return self.personas[name]

    def verb_info(self, name: str) -> VerbInfo:
        if name not in self.verbs:
            raise ValueError(f"no verb metadata for {name!r}")
        return self.verbs[name]


def _parse_step(raw: Any) -> Step:
    """A step is either ``"verb"`` or ``{"verb": {params}}``."""
    if isinstance(raw, str):
        return Step(verb=raw, params={})
    if isinstance(raw, dict) and len(raw) == 1:
        verb, params = next(iter(raw.items()))
        return Step(verb=verb, params=dict(params or {}))
    raise ValueError(f"malformed step: {raw!r}")


def _parse(data: Dict[str, Any]) -> Registry:
    verbs = {
        name: VerbInfo(name=name, does=str(meta.get("does", "")), cost=str(meta.get("cost", "")))
        for name, meta in (data.get("verbs") or {}).items()
    }
    personas: Dict[str, Persona] = {}
    for name, meta in (data.get("personas") or {}).items():
        steps = [_parse_step(s) for s in (meta.get("steps") or [])]
        for s in steps:
            if s.verb not in verbs:
                raise ValueError(
                    f"persona {name!r} uses unknown verb {s.verb!r} "
                    f"(declare it under `verbs:` in personas.yaml)"
                )
        personas[name] = Persona(
            name=name,
            label=str(meta.get("label", name)),
            advisor_hint=str(meta.get("advisor_hint", "")),
            steps=steps,
        )
    return Registry(verbs=verbs, personas=personas)


def load_registry(path: Optional[str] = None, *, text: Optional[str] = None) -> Registry:
    """Load the persona registry.

    ``text`` (inline YAML) takes precedence for tests. Otherwise reads ``path``
    (or the shipped ``personas.yaml``), then merges a ``personas.local.yaml`` if
    present (local personas/verbs override shipped ones)."""
    if text is not None:
        return _parse(yaml.safe_load(text) or {})
    base_path = Path(path) if path else _SHIPPED
    data: Dict[str, Any] = yaml.safe_load(base_path.read_text(encoding="utf-8")) or {}
    if path is None and _LOCAL.exists():
        local = yaml.safe_load(_LOCAL.read_text(encoding="utf-8")) or {}
        data.setdefault("verbs", {}).update(local.get("verbs") or {})
        data.setdefault("personas", {}).update(local.get("personas") or {})
    return _parse(data)
