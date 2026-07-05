"""YAML rubric registry for the Pass-2 summarize+judge prompt.

Rubrics are full prompt templates (placeholders ``{sender}/{date}/{subject}/{body}``
plus the strict-JSON output contract that :func:`src.llm.summary.parse_response`
parses). They resolve two-tier: a gitignored ``rubrics/local/`` override dir is
searched first, then the shipped ``rubrics/`` dir — so corpus-specific rubrics
stay local while the public repo still ships ``work`` and a ``personal.example``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

_BODY_CHARS = 4000
_REQUIRED_PLACEHOLDERS: Tuple[str, ...] = ("{sender}", "{date}", "{subject}", "{body}")

# Repo root is three levels up from this file (src/llm/rubrics.py).
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
# Search order: local override (gitignored) first, then the shipped dir.
_SEARCH_DIRS: Tuple[Path, ...] = (_REPO_ROOT / "rubrics" / "local", _REPO_ROOT / "rubrics")


@dataclass(frozen=True)
class Rubric:
    name: str
    template: str
    version: int = 1
    judge_template: str = ""  # optional verdict-only prompt for the `judge` verb


def _find(name: str) -> Path:
    for d in _SEARCH_DIRS:
        p = d / f"{name}.yaml"
        if p.exists():
            return p
    searched = ", ".join(str(d) for d in _SEARCH_DIRS)
    raise ValueError(
        f"no rubric named {name!r} (looked in: {searched}); "
        f"create rubrics/local/{name}.yaml "
        f"(e.g. copy rubrics/{name}.example.yaml if one ships)"
    )


def _validate(name: str, template: str) -> None:
    missing = [p for p in _REQUIRED_PLACEHOLDERS if p not in template]
    if missing:
        raise ValueError(f"rubric {name!r} template missing placeholders: {', '.join(missing)}")


def load_rubric(name: str) -> Rubric:
    """Load, validate, and return the named rubric (local override wins)."""
    path = _find(name)
    data: Dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    template = data.get("template")
    if not isinstance(template, str) or not template.strip():
        raise ValueError(f"rubric {name!r} ({path}) has no 'template' string")
    _validate(name, template)
    judge_template = data.get("judge_template") or ""
    return Rubric(
        name=str(data.get("name", name)),
        template=template,
        version=int(data.get("version", 1)),
        judge_template=str(judge_template),
    )


def build_prompt(name: str, email: Dict[str, Any], body_chars: int = _BODY_CHARS) -> str:
    """Format the named rubric for *email* (body truncated to *body_chars*).

    Matches :func:`src.llm.summary.build_prompt` field handling exactly so the
    ``work`` rubric is byte-identical to the legacy path.
    """
    template = load_rubric(name).template
    body = (email.get("body") or "").strip()
    if len(body) > body_chars:
        body = body[:body_chars] + " […truncated]"
    return template.format(
        sender=(email.get("sender") or "")[:200],
        date=email.get("date") or "unknown",
        subject=(email.get("subject") or "")[:300],
        body=body,
    )


def build_judge_prompt(name: str, email: Dict[str, Any], body_chars: int = _BODY_CHARS) -> str:
    """Format the rubric's verdict-only `judge_template` for *email*.

    The judge prompt asks only for ``is_noise``/``confidence``/``reason`` (no
    summary), so its output stays short and cheap. If the rubric has no
    ``judge_template``, fall back to the full summarize template (correct verdicts,
    but no token saving) and warn once."""
    rubric = load_rubric(name)
    template = rubric.judge_template
    if not template:
        import warnings

        warnings.warn(
            f"rubric {name!r} has no judge_template; judge will use the full "
            f"summarize template (no token saving). Add a judge_template to save cost.",
            stacklevel=2,
        )
        template = rubric.template
    body = (email.get("body") or "").strip()
    if len(body) > body_chars:
        body = body[:body_chars] + " […truncated]"
    return template.format(
        sender=(email.get("sender") or "")[:200],
        date=email.get("date") or "unknown",
        subject=(email.get("subject") or "")[:300],
        body=body,
    )


def names() -> List[str]:
    """Usable rubric names across the search path (``*.example.yaml`` excluded)."""
    seen: Dict[str, Path] = {}
    for d in _SEARCH_DIRS:
        if d.exists():
            for p in sorted(d.glob("*.yaml")):
                if p.name.endswith(".example.yaml"):
                    continue
                seen.setdefault(p.stem, p)
    return sorted(seen)
