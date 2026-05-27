"""Email noise filter — rule-based pre-index and post-index filtering.

Loads rules from ``config/noise_rules.yaml`` and classifies
``NormalizedEmail`` objects as noise or not without any LLM or embedding call.

Typical usage
-------------
Pre-index (skip noisy emails before embedding):

    from src.data.noise_filter import NoiseFilter

    noise_filter = NoiseFilter.from_project_rules()
    clean = [e for e in emails if not noise_filter.is_noise(e)]

Post-index (scan Qdrant payloads):

    matched, category = noise_filter.match_payload({"sender": "...", "subject": "..."})
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.data.models import NormalizedEmail

# Default rules file relative to the project root (two levels up from this file)
_DEFAULT_RULES_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "noise_rules.yaml"


@dataclass
class _CategoryRule:
    name: str
    description: str
    sender_domains: list[str] = field(default_factory=list)
    sender_patterns: list[re.Pattern] = field(default_factory=list)
    subject_patterns: list[re.Pattern] = field(default_factory=list)


class NoiseFilter:
    """Rule-based email noise classifier.

    Rules are loaded from a YAML file.  Each category defines one or more
    matching conditions; an email is considered noise if *any* condition in
    *any* category matches.

    Matching logic:
        sender_domains  — plain substring match in sender (case-insensitive)
        sender_patterns — regex match against full sender string
        subject_patterns — regex match against subject string
    """

    def __init__(self, rules: list[_CategoryRule]) -> None:
        self._rules = rules

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_file(cls, path: str | Path) -> "NoiseFilter":
        """Load rules from a YAML file."""
        try:
            import yaml
        except ImportError as exc:
            raise ImportError(
                "PyYAML is required for NoiseFilter. Install it with: pip install pyyaml"
            ) from exc

        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        rules: list[_CategoryRule] = []
        for category_name, cfg in (data.get("categories") or {}).items():
            if not isinstance(cfg, dict):
                continue
            rules.append(
                _CategoryRule(
                    name=category_name,
                    description=cfg.get("description", ""),
                    sender_domains=[
                        d.lower() for d in (cfg.get("sender_domains") or [])
                    ],
                    sender_patterns=[
                        re.compile(p, re.IGNORECASE)
                        for p in (cfg.get("sender_patterns") or [])
                    ],
                    subject_patterns=[
                        re.compile(p, re.IGNORECASE)
                        for p in (cfg.get("subject_patterns") or [])
                    ],
                )
            )
        return cls(rules)

    @classmethod
    def from_project_rules(cls) -> "NoiseFilter":
        """Load the default project-level noise_rules.yaml."""
        if not _DEFAULT_RULES_PATH.exists():
            # No rules file → nothing is filtered (safe default)
            return cls([])
        return cls.from_file(_DEFAULT_RULES_PATH)

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    def is_noise(self, email: "NormalizedEmail") -> bool:
        """Return True if the email matches any noise rule."""
        matched, _ = self._evaluate(email.sender, email.subject)
        return matched

    def matched_category(self, email: "NormalizedEmail") -> str | None:
        """Return the first matching category name, or None."""
        _, category = self._evaluate(email.sender, email.subject)
        return category

    def match_payload(self, payload: dict) -> tuple[bool, str | None]:
        """Classify a raw Qdrant payload dict.

        Returns (is_noise, category_name_or_None).
        Useful for post-index scans that work with raw payloads rather than
        NormalizedEmail objects.
        """
        return self._evaluate(
            payload.get("sender", ""),
            payload.get("subject", ""),
        )

    def category_names(self) -> list[str]:
        """Return the list of configured category names."""
        return [r.name for r in self._rules]

    def is_empty(self) -> bool:
        """Return True if no rules are loaded (nothing will be filtered)."""
        return len(self._rules) == 0

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _evaluate(self, sender: str, subject: str) -> tuple[bool, str | None]:
        sender_lower = sender.lower()
        for rule in self._rules:
            # sender_domains — match against the address part only, anchored so
            # that "linkedin.com" does not match "notlinkedin.com" or a display
            # name that happens to contain the domain string.
            # Accepts:  user@linkedin.com   user@sub.linkedin.com
            # Rejects:  user@notlinkedin.com  "linkedin.com info" <other@host>
            if any(
                f"@{domain}" in sender_lower or f".{domain}" in sender_lower
                for domain in rule.sender_domains
            ):
                return True, rule.name
            # sender_patterns — regex
            if any(p.search(sender) for p in rule.sender_patterns):
                return True, rule.name
            # subject_patterns — regex
            if any(p.search(subject) for p in rule.subject_patterns):
                return True, rule.name
        return False, None
