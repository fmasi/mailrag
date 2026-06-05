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

Filtering is conservative (Pass-1): besides the curated ``categories`` rules it
supports a header-driven ``bulk_filter`` that drops List-Unsubscribe/Precedence
bulk mail unless a keep-guard spares it (freemail sender, whitelisted domain, or
transactional subject), so human mailing-list traffic and receipts survive.
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


@dataclass
class _BulkRule:
    """Header-driven bulk-mail rule with keep-guards (conservative Pass-1).

    An email flagged ``is_bulk`` (List-Unsubscribe / Precedence:bulk) is noise
    *unless* a guard keeps it: the sender domain is in ``keep_domains`` (freemail
    senders + explicit whitelist, merged) or the subject matches a transactional
    keep-pattern. This biases toward keeping data — human mailing-list traffic
    and receipts survive; recall is left to later passes.
    """

    name: str = "bulk_unsubscribe"
    keep_domains: list[str] = field(default_factory=list)
    keep_sender_patterns: list[re.Pattern] = field(default_factory=list)
    keep_subject_patterns: list[re.Pattern] = field(default_factory=list)


class NoiseFilter:
    """Rule-based email noise classifier.

    Rules are loaded from a YAML file.  Each category defines one or more
    matching conditions; an email is considered noise if *any* condition in
    *any* category matches.

    Matching logic:
        sender_domains  — plain substring match in sender (case-insensitive)
        sender_patterns — regex match against full sender string
        subject_patterns — regex match against subject string

    In addition to the per-category rules, an optional ``bulk_filter`` section
    drops emails flagged ``is_bulk`` (List-Unsubscribe / Precedence:bulk on the
    raw headers) — unless a keep-guard spares them (freemail sender, whitelisted
    domain, or transactional subject). The bulk rule is evaluated last, so an
    explicit category match always wins and names the reason; bulk matches are
    reported under the category ``bulk_unsubscribe``.
    """

    def __init__(
        self, rules: list[_CategoryRule], bulk: "_BulkRule | None" = None
    ) -> None:
        self._rules = rules
        self._bulk = bulk

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

        bulk = None
        bcfg = data.get("bulk_filter")
        if isinstance(bcfg, dict):
            keep_domains = [
                d.lower() for d in (bcfg.get("keep_freemail_domains") or [])
            ] + [d.lower() for d in (bcfg.get("keep_domains") or [])]
            bulk = _BulkRule(
                name=bcfg.get("category_name", "bulk_unsubscribe"),
                keep_domains=keep_domains,
                keep_sender_patterns=[
                    re.compile(p, re.IGNORECASE)
                    for p in (bcfg.get("keep_sender_patterns") or [])
                ],
                keep_subject_patterns=[
                    re.compile(p, re.IGNORECASE)
                    for p in (bcfg.get("keep_subject_patterns") or [])
                ],
            )
        return cls(rules, bulk=bulk)

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
        matched, _ = self._evaluate(
            email.sender, email.subject, getattr(email, "is_bulk", False)
        )
        return matched

    def matched_category(self, email: "NormalizedEmail") -> str | None:
        """Return the first matching category name, or None."""
        _, category = self._evaluate(
            email.sender, email.subject, getattr(email, "is_bulk", False)
        )
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
            bool(payload.get("is_bulk", False)),
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

    @staticmethod
    def _domains_match(sender_lower: str, domains: list[str]) -> bool:
        # Match against the address part only, anchored so that "linkedin.com"
        # does not match "notlinkedin.com" or a display name that happens to
        # contain the domain string.
        # Accepts:  user@linkedin.com   user@sub.linkedin.com
        # Rejects:  user@notlinkedin.com  "linkedin.com info" <other@host>
        return any(
            f"@{domain}" in sender_lower or f".{domain}" in sender_lower
            for domain in domains
        )

    def _bulk_kept(self, sender_lower: str, subject: str) -> bool:
        """Return True if a bulk-flagged email is spared by a keep-guard."""
        b = self._bulk
        if self._domains_match(sender_lower, b.keep_domains):
            return True
        if any(p.search(sender_lower) for p in b.keep_sender_patterns):
            return True
        return any(p.search(subject) for p in b.keep_subject_patterns)

    def _evaluate(
        self, sender: str, subject: str, is_bulk: bool = False
    ) -> tuple[bool, str | None]:
        sender_lower = sender.lower()
        for rule in self._rules:
            if self._domains_match(sender_lower, rule.sender_domains):
                return True, rule.name
            # sender_patterns — regex
            if any(p.search(sender) for p in rule.sender_patterns):
                return True, rule.name
            # subject_patterns — regex
            if any(p.search(subject) for p in rule.subject_patterns):
                return True, rule.name
        # Bulk-header rule is evaluated last so an explicit category match always
        # wins and names the reason. Conservative: kept unless no guard spares it.
        if (
            is_bulk
            and self._bulk is not None
            and not self._bulk_kept(sender_lower, subject)
        ):
            return True, self._bulk.name
        return False, None
