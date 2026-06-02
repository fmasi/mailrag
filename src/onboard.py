"""ZTH onboarding: build a validated thread-aware contextual assistant from an
.eml directory in one bounded LLM pass. See
docs/superpowers/specs/2026-06-02-zth-onboard-design.md.
"""
import re
from pathlib import Path


def collection_slug(source_dir):
    """Default collection name for a source directory: ``mailrag-<slug>``."""
    name = Path(source_dir).resolve().name or "corpus"
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "corpus"
    return f"mailrag-{slug}"


def load_eml_dir(source_dir, *, limit=None):
    """Load every ``.eml`` under ``source_dir`` (recursively) into NormalizedEmail
    objects. Raises ValueError when the directory is missing or has no .eml files."""
    root = Path(source_dir)
    if not root.is_dir():
        raise ValueError(f"not a directory: {source_dir}")
    paths = sorted(str(p) for p in root.rglob("*.eml"))
    if limit:
        paths = paths[:limit]
    if not paths:
        raise ValueError(f"no .eml files found under {source_dir}")
    from src.data.loaders.mail_archive_x import MailArchiveXLoader
    return MailArchiveXLoader(eml_files=paths).load()


def filter_kept(emails, judgments, *, min_confidence=0.7):
    """Drop emails judged noise with confidence >= ``min_confidence``; set
    ``.summary`` on the kept ones from their judgment. Emails with no judgment are
    kept (conservative). Returns ``(kept_emails, n_noise_dropped)``."""
    kept, dropped = [], 0
    for e in emails:
        mid = getattr(e, "message_id", "") or ""
        rec = judgments.get(mid)
        if rec and rec["is_noise"] and rec["confidence"] >= min_confidence:
            dropped += 1
            continue
        if rec and rec.get("summary"):
            try:
                e.summary = rec["summary"]
            except (AttributeError, TypeError):
                pass
        kept.append(e)
    return kept, dropped
