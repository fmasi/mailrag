# src/profile.py
"""Single source of truth threaded through every pipeline stage."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from typing import Optional


@dataclass
class CorpusProfile:
    root: str = ""
    selection_rules: list = field(default_factory=list)
    chunk_size: int = 512
    chunk_overlap: int = 64
    rubric: str = "personal"  # stored now; consumed in increment 1b
    collection: str = "email-rag"
    qdrant_url: str = "http://localhost:6333"
    pass2_cache: Optional[str] = None
    blacklist: Optional[str] = None  # content-addressed drop set (written by prune)
    calibration: Optional[dict] = None  # written by 1b's calibrate gate
    updated_at: Optional[str] = None

    @classmethod
    def load(cls, path: str) -> "CorpusProfile":
        with open(os.path.expanduser(path), encoding="utf-8") as fh:
            data = json.load(fh)
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    def save(self, path: str) -> None:
        """Write the profile, stamping ``updated_at``.

        The field existed but nothing ever set it, so every profile on disk
        reported ``None`` — a provenance field carrying no provenance. Selection
        rules are a point-in-time snapshot of an interactive choice, and mail
        arrives continuously, so "when was this last decided" is exactly the
        question a stale profile needs to answer.
        """
        from datetime import datetime, timezone

        self.updated_at = datetime.now(timezone.utc).isoformat()
        with open(os.path.expanduser(path), "w", encoding="utf-8") as fh:
            json.dump(asdict(self), fh, indent=2)

    def resolved_root(self) -> str:
        return os.path.abspath(os.path.expanduser(self.root))
