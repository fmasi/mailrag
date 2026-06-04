# src/profile.py
"""Single source of truth threaded through every pipeline stage."""
from __future__ import annotations
import json, os
from dataclasses import dataclass, field, fields, asdict
from typing import Optional


@dataclass
class CorpusProfile:
    root: str = ""
    selection_rules: list = field(default_factory=list)
    chunk_size: int = 512
    chunk_overlap: int = 64
    rubric: str = "personal"          # stored now; consumed in increment 1b
    collection: str = "email-rag"
    qdrant_url: str = "http://localhost:6333"
    pass2_cache: Optional[str] = None
    calibration: Optional[dict] = None  # written by 1b's calibrate gate
    updated_at: Optional[str] = None

    @classmethod
    def load(cls, path: str) -> "CorpusProfile":
        with open(os.path.expanduser(path), encoding="utf-8") as fh:
            data = json.load(fh)
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    def save(self, path: str) -> None:
        with open(os.path.expanduser(path), "w", encoding="utf-8") as fh:
            json.dump(asdict(self), fh, indent=2)

    def resolved_root(self) -> str:
        return os.path.abspath(os.path.expanduser(self.root))
