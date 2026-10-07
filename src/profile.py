# src/profile.py
"""Single source of truth threaded through every pipeline stage."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, fields
from typing import Optional


class ProfileChangedError(RuntimeError):
    """The profile file was rewritten by someone else since it was loaded."""


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

    # mtime of the file this instance was loaded from, so save() can tell that
    # someone else rewrote it meanwhile. Not a profile field: excluded from the
    # serialised form because save() skips underscore-prefixed fields.
    _loaded_mtime: Optional[float] = field(default=None, repr=False, compare=False)

    @classmethod
    def load(cls, path: str) -> "CorpusProfile":
        full = os.path.expanduser(path)
        with open(full, encoding="utf-8") as fh:
            data = json.load(fh)
        known = {f.name for f in fields(cls)}
        prof = cls(**{k: v for k, v in data.items() if k in known})
        try:
            prof._loaded_mtime = os.path.getmtime(full)
        except OSError:
            prof._loaded_mtime = None
        return prof

    def save(self, path: str, *, force: bool = False) -> None:
        """Write the profile, stamping ``updated_at``.

        Refuses when the file changed on disk since this instance loaded it,
        unless ``force``. A command holds a profile in memory for as long as it
        runs — an index build, minutes to hours — and writing it back at the end
        silently reverts whatever another command recorded in between. A
        calibrate that finished mid-build lost its result exactly that way, and
        the loss was invisible: the file simply held older numbers. Detecting it
        is cheap; recovering a result nobody noticed was discarded is not.

        The field existed but nothing ever set it, so every profile on disk
        reported ``None`` — a provenance field carrying no provenance. Selection
        rules are a point-in-time snapshot of an interactive choice, and mail
        arrives continuously, so "when was this last decided" is exactly the
        question a stale profile needs to answer.
        """
        from datetime import datetime, timezone

        full = os.path.expanduser(path)
        if not force and self._loaded_mtime is not None:
            try:
                current = os.path.getmtime(full)
            except OSError:
                current = None
            if current is not None and current > self._loaded_mtime:
                raise ProfileChangedError(
                    f"{full} changed on disk since it was loaded; refusing to overwrite "
                    "it with a stale copy. Another command (calibrate, scope, prune) "
                    "probably wrote it while this one was running. Re-run this command, "
                    "or pass force=True if the in-memory copy is the one you want."
                )

        self.updated_at = datetime.now(timezone.utc).isoformat()
        data = {f.name: getattr(self, f.name) for f in fields(self) if not f.name.startswith("_")}
        with open(full, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
        try:
            self._loaded_mtime = os.path.getmtime(full)
        except OSError:
            pass

    def resolved_root(self) -> str:
        return os.path.abspath(os.path.expanduser(self.root))
