# src/profile.py
"""Single source of truth threaded through every pipeline stage."""

from __future__ import annotations

import errno
import json
import os
import shutil
import uuid
from dataclasses import dataclass, field, fields
from typing import Optional


class ProfileChangedError(RuntimeError):
    """Another command changed the same profile field this one is trying to save.

    ``fields`` names what was contested, for a caller that wants to say so
    without the file path the message carries."""

    def __init__(self, message: str, fields: Optional[list] = None):
        super().__init__(message)
        self.fields = list(fields or [])


# Stamped on every save, so it always differs and says nothing about who
# changed what.
_BOOKKEEPING = ("updated_at",)


def _frozen(value) -> str:
    """A field's value as comparable text. Taken as a copy on purpose: commands
    change lists and dicts in place, and a snapshot sharing them never differs."""
    return json.dumps(value, sort_keys=True, default=str)


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

    # What the file held when this instance read it, and which file that was, so
    # save() can tell this command's changes from someone else's. Bookkeeping,
    # not profile data: underscore-prefixed fields are never read from or
    # written to the file.
    _loaded: Optional[dict] = field(default=None, repr=False, compare=False)
    _loaded_path: Optional[str] = field(default=None, repr=False, compare=False)

    @classmethod
    def _data_fields(cls) -> list:
        return [f.name for f in fields(cls) if not f.name.startswith("_")]

    def _snapshot(self) -> dict:
        return {name: _frozen(getattr(self, name)) for name in self._data_fields()}

    @classmethod
    def load(cls, path: str) -> "CorpusProfile":
        full = os.path.expanduser(path)
        with open(full, encoding="utf-8") as fh:
            data = json.load(fh)
        known = set(cls._data_fields())
        prof = cls(**{k: v for k, v in data.items() if k in known})
        prof._loaded = prof._snapshot()
        prof._loaded_path = os.path.realpath(full)
        return prof

    def _merge_from_disk(self, full: str) -> dict:
        """Take over the fields someone else changed since this instance loaded.

        Three values per field: what was loaded, what this instance holds now,
        what the file holds now. Only the file moved: keep the file's. Only this
        instance moved: keep this one's. Both moved, to different values: that
        is a real conflict and nothing here can pick a winner. Those fields are
        returned as ``{name: the file's value}`` and left untouched in memory.
        """
        try:
            theirs = CorpusProfile.load(full)
        except (OSError, ValueError, TypeError, AttributeError):
            # Deleted since, or no longer a readable profile (truncated, not a
            # JSON object): there is no result in it to protect. Write it again.
            return {}
        loaded = self._loaded or {}
        on_disk = theirs._snapshot()
        conflicts = {}
        for name in self._data_fields():
            if name in _BOOKKEEPING:
                continue
            base = loaded.get(name)
            mine = _frozen(getattr(self, name))
            other = on_disk[name]
            if other == base or other == mine:
                continue
            if mine == base:
                setattr(self, name, getattr(theirs, name))
            else:
                conflicts[name] = getattr(theirs, name)
        return conflicts

    def save(self, path: str, *, force: bool = False) -> None:
        """Write the profile, stamping ``updated_at``.

        A command holds a profile in memory for as long as it runs, which for
        an index build is minutes to hours. Writing that copy back whole reverts
        whatever another command recorded in between: a calibrate that finished
        mid-build lost its result exactly that way, and the loss was invisible,
        because the file simply held older numbers.

        So a save to the file this instance was loaded from writes only the
        fields this instance changed, and keeps the rest as the file has them
        now. Two commands that changed different fields both keep their result.
        When both changed the SAME field, the other command's value stays in the
        file, every other field is still written, and :class:`ProfileChangedError`
        names the field that was not. Saving again does not slip it through.
        ``force`` writes this copy as it stands. A profile that was never loaded,
        or one saved under another name, has nothing to merge with and is
        written whole.

        The change is detected from the file's content, not its mtime: a rewrite
        inside one clock tick, or a file restored with an older stamp, is still
        a change. What is left is the instant between re-reading the file and
        renaming the new one over it. Two saves landing inside it can still lose
        one field, against the whole length of a run before.

        ``updated_at`` existed but nothing ever set it, so every profile on disk
        reported ``None`` — a provenance field carrying no provenance. Selection
        rules are a point-in-time snapshot of an interactive choice, and mail
        arrives continuously, so "when was this last decided" is exactly the
        question a stale profile needs to answer.
        """
        from datetime import datetime, timezone

        # The real file, so a symlinked profile stays a link: renaming over the
        # link would replace it with a plain file and leave its target stale.
        full = os.path.realpath(os.path.expanduser(path))
        exists = os.path.exists(full)
        if exists and not os.access(full, os.W_OK):
            # The rename below only needs a writable directory, so it would go
            # straight through a file its owner made read-only.
            raise PermissionError(errno.EACCES, "profile is read-only", full)

        conflicts: dict = {}
        if not force and self._loaded is not None and self._loaded_path == full:
            conflicts = self._merge_from_disk(full)

        self.updated_at = datetime.now(timezone.utc).isoformat()
        data = {name: getattr(self, name) for name in self._data_fields()}
        data.update(conflicts)  # a contested field keeps the other command's value
        # Temp file + rename: a reader (another command's merge, the long-lived
        # MCP server) sees the old profile or the new one, never a truncated one.
        tmp = f"{full}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2)
            if exists:
                shutil.copymode(full, tmp)
            os.replace(tmp, full)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

        # The new base is what the file now holds, EXCEPT for a contested field:
        # its base stays where it was, so saving again raises again instead of
        # reading the other command's value as "unchanged" and overwriting it.
        base = self._loaded or {}
        self._loaded = {
            name: (base.get(name) if name in conflicts else _frozen(data[name]))
            for name in self._data_fields()
        }
        self._loaded_path = full
        if conflicts:
            raise ProfileChangedError(
                f"{full}: another command changed {', '.join(sorted(conflicts))} while this "
                "one was running, and this one changed it too. The other command's value "
                "was kept for that field and everything else was saved. Re-run this "
                "command to redo that part on top of it.",
                fields=sorted(conflicts),
            )

    def resolved_root(self) -> str:
        return os.path.abspath(os.path.expanduser(self.root))
