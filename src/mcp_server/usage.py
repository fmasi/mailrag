"""Append-only usage log for the MCP tools — which tools get used, and how.

Every tool call writes one JSON line: what was called, with what argument shape,
how long it took, and whether it failed. The point is to answer questions the
code cannot: which tools agents actually reach for, which ones are never called
(a dead tool is usually a badly *described* tool, not an unwanted one), which
arguments they bother to pass, and which calls are slow enough to be abandoned.

The log lives outside the repo (``~/.mailrag/mcp_usage.jsonl`` by default) because
it records the user's own queries against their own mail.

Configuration:

* ``MAILRAG_MCP_USAGE_LOG`` — path to write to. Set it to ``""``, ``0``, ``off``
  or ``none`` to disable logging entirely.
* ``MAILRAG_MCP_USAGE_ARGS`` — ``values`` (default) logs argument values,
  truncated; ``names`` logs only names and types, for when even a search query is
  too sensitive to keep on disk.

**Logging must never break a tool call.** Every failure path here is swallowed:
a broken log is worth strictly less than a working search.
"""

from __future__ import annotations

import functools
import json
import os
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional

DEFAULT_USAGE_LOG = "~/.mailrag/mcp_usage.jsonl"

# Argument values are truncated rather than dropped: enough to see that a query
# was a natural-language question vs. an id lookup, not enough to mirror the
# corpus into the log.
_MAX_VALUE_CHARS = 200
_DISABLED = {"", "0", "off", "none", "false", "no"}


def resolve_usage_log() -> Optional[str]:
    """Resolve the usage-log path, or ``None`` when logging is disabled.

    Precedence: ``$MAILRAG_MCP_USAGE_LOG`` > ``~/.mailrag/mcp_usage.jsonl``. An
    explicit empty/``0``/``off``/``none`` value disables logging; the variable
    being *unset* does not (logging is on by default, locally).
    """
    raw = os.environ.get("MAILRAG_MCP_USAGE_LOG")
    if raw is not None and raw.strip().lower() in _DISABLED:
        return None
    return os.path.expanduser(raw or DEFAULT_USAGE_LOG)


def _redact_values() -> bool:
    """True when only argument names/types should be logged, not their values."""
    return os.environ.get("MAILRAG_MCP_USAGE_ARGS", "values").strip().lower() == "names"


def _summarize_value(value: Any) -> Any:
    """Reduce one argument to something small and JSON-safe."""
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value if len(value) <= _MAX_VALUE_CHARS else value[:_MAX_VALUE_CHARS] + "…"
    return f"<{type(value).__name__}>"


def summarize_args(args: Dict[str, Any]) -> Dict[str, Any]:
    """Summarize a call's kwargs for the log.

    Drops arguments left at ``None`` so the log shows what the caller *chose* to
    pass — that is the signal for whether a parameter earns its place in the
    tool schema. Values are truncated, or replaced by their type name when
    ``MAILRAG_MCP_USAGE_ARGS=names``.
    """
    names_only = _redact_values()
    out: Dict[str, Any] = {}
    for key, value in args.items():
        if value is None:
            continue
        out[key] = f"<{type(value).__name__}>" if names_only else _summarize_value(value)
    return out


def _result_size(result: Any) -> Optional[int]:
    """Number of rows a result carries, when that is a meaningful notion.

    Handles the two shapes the tools return: a bare list of rows, and a dict
    wrapping its rows under ``matches``/``sources`` (grep_email, answer_question).
    """
    if isinstance(result, list):
        return len(result)
    if isinstance(result, dict):
        for key in ("matches", "sources"):
            inner = result.get(key)
            if isinstance(inner, list):
                return len(inner)
    return None


def record(
    tool: str,
    args: Dict[str, Any],
    *,
    duration_s: float,
    ok: bool,
    result: Any = None,
    error: Optional[BaseException] = None,
) -> None:
    """Append one call record to the usage log. Never raises."""
    try:
        path = resolve_usage_log()
        if path is None:
            return
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "tool": tool,
            "args": summarize_args(args),
            "duration_ms": round(duration_s * 1000, 1),
            "ok": ok,
        }
        size = _result_size(result)
        if size is not None:
            entry["result_count"] = size
        if isinstance(result, dict) and "complete" in result:
            # grep_email: a partial scan is the interesting case to spot later.
            entry["complete"] = result.get("complete")
            entry["scanned"] = result.get("scanned")
            entry["stop_reason"] = result.get("stop_reason")
        if error is not None:
            entry["error"] = f"{type(error).__name__}: {error}"[:_MAX_VALUE_CHARS]
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        # A failed write must never surface as a failed tool call.
        return


def instrument(tool: str) -> Callable:
    """Decorator logging one usage record per call of an MCP tool function.

    Applied *under* ``@server.tool`` so the SDK still registers the original
    signature and docstring: ``functools.wraps`` copies ``__doc__`` and
    ``__annotations__`` and sets ``__wrapped__``, which is what
    ``inspect.signature`` follows when the SDK builds the tool schema.

    Failures are logged and re-raised unchanged — the caller still sees the real
    ``ValueError``.
    """

    def decorate(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            import inspect

            try:
                bound = inspect.signature(fn).bind(*args, **kwargs)
                bound.apply_defaults()
                call_args = dict(bound.arguments)
            except Exception:
                call_args = dict(kwargs)
            started = time.monotonic()
            try:
                result = fn(*args, **kwargs)
            except BaseException as exc:
                record(tool, call_args, duration_s=time.monotonic() - started, ok=False, error=exc)
                raise
            record(tool, call_args, duration_s=time.monotonic() - started, ok=True, result=result)
            return result

        return wrapper

    return decorate
