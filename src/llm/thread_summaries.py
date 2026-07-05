"""Generate per-email PRECEDING-thread-context summaries for a list of emails (live).

In-memory analogue of scripts/eval/gen_thread_summaries.py::run(), for the demo build.
Returns {message_id: summary}; noise emails get "".
"""

from collections import defaultdict
from datetime import datetime, timezone

from src.data.threading import compute_thread_id
from src.llm.client import chat, default_model, make_client
from src.llm.summary import build_thread_aware_prompt, parse_response


def _as_dict(e):
    """Convert a NormalizedEmail to the dict shape expected by build_thread_aware_prompt."""
    return {
        "sender": getattr(e, "sender", "") or "",
        "date": getattr(e, "date", None),
        "subject": getattr(e, "subject", "") or "",
        "body": getattr(e, "body", "") or "",
        "message_id": getattr(e, "message_id", "") or "",
    }


def _tid(e):
    """Derive the thread_id the same way NormalizedEmail.to_document() does.

    Passes the subject so that header-less datasets (e.g. the public HF Enron
    corpus) group by normalised subject slug instead of all landing in a single
    empty-key bucket.
    """
    tid = getattr(e, "thread_id", None)
    if tid:
        return tid
    return compute_thread_id(
        getattr(e, "message_id", "") or "",
        getattr(e, "in_reply_to", "") or "",
        getattr(e, "references", "") or "",
        subject=getattr(e, "subject", "") or "",
    )


def _dkey(e):
    """Sort key: datetime (tz-normalised) or ISO string, then message_id fallback."""
    d = getattr(e, "date", None)
    if isinstance(d, datetime):
        if d.tzinfo is not None:
            d = d.astimezone(timezone.utc).replace(tzinfo=None)
        return (0, d.isoformat())
    if d is not None:
        return (0, str(d))
    return (1, getattr(e, "message_id", "") or "")


def generate_thread_summaries(emails, *, model=None, client=None):
    """Generate per-email summaries conditioned on the preceding thread context.

    Args:
        emails: Iterable of :class:`~src.data.models.NormalizedEmail` objects.
        model:  LLM model name; defaults to :func:`~src.llm.client.default_model`.
        client: Pre-built LLM client; defaults to :func:`~src.llm.client.make_client`.

    Returns:
        dict mapping ``message_id`` -> summary string.  Noise emails map to ``""``.
        Emails with no ``message_id`` are skipped.
    """
    model = model or default_model()
    client = client or make_client()

    # Group by thread_id (mirrors gen_thread_summaries._tid logic exactly).
    by_thread: dict = defaultdict(list)
    for e in emails:
        tid = _tid(e)
        by_thread[tid].append(e)

    # Sort each thread oldest-first, then walk append-only.
    out: dict = {}
    for tid, thread in by_thread.items():
        thread = sorted(thread, key=_dkey)
        # preceding is a list of *dicts* — _format_preceding calls .get() on each item.
        preceding: list = []
        for e in thread:
            mid = getattr(e, "message_id", "") or ""
            ed = _as_dict(e)
            try:
                prompt = build_thread_aware_prompt(ed, preceding)
                raw = chat(client, model, prompt)
                rec = parse_response(raw)
                out[mid] = "" if rec["is_noise"] else rec["summary"]
            except Exception:  # noqa: BLE001 - never crash the caller
                out[mid] = ""
            preceding.append(ed)
    return out
