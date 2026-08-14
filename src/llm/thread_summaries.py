"""Generate per-email PRECEDING-thread-context summaries for a list of emails (live).

In-memory analogue of scripts/eval/gen_thread_summaries.py::run(), for the demo build.
Returns {message_id: summary}; noise emails get "".
"""

import logging
from collections import defaultdict
from datetime import datetime, timezone

from src.data.threading import compute_thread_id
from src.llm.client import chat, default_model, healthcheck, make_client
from src.llm.summary import build_thread_aware_prompt, parse_response

log = logging.getLogger(__name__)


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


class SummaryGenerationError(RuntimeError):
    """Summary generation could not proceed, or failed for most of the corpus."""


def generate_thread_summaries(
    emails, *, model=None, client=None, preflight=True, max_failure_rate=0.25
):
    """Generate per-email summaries conditioned on the preceding thread context.

    Args:
        emails: Iterable of :class:`~src.data.models.NormalizedEmail` objects.
        model:  LLM model name; defaults to :func:`~src.llm.client.default_model`.
        client: Pre-built LLM client; defaults to :func:`~src.llm.client.make_client`.
        preflight: Verify the endpoint once before spending a call per email.
        max_failure_rate: Abort if more than this fraction of calls fail.

    Returns:
        dict mapping ``message_id`` -> summary string.  **A judged-noise email maps
        to ``""``; an email whose call FAILED is absent from the mapping entirely.**
        Emails with no ``message_id`` are skipped.

    Raises:
        SummaryGenerationError: if the preflight fails, or if the failure rate
        exceeds *max_failure_rate*.

    The absent-vs-empty distinction exists because it used to be missing (#135):
    every exception was converted to ``""``, which is also the value meaning "the
    model judged this noise". A dead endpoint therefore produced a corpus in which
    every email looked like a confident noise verdict, with no error and exit 0 —
    silently disabling the contextual-summary lever this function exists to
    provide. Failing loudly on a total outage matters more than never raising.
    """
    model = model or default_model()
    client = client or make_client()

    if preflight:
        # One call, before the corpus. A misconfigured endpoint should cost a
        # round trip, not one per email plus a plausible-looking empty result.
        try:
            healthcheck(client, model=model)
        except Exception as exc:  # noqa: BLE001 — re-raised with context below
            raise SummaryGenerationError(
                f"LLM endpoint unusable, no summaries generated: {exc}"
            ) from exc

    # Group by thread_id (mirrors gen_thread_summaries._tid logic exactly).
    by_thread: dict = defaultdict(list)
    for e in emails:
        tid = _tid(e)
        by_thread[tid].append(e)

    # Sort each thread oldest-first, then walk append-only.
    out: dict = {}
    attempted = 0
    failures: list = []
    for tid, thread in by_thread.items():
        thread = sorted(thread, key=_dkey)
        # preceding is a list of *dicts* — _format_preceding calls .get() on each item.
        preceding: list = []
        for e in thread:
            mid = getattr(e, "message_id", "") or ""
            ed = _as_dict(e)
            attempted += 1
            try:
                prompt = build_thread_aware_prompt(ed, preceding)
                raw = chat(client, model, prompt)
                rec = parse_response(raw)
                out[mid] = "" if rec["is_noise"] else rec["summary"]
            except Exception as exc:  # noqa: BLE001 — one bad email must not end the run
                if not failures:
                    # First one only: a broken endpoint would otherwise log per email.
                    log.warning(
                        "summary generation failed for %s (%s: %s); "
                        "failed emails are omitted from the result, not marked noise",
                        mid or "<no message-id>",
                        type(exc).__name__,
                        exc,
                    )
                failures.append((mid, exc))
                # Deliberately NOT out[mid] = "" — see the docstring.
            preceding.append(ed)

    if failures and attempted and len(failures) / attempted > max_failure_rate:
        raise SummaryGenerationError(
            f"{len(failures)} of {attempted} summary calls failed "
            f"({100 * len(failures) / attempted:.0f}%, limit "
            f"{100 * max_failure_rate:.0f}%). First error: "
            f"{type(failures[0][1]).__name__}: {failures[0][1]}"
        )
    if failures:
        log.warning("%d of %d summary calls failed and were omitted", len(failures), attempted)
    return out
