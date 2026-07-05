"""Unified, resumable clean+summarize pass for ZTH onboarding.

For each email (walked in thread order with preceding context) produce a
``{is_noise, confidence, summary, reason}`` record via the thread-aware prompt and
persist it to a content-addressed Pass2Cache, so an interrupted mailbox run
resumes for free. Unlike src.llm.thread_summaries.generate_thread_summaries, this
KEEPS the noise judgment so the caller can drop noise from the corpus.
"""

from collections import defaultdict

from src.data.identity import content_sha256
from src.llm.client import chat, default_model, make_client
from src.llm.summary import build_thread_aware_prompt, parse_response
from src.llm.thread_summaries import _as_dict, _dkey, _tid


def _record_from_row(row):
    return {
        "is_noise": bool(row["is_noise"]),
        "confidence": float(row["confidence"]),
        "summary": row["summary"] or "",
        "reason": row["reason"] or "",
    }


def generate_thread_judgments(emails, *, cache, model=None, client=None, progress=None):
    """Return ``{message_id: {is_noise, confidence, summary, reason}}`` for every
    email with a message_id. Resumable via ``cache`` (a Pass2Cache). ``progress``,
    if given, is called once per email (e.g. ``tqdm(...).update``)."""
    model = model or default_model()
    client = client or make_client()

    by_thread = defaultdict(list)
    for e in emails:
        by_thread[_tid(e)].append(e)

    out = {}
    for _tid_key, thread in by_thread.items():
        thread = sorted(thread, key=_dkey)
        preceding = []
        for e in thread:
            mid = getattr(e, "message_id", "") or ""
            ed = _as_dict(e)
            sha = content_sha256(sender=ed["sender"], subject=ed["subject"], body=ed["body"])
            row = cache.get(sha)
            if row is not None:
                rec = _record_from_row(row)
            else:
                try:
                    raw = chat(client, model, build_thread_aware_prompt(ed, preceding))
                    rec = parse_response(raw)
                    cache.put(sha, rec, model=model, message_id=mid or None, content_sha256=sha)
                except Exception as exc:  # conservative keep: never cache a transient
                    rec = {
                        "is_noise": False,
                        "confidence": 0.0,
                        "summary": "",
                        "reason": f"llm_error: {exc}",
                    }  # not persisted -> retried next run
            if mid:
                out[mid] = rec
            preceding.append(ed)
            if progress is not None:
                progress()
    return out
