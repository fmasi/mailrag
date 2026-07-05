"""Pass-2 stage: resumable LLM summarize+judge sweep over the profile's selection."""

from __future__ import annotations

from src.ingest.local_source import resolve_index_files
from src.llm import client as llm_client
from src.llm import rubrics, summary
from src.llm.cache import Pass2Cache
from src.llm.pass2 import run_pass


def _make_load_email(body_chars):
    from src.data.loaders.mail_archive_x import MailArchiveXLoader

    def load_email(path):
        emails = list(MailArchiveXLoader(eml_files=[path], verbose=False).load())
        if not emails:
            raise ValueError("no email parsed")
        e = emails[0]
        return {
            "sender": e.sender,
            "subject": e.subject,
            "date": e.date.isoformat() if e.date else "unknown",
            "body": e.body,
            "message_id": e.message_id or "",
        }

    return load_email


def run(profile, *, model, workers=1, body_chars=4000, limit=None, sample=None, progress=True):
    kept, _ = resolve_index_files(
        profile.resolved_root(), profile.selection_rules, getattr(profile, "blacklist", None)
    )
    cache = Pass2Cache(profile.pass2_cache)
    cl = llm_client.make_client()
    load_email = _make_load_email(body_chars)

    def summarize(email):
        return summary.parse_response(
            llm_client.chat(cl, model, rubrics.build_prompt(profile.rubric, email, body_chars))
        )

    counts = run_pass(
        kept, cache, load_email, summarize, model, limit=limit, progress=progress, workers=workers
    )
    cache.close()
    return counts
