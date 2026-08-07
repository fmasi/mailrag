"""Pass-2 stage: resumable LLM summarize+judge sweep over the profile's selection."""

from __future__ import annotations

from src.ingest.local_source import resolve_index_files
from src.llm import client as llm_client
from src.llm import rubrics, summary
from src.llm.cache import Pass2Cache
from src.llm.pass2 import run_pass
from src.llm.provenance import describe_backend


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

    # Capture WHICH judge this sweep uses, and say so before spending anything.
    # A model id alone does not identify one — quantisation and local-vs-remote
    # change the output — and a corpus judged by two of them cannot be compared
    # with itself afterwards.
    prov = describe_backend(model=model, api_base=getattr(cl, "base_url", ""))
    print(f"summarize judge: {prov.label()}")
    existing = cache.judges()
    if existing:
        print(f"  cache already holds judgments from: {existing}")
        # A DIFFERENT MODEL is the thing that breaks comparability. The same model
        # with the quant newly recorded is not a second judge — it is better
        # metadata — so that case is a note, not a warning, or this would cry wolf
        # on every run for the life of the cache.
        other_models = {k.split("@")[0].split(" [")[0] for k in existing} - {prov.model}
        if other_models:
            print(
                f"  WARNING: this sweep ADDS A SECOND MODEL to the same corpus.\n"
                f"           existing: {sorted(other_models)}\n"
                f"           this run: {prov.model}\n"
                f"           Noise rates across the corpus stop being comparable."
            )
        elif any("@" not in k for k in existing):
            print(
                "  note: earlier rows predate quantisation recording, so their quant "
                "is unknown. Same model, so judgments remain comparable."
            )

    def summarize(email):
        return summary.parse_response(
            llm_client.chat(cl, model, rubrics.build_prompt(profile.rubric, email, body_chars))
        )

    counts = run_pass(
        kept,
        cache,
        load_email,
        summarize,
        model,
        limit=limit,
        progress=progress,
        workers=workers,
        provenance=prov,
    )
    cache.close()
    return counts
