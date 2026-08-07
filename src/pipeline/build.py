"""Build stage: load -> pass1 tag -> bge-m3 hybrid embed -> Qdrant.
Imports are module-level so tests can patch them by name."""

from __future__ import annotations

from src.data.loaders.mail_archive_x import MailArchiveXLoader
from src.data.noise_filter import NoiseFilter
from src.indexing.attachment_docs import build_attachment_documents
from src.indexing.contextual_index import build_contextual_index
from src.ingest.local_source import resolve_index_files
from src.llm.cache import Pass2Cache
from src.llm.pass2 import apply_pass2
from src.pipeline import pass1


def run(
    profile,
    *,
    embedder,
    recreate=True,
    embed_summary=False,
    summaries=None,
    limit=None,
    noise_min_confidence=0.7,
    index_attachments=True,
    attachment_extractor_name="tesseract",
    allow_legacy_append=False,
):
    kept, _ = resolve_index_files(
        profile.resolved_root(), profile.selection_rules, getattr(profile, "blacklist", None)
    )
    if limit:
        kept = kept[:limit]
    emails = MailArchiveXLoader(eml_files=kept).load()
    nf = NoiseFilter.from_project_rules()
    emails, stats = pass1.run(emails, nf)
    print(
        f"pass1 (zero-loss): dropped {stats.dropped}; kept {stats.kept}; "
        f"tagged {stats.tagged} noise_candidate"
    )
    if embed_summary:
        # Consume the calibrated Pass-2 judgments: drop confident noise (save DB
        # space) and embed summaries onto kept mail. pass1 stays zero-loss; the
        # raw .eml files on disk make any drop reversible by rebuilding.
        cache = Pass2Cache(profile.pass2_cache)
        try:
            emails, dropped = apply_pass2(emails, cache, min_confidence=noise_min_confidence)
        finally:
            cache.close()
        print(
            f"pass2 apply: dropped {dropped} noise (confidence >= "
            f"{noise_min_confidence}); indexing {len(emails)} with summaries"
        )
    # Attachment content (issue #80). Extract text from the attachments of the
    # emails we are actually indexing and hand it in as SEPARATE documents so a
    # fact that lives only inside an .xlsx/.pdf/.docx/.pptx becomes searchable and
    # traces back to its email via parent_message_id/thread_id. Built from the
    # final `emails` (post-pass2) so dropped-as-noise emails don't sneak their
    # attachments back in. source_id is the .eml path (set by the loader).
    attachment_docs = None
    if index_attachments:
        eml_paths = [e.source_id for e in emails if getattr(e, "source_id", None)]
        attachment_docs = build_attachment_documents(
            eml_paths, extractor_name=attachment_extractor_name
        )
        print(f"attachments: indexed text from {len(attachment_docs)} attachment(s)")

    return build_contextual_index(
        emails,
        collection=profile.collection,
        embedder=embedder,
        summaries=summaries,
        chunk_size=profile.chunk_size,
        chunk_overlap=profile.chunk_overlap,
        embed_summary=embed_summary,
        recreate=recreate,
        qdrant_url=profile.qdrant_url,
        apply_noise_filter=False,
        extra_docs=attachment_docs,
        allow_legacy_append=allow_legacy_append,
    )
