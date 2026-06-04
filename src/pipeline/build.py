"""Build stage: load -> pass1 tag -> bge-m3 hybrid embed -> Qdrant.
Imports are module-level so tests can patch them by name."""
from __future__ import annotations
from src.ingest.local_source import resolve_index_files
from src.data.loaders.mail_archive_x import MailArchiveXLoader
from src.data.noise_filter import NoiseFilter
from src.indexing.contextual_index import build_contextual_index
from src.pipeline import pass1


def run(profile, *, embedder, recreate=True, embed_summary=False,
        summaries=None, limit=None):
    kept, _ = resolve_index_files(profile.resolved_root(), profile.selection_rules, None)
    if limit:
        kept = kept[:limit]
    emails = MailArchiveXLoader(eml_files=kept).load()
    nf = NoiseFilter.from_project_rules()
    emails, stats = pass1.run(emails, nf)
    print(f"pass1 (zero-loss): dropped {stats.dropped}; kept {stats.kept}; "
          f"tagged {stats.tagged} noise_candidate")
    return build_contextual_index(
        emails, collection=profile.collection, embedder=embedder,
        summaries=summaries, chunk_size=profile.chunk_size,
        chunk_overlap=profile.chunk_overlap, embed_summary=embed_summary,
        recreate=recreate, qdrant_url=profile.qdrant_url,
        apply_noise_filter=False,
    )
