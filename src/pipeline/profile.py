"""Chunk-size profiling stage: report cleaned-body token-length distribution and a
suggested chunk_size (p90 rounded). Extracted from build_local_eml_rag.py --profile."""

from __future__ import annotations

import statistics
from dataclasses import dataclass

from src.data.loaders.mail_archive_x import MailArchiveXLoader
from src.data.noise_filter import NoiseFilter
from src.ingest.local_source import resolve_index_files
from src.ingest.profile import percentiles, suggest_chunk_size


@dataclass
class ProfileReport:
    percentiles: dict
    mean: float
    bodies: int
    suggested_chunk_size: int


def _cleaned_token_lengths(profile):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("BAAI/bge-m3")
    kept, _ = resolve_index_files(profile.resolved_root(), profile.selection_rules, None)
    emails = MailArchiveXLoader(eml_files=kept).load()
    nf = NoiseFilter.from_project_rules()
    emails = [e for e in emails if not nf.is_noise(e)]
    return [
        len(tok.encode(e.body, add_special_tokens=False)) for e in emails if (e.body or "").strip()
    ]


def run(profile, *, set_profile=False) -> ProfileReport:
    lens = _cleaned_token_lengths(profile)
    suggested = suggest_chunk_size(lens)
    report = ProfileReport(
        percentiles=percentiles(lens),
        mean=statistics.mean(lens) if lens else 0.0,
        bodies=len(lens),
        suggested_chunk_size=suggested,
    )
    if set_profile:
        profile.chunk_size = suggested
    return report
