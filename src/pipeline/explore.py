"""Explore stage: cluster the corpus embeddings at thread granularity and rank
noise pockets without spending the LLM.

Vectors auto-resolve: reuse the profile's Qdrant collection if it exists, else
embed fresh with BGE-M3. Metadata + pass1 tags always come from loading the
selected .eml on disk. See docs/superpowers/specs/2026-06-05-explore-verb-design.md.
"""

from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import numpy as np

from src.cluster.noise_pockets import (
    ClusterReport,
    ThreadMeta,
    cluster_threads,
    format_report,  # re-export for the CLI
)
from src.data.noise_filter import NoiseFilter
from src.data.threading import compute_thread_id
from src.ingest.embedder import BgeM3Embedder
from src.ingest.hybrid_qdrant import get_client
from src.ingest.local_source import resolve_index_files
from src.ingest.qdrant_vectors import read_thread_vectors
from src.pipeline import pass1

__all__ = ["aggregate_threads", "run", "format_report"]


def _thread_id(email) -> str:
    return compute_thread_id(
        getattr(email, "message_id", "") or "",
        getattr(email, "in_reply_to", "") or "",
        getattr(email, "references", "") or "",
        subject=getattr(email, "subject", "") or "",
    )


def _is_tagged(email) -> bool:
    return bool(getattr(email, "noise_candidate", False) or getattr(email, "is_bulk", False))


def aggregate_threads(emails) -> Tuple[List[ThreadMeta], Dict[str, List[int]]]:
    """Group tagged emails into per-thread metadata.

    Returns ``(metas, tid_to_indices)`` where ``metas[j].thread_id`` corresponds
    to a ``tid_to_indices`` key; the index lists point back into ``emails`` (used
    to mean-pool fresh per-email vectors)."""
    tid_to_idx: Dict[str, List[int]] = defaultdict(list)
    for i, e in enumerate(emails):
        tid_to_idx[_thread_id(e)].append(i)

    metas: List[ThreadMeta] = []
    for tid, idxs in tid_to_idx.items():
        members = [emails[i] for i in idxs]
        senders = [getattr(m, "sender", "") or "unknown" for m in members]
        sender_counts = Counter(senders)
        dominant, dom_count = sender_counts.most_common(1)[0]
        subjects, seen = [], set()
        for m in members:
            s = getattr(m, "subject", "") or ""
            if s and s not in seen:
                seen.add(s)
                subjects.append(s)
            if len(subjects) >= 3:
                break
        metas.append(
            ThreadMeta(
                thread_id=tid,
                n_emails=len(members),
                dominant_sender=dominant,
                top_sender_share=round(dom_count / len(members), 4),
                n_senders=len(sender_counts),
                tag_fraction=round(sum(_is_tagged(m) for m in members) / len(members), 4),
                sample_subjects=subjects or ["(no subject)"],
                # source_id is the .eml file path (source is just the loader type);
                # downstream judge/prune hash these paths, so they must be real files.
                paths=[getattr(m, "source_id", "") or "" for m in members],
            )
        )
    return metas, dict(tid_to_idx)


def _load_emails(paths):
    from src.data.loaders.mail_archive_x import MailArchiveXLoader

    return MailArchiveXLoader(eml_files=paths).load()


def _fresh_thread_vectors(emails, tid_to_idx, *, batch_size=32) -> Dict[str, np.ndarray]:
    """Embed each email body once, then mean-pool per thread."""
    embedder = BgeM3Embedder(device="mps", use_fp16=True)
    dense, _ = embedder.encode(
        [getattr(e, "body", "") or "" for e in emails], batch_size=batch_size
    )
    dense = np.asarray(dense, dtype=float)
    return {tid: dense[idxs].mean(axis=0) for tid, idxs in tid_to_idx.items()}


def run(
    profile,
    *,
    json_path: str,
    clusters: int | None = None,
    seed: int = 11,
    limit: int | None = None,
    force_fresh: bool = False,
    qdrant_url: str | None = None,
    top: int = 15,
    profile_path: str = "",
) -> ClusterReport:
    """Resolve thread vectors (Qdrant-or-fresh), cluster, write JSON, return the
    report. ``profile_path`` is recorded in the JSON for provenance only."""
    kept, _ = resolve_index_files(profile.resolved_root(), profile.selection_rules, None)
    if limit:
        kept = kept[:limit]
    if not kept:
        raise ValueError("no files selected; check the profile's selection_rules")

    emails = _load_emails(kept)
    pass1.run(emails, NoiseFilter.from_project_rules())  # tags noise_candidate
    metas, tid_to_idx = aggregate_threads(emails)

    url = qdrant_url or getattr(profile, "qdrant_url", None) or "http://localhost:6333"
    vector_source = "fresh"
    tvecs: Dict[str, np.ndarray] = {}
    if not force_fresh:
        client = get_client(url)
        if client.collection_exists(profile.collection):
            tvecs = read_thread_vectors(client, profile.collection)
            vector_source = "qdrant"
    if vector_source == "fresh":
        tvecs = _fresh_thread_vectors(emails, tid_to_idx)

    # Align metas to the threads that actually have a vector (a built collection
    # may have dropped some at index time; fresh embed covers all).
    aligned = [(m, tvecs[m.thread_id]) for m in metas if m.thread_id in tvecs]
    if not aligned:
        raise ValueError("no thread vectors resolved")
    metas_v = [m for m, _ in aligned]
    matrix = np.array([v for _, v in aligned], dtype=float)

    report = cluster_threads(matrix, metas_v, k=clusters, seed=seed)
    report.vector_source = vector_source

    payload = report.to_json_dict(
        profile=profile_path, collection=profile.collection, vector_source=vector_source
    )
    os.makedirs(os.path.dirname(os.path.abspath(json_path)), exist_ok=True)
    with open(json_path, "w") as fh:
        json.dump(payload, fh, indent=2)
    return report
