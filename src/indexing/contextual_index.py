"""Reusable contextual-retrieval build pipeline: NormalizedEmails -> hybrid Qdrant.

Shared by scripts/build_local_eml_rag.py and the main.py demo. Mirrors the original
build-mode of build_local_eml_rag.py but takes already-loaded emails + an embedder +
an optional {message_id: summary} map (so callers choose live vs cached summaries).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import MetadataMode

from src.data.dedup import dedup_by_content
from src.data.noise_filter import NoiseFilter
from src.ingest import hybrid_qdrant as hq
from src.ingest.embed_text import embed_max_length, prepend_summary
from src.ingest.sparse import lexical_weights_to_sparse


@dataclass
class BuildResult:
    """Summary of a completed build_contextual_index() run."""

    collection: str
    kept_emails: int
    chunks: int


def build_contextual_index(
    emails: List,
    *,
    collection: str,
    embedder,
    summaries: Optional[Dict[str, str]] = None,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    embed_summary: bool = False,
    embed_max_length_override: Optional[int] = None,
    embed_batch: int = 32,
    upsert_batch: int = 256,
    recreate: bool = True,
    qdrant_url: str = "http://localhost:6333",
    apply_noise_filter: bool = True,
) -> BuildResult:
    """Clean -> (inject summaries) -> split -> dedup -> bge-m3 hybrid embed -> upsert.

    Parameters
    ----------
    emails:
        Already-loaded ``NormalizedEmail`` objects to index.
    collection:
        Qdrant collection name.
    embedder:
        An :class:`~src.ingest.embedder.Embedder` — an ``encode(texts, batch_size,
        max_length)`` method returning ``(dense_vecs, sparse_weight_dicts)``, plus a
        ``dim`` used to size the collection (default 1024 if absent). Typically a
        ``BgeM3Embedder``. (A dense-only embedder needs a dense-only collection;
        that path is added with the NVIDIA-native embedder.)
    summaries:
        Optional mapping of ``message_id -> summary text``.  When supplied, each
        email's ``.summary`` field is set before chunking so that
        ``to_document()`` surfaces it as ``metadata["summary"]``.
    chunk_size:
        Token chunk size passed to ``SentenceSplitter`` (bge-m3 tokenizer).
    chunk_overlap:
        Overlap in tokens between consecutive chunks.
    embed_summary:
        When True, prepend the per-chunk summary to the embedded text (contextual
        retrieval).  Also widens the ``max_length`` by ``SUMMARY_EMBED_HEADROOM``
        tokens so the summary does not displace body content.
    embed_max_length_override:
        Explicit ``max_length`` ceiling for bge-m3 encoding.  Overrides the
        automatic ``chunk_size + headroom`` calculation when provided.
    embed_batch:
        Number of texts to encode per embedder call.
    upsert_batch:
        Number of Qdrant points to upsert per batch.
    recreate:
        Drop and recreate the collection before indexing when True.
    qdrant_url:
        URL of the Qdrant instance.

    Returns
    -------
    BuildResult
        Collection name, number of emails that survived noise-filtering, and
        total chunks upserted.
    """
    # 1. Noise filter. Callers that already pre-filter (and tag kept bulk via
    # noise_candidate) pass apply_noise_filter=False so this redundant pass does
    # not silently re-drop the bulk mail they deliberately kept.
    if apply_noise_filter:
        nf = NoiseFilter.from_project_rules()
        emails = [e for e in emails if not nf.is_noise(e)]

    # 2. Inject summaries into .summary field (surfaced by to_document())
    if summaries:
        for e in emails:
            s = summaries.get(getattr(e, "message_id", None))
            if s:
                e.summary = s

    kept_emails = len(emails)

    # 3. Convert to LlamaIndex Documents
    docs = [e.to_document(doc_id=f"{e.source}_{i}") for i, e in enumerate(emails)]
    if not docs:
        return BuildResult(collection=collection, kept_emails=0, chunks=0)

    # 4. Tokenise + split into chunks
    # NOTE: transformers is a runtime dep (not installed in mailrag-test env).
    # The import is intentionally lazy so the module is importable without it.
    # Tests that reach this code must patch 'transformers.AutoTokenizer' or
    # install transformers in the test env.
    from transformers import AutoTokenizer  # noqa: PLC0415  (lazy import by design)
    tok = AutoTokenizer.from_pretrained("BAAI/bge-m3")
    encode_len = lambda text: tok.encode(text, add_special_tokens=False)  # noqa: E731
    splitter = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        tokenizer=encode_len,
    )
    nodes = splitter.get_nodes_from_documents(docs, show_progress=False)

    # 5. Exact-content deduplication
    nodes = dedup_by_content(
        nodes, key=lambda n: n.get_content(metadata_mode=MetadataMode.NONE)
    )
    if not nodes:
        return BuildResult(collection=collection, kept_emails=kept_emails, chunks=0)

    # 6. Prepare Qdrant collection. Size it from the embedder so a non-bge-m3
    # embedder (e.g. a NIM at 2048-d) gets a correctly-sized collection; default
    # to 1024 for a minimal duck-typed embedder without the metadata.
    client = hq.get_client(qdrant_url)
    hq.ensure_hybrid_collection(
        client, collection, dim=getattr(embedder, "dim", 1024), recreate=recreate
    )

    enc_max_len = embed_max_length(
        chunk_size, embed_summary, override=embed_max_length_override
    )

    # 7. Embed in upsert_batch-sized batches and upsert
    done = 0
    for i in range(0, len(nodes), upsert_batch):
        batch = nodes[i : i + upsert_batch]
        embed_texts = []
        for n in batch:
            t = n.get_content(metadata_mode=MetadataMode.EMBED)
            if embed_summary:
                t = prepend_summary(t, n.metadata.get("summary"))
            embed_texts.append(t)
        dense, sparse = embedder.encode(
            embed_texts, batch_size=embed_batch, max_length=enc_max_len
        )
        points = []
        for n, dv, lw in zip(batch, dense, sparse):
            idx, val = lexical_weights_to_sparse(lw)
            payload = dict(n.metadata)
            payload["text"] = n.get_content(metadata_mode=MetadataMode.NONE)
            points.append(hq.make_point(n.node_id, dv, idx, val, payload))
        hq.upsert(client, collection, points)
        done += len(batch)

    return BuildResult(collection=collection, kept_emails=kept_emails, chunks=done)
