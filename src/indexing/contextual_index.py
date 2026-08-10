"""Reusable contextual-retrieval build pipeline: NormalizedEmails -> hybrid Qdrant.

Shared by scripts/build_local_eml_rag.py and the main.py demo. Mirrors the original
build-mode of build_local_eml_rag.py but takes already-loaded emails + an embedder +
an optional {message_id: summary} map (so callers choose live vs cached summaries).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import MetadataMode

from src.data.dedup import dedup_by_content
from src.data.noise_filter import NoiseFilter
from src.indexing.point_ids import assign_deterministic_ids, content_hash
from src.indexing.policy import describe_mismatch, policy_fingerprint
from src.ingest import hybrid_qdrant as hq
from src.ingest.embed_text import embed_max_length, prepend_summary
from src.ingest.numeric import augment_numeric
from src.ingest.sparse import lexical_weights_to_sparse

log = logging.getLogger(__name__)


def _split_documents(splitter, docs):
    """Split *docs*, quarantining any document the splitter rejects.

    ``SentenceSplitter`` raises when a document's METADATA alone exceeds the
    chunk budget — which one inbound email with a very long subject in a
    token-dense script can cause. Splitting the whole list in one call means that
    single message takes every other document in the batch down with it, and the
    caller has no way to tell which one was at fault. The fast path is still the
    bulk call; only on failure does it fall back to per-document splitting to
    isolate the offender.

    Returns ``(nodes, failed_message_keys)``.
    """
    try:
        return splitter.get_nodes_from_documents(docs, show_progress=False), frozenset()
    except Exception:  # noqa: BLE001 — identify the culprit rather than give up
        pass

    nodes, failed = [], set()
    for doc in docs:
        try:
            nodes.extend(splitter.get_nodes_from_documents([doc], show_progress=False))
        except Exception as exc:  # noqa: BLE001 — one bad document, not a bad batch
            key = doc.metadata.get("message_key") or doc.doc_id
            failed.add(key)
            log.warning("skipping undindexable document %s: %s", key, exc)
    return nodes, frozenset(failed)


@dataclass
class BuildResult:
    """Summary of a completed build_contextual_index() run."""

    collection: str
    kept_emails: int
    chunks: int
    # Documents the splitter refused (e.g. metadata alone over the chunk budget).
    # Reported so the caller can give them a terminal state instead of retrying
    # the same failure on every run.
    failed_message_keys: frozenset = frozenset()
    # message_keys that actually produced points in THIS run. Callers that track
    # per-message state (the sync ledger) must mark only these as indexed — an
    # email whose chunks were all removed by the corpus-wide dedup contributes
    # nothing, and recording it as indexed would strand it forever.
    indexed_message_keys: frozenset = frozenset()


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
    extra_docs: Optional[List] = None,
    allow_legacy_append: bool = False,
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
        Drop and recreate the collection before indexing when True.  When False
        the run is **incremental**: the collection is left in place and each
        indexed email's existing points are deleted before its new ones are
        upserted, so re-indexing the same mail is idempotent and re-indexing
        changed mail leaves no stale chunks (issue #101).
    qdrant_url:
        URL of the Qdrant instance.
    extra_docs:
        Optional pre-built LlamaIndex ``Document`` objects to index alongside the
        email bodies — the attachment documents from
        ``src.indexing.attachment_docs.build_attachment_documents`` (issue #80).
        They are chunked with the same splitter but carry their own
        ``content_kind="attachment"`` / ``attachment_name`` / ``parent_message_id``
        payload so an attachment hit traces back to its email. Kept separate from
        the body documents so a terse 4-line body never dilutes a 500-row sheet.
    allow_legacy_append:
        Permit an incremental run into a collection built before deterministic ids
        existed.  Off by default because such a run *duplicates* every chunk rather
        than replacing it (see ``hq.has_legacy_points``); the fix is a one-time
        ``recreate=True`` rebuild, which costs no LLM calls.

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
            # Header-less corpora leave message_id empty; those emails simply
            # have no summary to inject rather than keying the map on None.
            message_id = getattr(e, "message_id", None)
            s = summaries.get(message_id) if message_id else None
            if s:
                e.summary = s

    kept_emails = len(emails)

    # 3. Convert to LlamaIndex Documents. Attachment documents (issue #80) are
    # appended as their own Documents so they chunk independently of the body —
    # a 4-line body must not be split into the same chunk as a 500-row sheet.
    # doc_id is left to default to the stable ``body:<message_key>`` — the old
    # positional ``<source>_<i>`` id changed whenever the corpus order changed,
    # which made the point ids derived from it change too (issue #101).
    docs = [e.to_document() for e in emails]
    if extra_docs:
        docs.extend(extra_docs)
    if not docs:
        return BuildResult(collection=collection, kept_emails=0, chunks=0)

    # 4. Tokenise + split into chunks
    # NOTE: transformers is a runtime dep (not installed in mailrag env).
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
    nodes, failed_keys = _split_documents(splitter, docs)

    # 4b. Deterministic point ids (issue #101). Assigned BEFORE the dedup pass:
    # dedup is corpus-order-dependent, so deriving a chunk's ordinal after it would
    # let an unrelated email elsewhere in the run shift this email's ids.
    assign_deterministic_ids(nodes)

    # 5. Exact-content deduplication. NOTE this is corpus-wide: a chunk whose
    # exact text already appeared under a DIFFERENT email is dropped here, so an
    # email can finish this step contributing zero nodes.
    nodes = dedup_by_content(nodes, key=lambda n: n.get_content(metadata_mode=MetadataMode.NONE))
    if not nodes:
        return BuildResult(
            collection=collection,
            kept_emails=kept_emails,
            chunks=0,
            failed_message_keys=failed_keys,
        )

    # 5b. The set of emails that will actually be written. Derived from the
    # SURVIVING nodes, never from `docs`: deleting an email's existing points and
    # then upserting nothing for it would erase it from the collection while the
    # caller believed it had been indexed (found in review of #101).
    surviving_keys = frozenset(k for k in (n.metadata.get("message_key") for n in nodes) if k)

    # 6. Prepare the Qdrant collection. Size it from the embedder so a non-bge-m3
    # embedder (e.g. a NIM at 2048-d) gets a correctly-sized collection (default
    # 1024 for a minimal duck-typed embedder). A dense-only embedder
    # (produces_sparse=False) gets a dense-only collection — its endpoint cannot
    # emit learned sparse weights, so there is no sparse leg to populate.
    hybrid = getattr(embedder, "produces_sparse", True)
    dim = getattr(embedder, "dim", 1024)
    fingerprint = policy_fingerprint(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embed_summary=embed_summary,
        embedder_name=type(embedder).__name__,
        dim=dim,
    )
    client = hq.get_client(qdrant_url)
    if hybrid:
        hq.ensure_hybrid_collection(client, collection, dim=dim, recreate=recreate)
    else:
        hq.ensure_dense_collection(client, collection, dim=dim, recreate=recreate)

    # 6b. Append mode (recreate=False): this is an *incremental* run into a
    # collection that may already hold these emails. Delete their existing points
    # first so the upsert is a true replacement — deterministic ids make re-writing
    # a chunk idempotent, but only a delete removes a chunk that no longer exists
    # (an email re-processed into fewer chunks would leave a stale tail).
    # ensure_payload_indexes backfills the message_key index on collections built
    # before it existed, since the delete filter depends on it.
    if not recreate:
        if not allow_legacy_append and hq.has_legacy_points(client, collection):
            raise RuntimeError(
                f"collection '{collection}' holds points written before deterministic "
                "ids (no message_key payload). Appending would duplicate every chunk "
                "instead of replacing it. Rebuild it once with recreate=True "
                f"(`mailrag index --profile <p> --recreate`) — this costs no LLM calls, "
                "every judgment is already cached — or pass allow_legacy_append=True "
                "to index anyway and accept the duplicates."
            )
        # An append under different preprocessing/chunking rules would put two
        # incomparable vector populations in one collection with no signal that
        # it happened (src/indexing/policy.py).
        existing_policy = hq.collection_policy(client, collection)
        if existing_policy and existing_policy != fingerprint:
            raise RuntimeError(describe_mismatch(collection, existing_policy, fingerprint))
        hq.ensure_payload_indexes(client, collection)
        # NOTE: the delete happens per BATCH inside the upsert loop below, not
        # here. Deleting the whole delta up front means a failure partway through
        # the loop leaves every not-yet-rewritten email deleted with no
        # replacement — reproduced as 4 of 6 emails vanishing from a live
        # collection while the run reported only "index deferred".

    enc_max_len = embed_max_length(chunk_size, embed_summary, override=embed_max_length_override)

    # 7. Embed in upsert_batch-sized batches and upsert
    done = 0
    deleted_keys: set = set()
    for i in range(0, len(nodes), upsert_batch):
        batch = nodes[i : i + upsert_batch]
        if not recreate:
            # Replace this batch's emails immediately before writing them, so an
            # interrupted loop can only ever leave ONE batch in the deleted-but-
            # not-rewritten window instead of the entire delta.
            fresh = {
                k
                for k in (n.metadata.get("message_key") for n in batch)
                if k and k not in deleted_keys
            }
            if fresh:
                hq.delete_by_message_keys(client, collection, fresh)
                deleted_keys |= fresh
        embed_texts = []
        for n in batch:
            t = n.get_content(metadata_mode=MetadataMode.EMBED)
            if embed_summary:
                t = prepend_summary(t, n.metadata.get("summary"))
            # Append canonical numeric tokens ($210,000,000 -> 210000000) so the
            # sparse/dense legs gain an exact-figure hit (issue #82). The same
            # augmentation runs at query time (src/query/hybrid path) so both sides
            # share the token-id vocabulary. The stored `text` payload below keeps
            # the untouched surface form.
            t = augment_numeric(t)
            embed_texts.append(t)
        dense, sparse = embedder.encode(embed_texts, batch_size=embed_batch, max_length=enc_max_len)
        points = []
        for n, dv, lw in zip(batch, dense, sparse):
            payload = dict(n.metadata)
            payload["text"] = n.get_content(metadata_mode=MetadataMode.NONE)
            # Durable counterpart to the in-run content dedup: lets a later delta
            # run recognise a chunk it already holds (issue #101).
            payload["content_hash"] = content_hash(payload["text"])
            payload["policy_fingerprint"] = fingerprint
            if hybrid:
                idx, val = lexical_weights_to_sparse(lw)
                points.append(hq.make_point(n.node_id, dv, idx, val, payload))
            else:
                points.append(hq.make_dense_point(n.node_id, dv, payload))
        hq.upsert(client, collection, points)
        done += len(batch)

    return BuildResult(
        collection=collection,
        kept_emails=kept_emails,
        chunks=done,
        failed_message_keys=failed_keys,
        indexed_message_keys=surviving_keys,
    )
