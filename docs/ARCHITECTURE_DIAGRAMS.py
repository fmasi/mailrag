"""ASCII architecture diagrams for mailrag.

Two terminal-friendly diagrams that complement docs/ARCHITECTURE.md (the
narrative) and docs/architecture-diagram.html (the visual overview) by
naming the module behind every stage:

- ``DIAGRAM_DATA_LIFECYCLE`` — how mail becomes an indexed Qdrant
  collection (ingest -> clean -> chunk -> embed -> store).
- ``DIAGRAM_QUERY_FLOW`` — how a question becomes a grounded answer
  (hybrid retrieval, RRF fusion, optional rerank, thread expansion, one
  grounded LLM call).

Run ``python docs/ARCHITECTURE_DIAGRAMS.py`` to print both. Importing the
module is silent.
"""

DIAGRAM_DATA_LIFECYCLE = """
╔════════════════════════════════════════════════════════════════════════════╗
║         DATA LIFECYCLE — mail source to indexed Qdrant collection          ║
╚════════════════════════════════════════════════════════════════════════════╝

┌─ INGEST — the corpus is a directory of .eml files ─────────────────────────┐
│ Maildir ─┐                                                                 │
│ IMAP ────┼─► .eml files on disk; every later stage reads files,            │
│ archives ┘   so a new mail source only has to write them                   │
│                                                                            │
│ sync sources   src/sync/maildir_source.py, imap_source.py; each            │
│                fetched message is spooled to disk (src/sync/spool.py)      │
│ onboarding     src/onboard.py + src/data/loaders/mail_archive_x.py         │
│ normalise      all loaders emit NormalizedEmail (src/data/models.py)       │
│ threading      thread identity from RFC 5322 headers                       │
│                (src/data/threading.py)                                     │
└────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─ CLEAN — two passes, ordered by cost ──────────────────────────────────────┐
│ Pass-1  rules + headers, no LLM: tags noise_candidate, drops               │
│         nothing (src/data/noise_filter.py, src/pipeline/pass1.py)          │
│ scrub   base64 blobs, URL tracking params and signature blocks             │
│         stripped from bodies (src/data/body_cleanup.py);                   │
│         exact-content dedup (src/data/dedup.py)                            │
│ Pass-2  ONE LLM call per email → summary + noise verdict                   │
│         (src/pipeline/pass2.py, src/llm/pass2.py); content-keyed           │
│         cache (src/llm/cache.py) makes every re-run free                   │
│ prune   confident noise blacklisted before indexing                        │
│         (src/data/blacklist.py)                                            │
└────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─ CHUNK — structure-aware, deterministic ───────────────────────────────────┐
│ bodies       sentence-aware splitter; Pass-2 summaries optionally          │
│              prepended to the embedded text                                │
│              (src/indexing/contextual_index.py)                            │
│ attachments  per-MIME extraction + OCR into a content-addressed            │
│              store (src/attachments/); chunked by structure —              │
│              PDF page, spreadsheet row-group, slide, section —             │
│              never one giant truncated blob                                │
│              (src/indexing/attachment_chunking.py)                         │
│ point IDs    deterministic, content-derived                                │
│              (src/indexing/point_ids.py) → re-indexing is                  │
│              idempotent                                                    │
└────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─ EMBED — in-process, no serving layer ─────────────────────────────────────┐
│ bge-m3 via FlagEmbedding (src/ingest/embedder.py): ONE forward             │
│ pass emits both a 1024-d dense vector and learned-sparse lexical           │
│ weights. Pluggable Embedder protocol; the dense-only NimEmbedder           │
│ (hosted NVIDIA NIM) is the alternative — the local hybrid                  │
│ measured better on real email, which is why it is the default.             │
└────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─ STORE — Qdrant ───────────────────────────────────────────────────────────┐
│ One point per chunk with TWO named vectors — dense (cosine) and            │
│ sparse — plus a payload of email metadata, thread ID and message           │
│ key (src/ingest/hybrid_qdrant.py). One collection per corpus;              │
│ the client is built through a single seam (src/config/qdrant.py).          │
└────────────────────────────────────────────────────────────────────────────┘

Re-runs are cheap by design: the Pass-2 cache and deterministic point IDs
make every stage safe to kill and repeat, and scheduled sync resumes each
message from a per-stage SQLite ledger (src/sync/state.py).
"""

DIAGRAM_QUERY_FLOW = """
╔════════════════════════════════════════════════════════════════════════════╗
║                  QUERY FLOW — question to grounded answer                  ║
╚════════════════════════════════════════════════════════════════════════════╝

┌─ SURFACES — all share the searcher and answer path underneath ─────────────┐
│ CLI `mailrag ask` (src/cli.py) · Textual TUI (src/tui/) ·                  │
│ MCP server (src/mcp_server/) exposing seven tools:                         │
│   list_collections, search_email, get_thread, grep_email,                  │
│   answer_question, list_attachments, get_attachment                        │
└────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─ 1. EMBED THE QUERY ───────────────────────────────────────────────────────┐
│ The same in-process bge-m3 model used at index time embeds the             │
│ question into a dense vector and sparse query weights                      │
│ (src/query/bge_m3_embedding.py) — query and corpus share one               │
│ vocabulary.                                                                │
└────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─ 2. HYBRID SEARCH ─────────────────────────────────────────────────────────┐
│ A LlamaIndex VectorStoreIndex over QdrantVectorStore in hybrid             │
│ mode: Qdrant is queried on both named vectors and returns a                │
│ dense-ranked and a sparse-ranked candidate list                            │
│ (src/query/hybrid.py).                                                     │
└────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─ 3. RRF FUSION ────────────────────────────────────────────────────────────┐
│ The two lists are fused client-side by Reciprocal Rank Fusion              │
│ (src/query/fusion.py). RRF fuses ranks, not scores, so no score            │
│ normalisation is needed between legs whose scores mean different           │
│ things.                                                                    │
└────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─ 4. OPTIONAL RERANK — off by default ──────────────────────────────────────┐
│ A cross-encoder (BAAI/bge-reranker-v2-m3, or a hosted NVIDIA               │
│ reranking NIM) rescores the fused candidates; the summary-aware            │
│ variant scores on summary+body (src/query/summary_rerank.py).              │
│ Opt-in because it helps pointed questions but hurts                        │
│ thread-spanning ones.                                                      │
└────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─ 5. THREAD EXPANSION ──────────────────────────────────────────────────────┐
│ Matching chunks are expanded into whole attributed threads —               │
│ every message with sender, recipients, date and subject, plus              │
│ summaries (src/query/thread_expand.py). Threads, not chunks,               │
│ are the unit handed onward.                                                │
└────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─ 6. GROUNDED ANSWER ───────────────────────────────────────────────────────┐
│ One LLM call over the top-k threads, instructed to answer from             │
│ them alone (src/llm/answer.py), through the unified client                 │
│ (src/llm/client.py) — any OpenAI-compatible endpoint, local by             │
│ default.                                                                   │
└────────────────────────────────────────────────────────────────────────────┘

Escape hatch: the MCP tool `grep_email` (src/mcp_server/grep.py) is a
literal/regex scan over the raw .eml corpus that bypasses this pipeline
entirely — needle hunts (an ID, an amount, an error string) are exactly
where semantic retrieval is blind.
"""

if __name__ == "__main__":
    print(DIAGRAM_DATA_LIFECYCLE)
    print(DIAGRAM_QUERY_FLOW)
