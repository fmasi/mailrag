# Architecture

mailrag turns a personal email archive into a locally hosted, queryable knowledge
base. Everything runs on one machine: embedding in-process, the LLM behind a
local OpenAI-compatible endpoint, and Qdrant in Docker. The cloud is an option
at two seams (the LLM endpoint and the embedder), never a requirement.

This document describes the system end to end: the data flow, the design
decisions and why they were made, the module map, and the seams that matter.
Deep dives live in the companion docs — [RETRIEVAL_GUIDE.md](RETRIEVAL_GUIDE.md),
[CHUNKING.md](CHUNKING.md), [SYNC.md](SYNC.md), [MCP_SERVER.md](MCP_SERVER.md),
[BACKENDS.md](BACKENDS.md) and [VERBS.md](VERBS.md) — and are referenced rather
than repeated.

## The pipeline at a glance

```
             ingest                    clean                       chunk
  Maildir ─┐                 ┌──────────────────────┐   ┌─────────────────────────┐
  IMAP ────┼─► .eml files ──►│ Pass-1: rules/headers │──►│ bodies: sentence-aware  │
  .eml ────┘  (the corpus    │   (tag, never drop)   │   │ attachments: structure- │
  archives     is a          │ Pass-2: one LLM call  │   │   aware (page / row-    │
               directory     │   per email —         │   │   group / slide) + OCR  │
               of files)     │   summary + verdict   │   └───────────┬─────────────┘
                             └──────────────────────┘               │
                                                                    ▼
             answer                    retrieve                    embed & store
  ┌─────────────────────┐   ┌───────────────────────────┐   ┌─────────────────────┐
  │ one grounded LLM    │◄──│ hybrid dense+sparse, RRF  │◄──│ bge-m3 in-process:  │
  │ call over the top-k │   │ fusion, optional cross-   │   │ dense 1024-d +      │
  │ threads             │   │ encoder rerank, then      │   │ learned sparse, one │
  └─────────────────────┘   │ expansion into whole      │   │ forward pass ──►    │
                            │ attributed threads        │   │ Qdrant (two named   │
                            └───────────────────────────┘   │ vectors per chunk)  │
                                                            └─────────────────────┘
```

Four surfaces sit on top of this: a CLI of cost-ordered verbs, a full-screen
Textual TUI, a continuous sync runner under launchd/systemd, and a seven-tool
MCP server for agent clients.

## Ingest

The corpus is a directory of `.eml` files. That is a deliberate anchor, not an
accident: every downstream stage (selection, noise filtering, judging,
attachment extraction, chunking, embedding) is driven by files on disk, so any
new mail source only has to materialise messages as `.eml` and the rest of the
pipeline needs no changes.

Three ways in:

- **Archive onboarding** (`mailrag onboard`, `src/onboard.py`) — point it at a
  directory of `.eml` files and it builds a validated, thread-aware collection
  in one bounded LLM pass. Parsing is done by `MailArchiveXLoader`
  (`src/data/loaders/mail_archive_x.py`).
- **Continuous sync** (`mailrag sync`, `src/sync/`) — fetches new mail from a
  local **Maildir** (`maildir_source.py`) or over **IMAP**
  (`imap_source.py`), spools each message to disk as `.eml`
  (`spool.py`), then judges and indexes the delta. See [SYNC.md](SYNC.md).
- **Other loaders** (`src/data/loaders/`) — a small pluggable family behind the
  `EmailLoader` base class; today that means the public Enron dataset
  (HuggingFace) used by the demo and evals. All loaders normalise into one
  `NormalizedEmail` dataclass (`src/data/models.py`), so downstream code never
  knows the source.

Thread identity is derived at ingest (`src/data/threading.py`) from RFC 5322
headers — the root of `References`, else `In-Reply-To`, else the message's own
`Message-ID` — with a normalised-subject fallback for header-less corpora such
as the public Enron set. It is not full JWZ threading, but in practice messages
in a thread share a `References` root, and the simple key groups reliably.

### Sync is spool-first, with per-stage skipping

Onboarding fails fast — if Qdrant is down there is no point starting a
six-hour build. Scheduled sync is the opposite: it runs unattended on a laptop
that sleeps, changes networks, and has Docker stopped half the time. So the
runner (`src/sync/runner.py`) treats fetch, judge and index as independent
stages: whatever backend is unavailable, the earlier stages still run, and a
SQLite ledger (`src/sync/state.py`) records per-message progress
(`fetched` / `judged` / `indexed`) so the next run picks up exactly where this
one could not. Cursors advance past poison messages rather than wedging a
folder forever. Scheduling is launchd on macOS and a systemd user timer on
Linux (`src/sync/schedule.py`) — not cron, because cron silently skips ticks
that fall while a laptop is asleep.

## Clean

Noise handling is two-pass, ordered by cost, and calibrated before it is
trusted. The prompts and thresholds are corpus-specific; `mailrag calibrate`
judges a sample against the rubric and buckets the mistakes before any
full-scale sweep.

**Pass-1 — rules and headers, no LLM** (`src/data/noise_filter.py`,
`src/pipeline/pass1.py`). Curated rules in `config/noise_rules.yaml` (sender
domains, sender/subject regexes) plus a header-driven bulk filter
(`List-Unsubscribe`, `Precedence: bulk`) with keep-guards so human mailing-list
traffic and transactional receipts survive. Crucially this pass **tags** — it
sets a `noise_candidate` flag and drops nothing. Cheap heuristics get to raise
suspicion; only the LLM pass gets to condemn.

**Pass-2 — one LLM call per email** (`src/pipeline/pass2.py`,
`src/llm/pass2.py`, `src/llm/summary.py`). A single prompt returns both a
summary and a noise judgement, because the marginal cost of a second question
inside one call is near zero while a second call per email doubles the bill.
Results are cached (`src/llm/cache.py`) keyed on a content hash that
deliberately excludes the `Message-ID`, so a re-export of the same mail is a
cache hit, and the sweep is resumable — endpoint failures never consume a
message's retry budget. Confident noise can then be blacklisted before
indexing (`mailrag prune`, `src/data/blacklist.py`).

Between the passes, bodies get a final scrub (`src/data/body_cleanup.py`):
leaked base64/data-URI blobs, URL tracking parameters (`utm_*`, `fbclid`, …)
and RFC 3676 signature blocks are stripped. This matters more here than in a
store-and-search tool — a 30 KB base64 blob is not just a diluted vector, it
is several junk chunks, real embedding compute and burned LLM tokens, and
tracking parameters defeat the exact-content dedup in `src/data/dedup.py`.

There is also an embeddings-side scan (`mailrag scan`,
`src/cluster/noise_pockets.py`) that clusters vectors to surface noise pockets
the rules missed, and a cheap verdict-only LLM check (`mailrag judge`) over the
suspects. See [EMAIL_PREPROCESSING.md](EMAIL_PREPROCESSING.md).

## Chunk

Email bodies are split by a sentence-aware splitter inside the build pipeline
(`src/indexing/contextual_index.py`), with chunk size suggested per corpus by
`mailrag measure`. Pass-2 summaries can optionally be prepended to the embedded
text (`--embed-summary`) so each chunk carries thread context into the vector.

Attachments get their own treatment (`src/attachments/`,
`src/indexing/attachment_chunking.py`). Extraction is per-MIME-type
(PDF, DOCX, XLSX/CSV, PPTX, HTML, images, plain text) with OCR through
Tesseract or an LLM-vision fallback, and bytes live in a content-addressed
store (`src/attachments/store.py`: sha256-keyed blobs plus a SQLite index,
write-once, deduplicated). Chunking is **structure-aware**, because a
sentence splitter given a flattened spreadsheet finds no sentence boundaries,
emits one giant chunk, and the embedder silently truncates it at 8 192 tokens
— observed, not hypothetical. So spreadsheets are chunked by row-groups with
the header row repeated in every chunk, PDFs by page, PPTX by slide, DOCX by
heading/section, with a token-budget hard cap on every emitted chunk. Details
in [CHUNKING.md](CHUNKING.md).

Every chunk gets a deterministic point ID derived from content
(`src/indexing/point_ids.py`), which is what makes re-indexing idempotent —
sync can re-run the index stage over already-processed mail for free. A policy
fingerprint (`src/indexing/policy.py`) recorded on the collection detects when
an existing collection was built under different settings than the ones now in
force, instead of silently appending incompatible data.

## Embed

Embedding is in-process — there is no serving layer. The default is
**bge-m3** via FlagEmbedding (`src/ingest/embedder.py`), which emits a 1024-d
dense vector *and* learned sparse lexical weights from one forward pass. That
single property drives the retrieval design: hybrid search costs one model,
one pass. On Apple Silicon it runs on MPS by default (`src/ingest/device.py`).

The seam is a small structural `Embedder` protocol: `name`, `dim`,
`produces_sparse`, and `encode(texts) -> (dense, sparse)`. A second
implementation (`NimEmbedder`, backed by hosted NVIDIA embedding NIMs) is
dense-only — an OpenAI-style `/embeddings` endpoint returns a single dense
vector and cannot carry learned sparse weights, so a remotely embedded
collection loses the sparse leg. This was measured, not assumed: on a real
personal-email corpus the local bge-m3 hybrid beat the dense+rerank NIM
configuration, which is why it is the default. See [BACKENDS.md](BACKENDS.md)
and the measured comparison in [EXPERIMENTS.md](EXPERIMENTS.md).

## Store

The vector store is **Qdrant**. Each chunk is one point with two named vectors
— `dense` (1024-d, cosine) and `sparse` (bge-m3 lexical weights) — plus a
payload carrying the email metadata, thread ID and message key
(`src/ingest/hybrid_qdrant.py`). One collection per corpus; the client is
built through a single seam (`src/config/qdrant.py`) so URL/auth handling
lives in exactly one place.

Qdrant is *the* vector backend — there is no provider switch and no second
persistence path. The `SimpleVectorStore` and Pinecone branches, along with the
Azure Blob loader and the cloud batch-indexing scripts that were their only
callers, were removed in [#49](https://github.com/fmasi/mailrag/issues/49).
The reasoning is in [ROADMAP.md](ROADMAP.md): storing learned-sparse vectors
alongside dense ones as named vectors on the same point is a Qdrant-specific
facility, so a portable posture would cost more than it buys.

## Retrieve

Retrieval (`src/query/hybrid.py`) is deliberately framework-native: a
LlamaIndex `VectorStoreIndex` over `QdrantVectorStore` in hybrid mode. The
custom pieces are exactly two adapters and one callback:

- **bge-m3 adapters** (`src/query/bge_m3_embedding.py`) — dense query
  embedding and a sparse-query function over the same in-process model used at
  index time, so query and corpus share one vocabulary.
- **RRF fusion** (`src/query/fusion.py`) — dense and sparse result lists are
  fused by Reciprocal Rank Fusion, supplied through the framework's documented
  `hybrid_fusion_fn` hook. RRF fuses *ranks*, not scores, so it needs no score
  normalisation between two legs whose scores mean different things.
- **Optional cross-encoder rerank** — off by default; when enabled it scores
  the fused candidates with `BAAI/bge-reranker-v2-m3` (or a hosted NVIDIA
  reranking NIM). The summary-aware variant (`src/query/summary_rerank.py`)
  scores on summary+body so thread context informs *ranking* without being
  embedded into the vector, where it causes drift. The benchmark verdict:
  reranking helps pointed questions and hurts thread-spanning ones, hence
  opt-in rather than default.

The last step is what makes retrieval email-shaped: **thread expansion**
(`src/query/thread_expand.py`). Chunks are what match, but a chunk of an email
is a terrible unit to hand to an LLM — it has no sender, no date, no
surrounding conversation. So ranked hits are expanded into whole attributed
threads (every message with sender/recipients/date/subject, plus summaries),
and threads are the unit passed onward. `HybridSearcher.search()` returns
nodes; `search_threads()` returns `ThreadContext` objects.

A HyDE query-expansion helper (`src/query/hyde.py`) exists for bridging the
question-to-answer vocabulary gap. Tuning knobs and mode trade-offs
(hybrid / dense / sparse) are covered in
[RETRIEVAL_GUIDE.md](RETRIEVAL_GUIDE.md).

## Answer

Answering is one grounded LLM call (`src/llm/answer.py`): the top-k retrieved
threads are joined into a context block and the model is instructed to answer
from them alone. There is a single answer path shared by the CLI, the demo,
the onboarding report and the MCP server — one prompt to maintain, one place
to fix.

## The LLM seam

All single-turn completions — Pass-2 judging, summaries, HyDE, answering — go
through one client (`src/llm/client.py`), a LlamaIndex `OpenAILike` LLM wired
into `Settings.llm`. Any OpenAI-compatible endpoint works: LM Studio (the
local default), Ollama, vLLM, NVIDIA NIM, or OpenAI itself, selected by
`RAG_*` environment variables (endpoint, model, key). Production is fully
offline by default; the only cloud caller is the dev-only eval harness. A thin
raw-OpenAI shim exists solely for the inline-image vision path used by OCR
fallback, which `OpenAILike` cannot carry.

## Surfaces

**CLI** (`src/cli.py`, run as `python -m src.cli` via the repo-root `mailrag`
shim). The pipeline verbs form a cost-ordered ladder — free inspection first,
LLM spend last:

```
scope · measure · tag · scan · judge · calibrate · summarize · prune · index · ask
```

plus `onboard` (one-shot build), `sync`, `run` (execute a named persona
recipe), `wizard`, `mcp`, and `attachments build|list|get`. Personas
(`src/persona/`, `personas.yaml`) bundle the ladder into named recipes so a
user picks an intent rather than ten flags. See [VERBS.md](VERBS.md).

**TUI** (`src/tui/`). A full-screen Textual wizard
(Welcome → Persona → Model → Scope → Review → Run) over the same verb
handlers as the CLI. All flow decisions live in `src/tui/flow.py`; `app.py` is
view code only, and the whole app is driven headlessly in tests via Textual's
Pilot. Tour in [GUIDE.md](GUIDE.md).

**Continuous sync** (`mailrag sync --install-agent`). Renders a launchd
LaunchAgent or systemd user timer that runs the one-shot, idempotent sync on a
schedule, with explicit conda-environment invocation and logging — because the
most common failure mode of scheduled jobs is dying on the first import,
silently, for weeks.

**MCP server** (`src/mcp_server/`, `mailrag mcp`). A stdio server exposing the
read/query surface to agent clients (Claude Code, etc.) as seven tools:
`list_collections`, `search_email`, `get_thread`, `grep_email`,
`answer_question`, `list_attachments`, `get_attachment`. Every tool is a thin
wrapper over the existing stack — nothing about ranking, fusion or answering
is reimplemented — and every query tool takes an optional `collection`, so one
running server serves any indexed corpus. `grep_email` deserves a note: it is
a literal/regex scan over the raw `.eml` corpus that bypasses embeddings
entirely, because needle hunts (an ID, an amount, an error string) are exactly
where semantic retrieval is blind. Search output is snippet-bounded with a
hard character cap so a single tool call can never flood an agent's context.
See [MCP_SERVER.md](MCP_SERVER.md).

## Module map

| Path | Responsibility |
|---|---|
| `src/data/loaders/` | Pluggable sources → `NormalizedEmail` (.eml archives, Enron/HF) |
| `src/data/` | Models, threading, dedup, body cleanup, rule-based noise filter, blacklist |
| `src/pipeline/` | The verb stages: profile, pass1, scan, judge, calibrate, pass2, prune, build |
| `src/llm/` | Unified client, Pass-2 orchestration + cache, summaries, rubrics, answering |
| `src/attachments/` | Content-addressed store, per-MIME extraction handlers, OCR |
| `src/indexing/` | Build pipeline, structure-aware attachment chunking, deterministic IDs, policy |
| `src/ingest/` | Embedder protocol + bge-m3/NIM impls, Qdrant hybrid collection management |
| `src/query/` | Hybrid searcher, RRF fusion, rerankers, thread expansion, HyDE |
| `src/sync/` | Maildir/IMAP sources, spool, ledger, runner, scheduler units |
| `src/mcp_server/` | Seven-tool stdio MCP server, corpus grep |
| `src/persona/`, `src/tui/` | Named recipes over the verbs; full-screen Textual wizard |
| `src/cluster/` | Embedding-space noise-pocket discovery |
| `src/eval/` | Retrieval/answer evaluation harness (dev-only) |
| `src/config/` | Qdrant client seam, secrets; plus the legacy `RAGConfig` |

## Design decisions, in one place

- **Local-first, cloud-optional.** Embedding in-process, LLM behind a local
  endpoint, Qdrant in Docker. The two escape hatches are explicit seams, not
  scattered conditionals.
- **The corpus is files.** `.eml` on disk is the join between every ingest
  path and the pipeline; sync integrates by writing files, not by threading a
  new code path through ten stages.
- **One LLM call per email.** Summary and noise verdict share a prompt; a
  content-keyed cache makes every re-run free. Cost discipline is a design
  input, not an afterthought.
- **Tag, don't drop.** Cheap heuristics only flag; nothing is destroyed
  without the LLM pass, and nothing is swept at scale without calibration on a
  sample first.
- **Hybrid from one model.** bge-m3's dense+sparse single pass makes hybrid
  retrieval nearly free, and rank-based RRF avoids cross-leg score
  normalisation. Reranking stays opt-in because it is not uniformly a win.
- **Threads are the answer unit.** Chunks match; attributed conversations are
  what the LLM (and the human) actually need.
- **Idempotence everywhere.** Deterministic point IDs, per-stage ledgers,
  resumable sweeps: any stage can be killed and re-run without loss or
  duplication.

## Extension points

- **New mail source** — either subclass `EmailLoader`
  (`src/data/loaders/base.py`) for batch loading, or implement a sync source
  (`src/sync/sources.py`) that yields messages the spool writes as `.eml`.
- **New embedder** — satisfy the `Embedder` protocol
  (`src/ingest/embedder.py`) and register it in `make_embedder`. Declare
  `produces_sparse` honestly; the index and query paths size and route from it.
- **New LLM** — no code: point `RAG_LLM_API_BASE` (and friends) at any
  OpenAI-compatible endpoint.
- **New attachment type** — add an extract handler
  (`src/attachments/extract/handlers/`) and, if the format has structure worth
  preserving, a chunker in `src/indexing/attachment_chunking.py`.
- **New agent tool** — add a plain function plus a thin `@mcp.tool` wrapper in
  `src/mcp_server/server.py`; keep it a wrapper over the existing stack.
