# mailrag v0.9.0 — first tagged release

*Dated by the tag when it is cut.*

This is the first tagged release of mailrag, so these notes introduce the whole
project rather than a delta. The repository has been developed in public since
May 2026 (365 commits, 52 merged PRs); everything below describes the state of
`main` at the tag.

## What mailrag is

A local-first email RAG system: it turns a personal mail archive into a private,
queryable knowledge base that runs on your own hardware, on open models, with
nothing required to leave your network. Cloud endpoints are supported but
optional. It is one half of a self-owned context stack for AI agents — the
email half; its sister project [parley](https://github.com/fmasi/parley) covers
calls and meetings.

The design bet, borne out by the project's own evaluation, is that email
retrieval is won at the *thread* level, not the message level: match a small
unit, then answer from its entire conversation.

## What it does today

- **Ingestion** from the public Enron corpus (HuggingFace), local `.eml`
  archives, or Azure Blob Storage, behind one `EmailLoader` interface;
  per-part body decoding (quoted-printable/base64), HTML stripping and
  multipart/alternative dedup.
- **A cost-ordered cleanup pipeline** expressed as CLI verbs —
  `scope → measure → tag → scan → judge → calibrate → summarize → prune → index`.
  Free regex tagging first, cheap no-LLM embedding-cluster triage next, LLM
  judgement and per-email contextual summaries last, so each email costs at
  most one LLM call. Only `index` ever deletes anything; every earlier stage
  tags rather than drops.
- **Attachments, extracted and indexed** — PDFs, Word/Excel/PowerPoint, HTML
  and images, with OCR for scans (local Tesseract, or a local vision model as
  the privacy-first default). Chunking is structure-aware: spreadsheets by
  row-group with the header repeated per chunk, PDFs by page, decks by slide,
  so a figure in a 500-row sheet is not truncated at the embedder's token
  limit. Every chunk carries a back-reference to its email and thread.
- **Hybrid retrieval** — bge-m3 dense + learned-sparse vectors fused with RRF
  in Qdrant, with optional cross-encoder reranking. Thread-aware expansion
  assembles the full conversation before answering.
- **Continuous sync** — `./mailrag sync` fetches new mail from a live account
  (IMAP or Maildir) and indexes only the delta. Point ids are deterministic,
  so re-indexing replaces rather than duplicates; a content-addressed LLM
  cache means only new mail costs an LLM call. `--install-agent` writes a
  launchd LaunchAgent (macOS) or a systemd user timer (Linux), both chosen for
  their run-on-wake semantics, with the environment a scheduled job will not
  otherwise inherit pinned into the unit.
- **Interfaces** — a CLI (`./mailrag <verb>`), a full-screen Textual TUI
  wizard for the guided pipeline, one-command onboarding
  (`./mailrag onboard`), and a read-only, multi-collection MCP server
  (`./mailrag mcp`) exposing seven tools to any MCP client:
  `list_collections`, `search_email`, `get_thread`, `grep_email`,
  `answer_question`, `list_attachments`, `get_attachment`.

## Architecture in brief

```
load (.eml / Enron / Azure) → tag (regex, no LLM) → scan (clusters, no LLM)
  → judge / summarize (local LLM, cached) → prune → chunk (structure-aware)
  → embed (bge-m3 dense + sparse) → Qdrant (hybrid RRF)
  → query (thread expansion · optional rerank) → CLI / TUI / MCP
```

Backends are pluggable: any OpenAI-compatible LLM endpoint (LM Studio, Ollama,
vLLM, NVIDIA NIM, OpenAI) via one LlamaIndex `Settings.llm` seam, and Qdrant,
local persistence or Pinecone for storage. See `docs/ARCHITECTURE.md`,
`docs/BACKENDS.md` and `docs/CHUNKING.md`.

## Install and run

Requires Python 3.11–3.13 and Docker (for Qdrant).

```bash
git clone https://github.com/fmasi/mailrag.git
cd mailrag
pip install -r requirements.txt   # first run downloads ~2 GB of bge-m3 weights
cp .env.example .env              # point at an LLM endpoint (local or cloud)
make demo                         # Qdrant up, contextual index over public Enron mail, example queries
```

Development installs use Poetry: `poetry install --with dev --all-extras`.
For your own mail, start with `./mailrag onboard` or the wizard
(`./mailrag wizard`); see `docs/QUICKSTART.md` and `docs/GUIDE.md`.

## What is verified

- **1,500 tests pass** (`unittest`-style under pytest; 3 live-integration
  tests are deselected by default). The suite runs on every PR and push to
  `main` with a coverage floor of 85% enforced in CI.
- **CI gates**, each a separately named required check: pytest + coverage,
  ruff (lint and format), mypy, pip-audit against the OSV database with
  **zero** `--ignore-vuln` entries, GitHub dependency review, and CodeQL.
  All GitHub Actions are pinned to commit SHAs.
- **Retrieval evaluation**: a 360-query eval on a real ~32k-email private
  mailbox, with confound controls and significance reporting
  (`docs/EXPERIMENTS.md`). Headline: thread-level recall@5 of **93%** against
  a plain-dense message-level baseline of **46%**, with thread reconstruction
  the single largest lever. The email-tuned bge-m3 hybrid was also
  benchmarked against NVIDIA's general-purpose retrieval stack: it wins on
  email while NVIDIA's wins on TREC legal e-discovery — same systems, opposite
  winners. These figures are author-reported on a private corpus; the
  *method* (not the private numbers) is reproducible on public Enron data via
  `make demo`, and the core techniques were cross-checked on the public
  Enron-QA benchmark with the same ordering.
- **The MCP `get_thread` fix** was verified against a live collection: a probe
  of 12 thread ids returned by `search_email` resolved 3/12 before the fix and
  12/12 after.

## Changes of note (recent work, grouped by meaning)

### Features

- MCP server migrated to the official MCP SDK v2 (#110). `FastMCP` became
  `MCPServer`; the server now speaks the 2026-07-28 protocol revision while
  still serving 2025-era clients from the same object, so the change is
  additive for existing MCP clients. The 1.x SDK line entered maintenance
  mode on 2026-07-28, which is why the floor is `mcp >= 2`, not merely
  preferred.
- Continuous sync shipped (#101): provider-agnostic `MessageSource` seam,
  IMAP (CONDSTORE/CHANGEDSINCE) and Maildir sources, resumable cursor with
  poison-message parking, `--install-agent` scheduling, `sync --status`
  staleness warning, and `start_from` to begin where a backup export ended.
- Structure-aware attachment chunking (#89) and attachment content indexing
  (#80) — attachments were previously extracted but silently unsearchable.
- `grep_email` MCP tool for exact-string search alongside semantic search.

### Fixes

- `get_thread` now resolves thread ids by an exact payload-filter key lookup
  in Qdrant rather than by embedding the id and hoping the owning thread
  ranks in the top-k (#109). An opaque message-id has no semantic
  relationship to its thread, so the old path resolved only ~25% of the ids
  `search_email` had just returned; it is now 100%. Surrounding angle
  brackets on incoming ids are normalised before validation.
- Message bodies are decoded per-part (quoted-printable/base64) with HTML
  stripped and multipart/alternative deduplicated (#81); exact numbers are
  normalised for both the sparse and dense legs so reference numbers and
  amounts are findable (#82).
- `search_email` output is bounded (it could previously return 130K
  characters of full-thread text), and `answer_question` reports LLM auth
  failures and endpoint health honestly (#83, #84).

### Security

- A dependency sweep cleared **all 11 open Dependabot alerts** in one pass,
  lifting six transitive packages past their advisories (aiohttp,
  cryptography, nltk, h2, torch, setuptools) via declared security floors in
  `pyproject.toml`, each annotated with its advisory IDs (#105). pypdf was
  subsequently floored at 6.15.0 for CVE-2026-71870/71852 — caught by the
  pip-audit gate before Dependabot alerted.
- The pip-audit CI gate now runs with an empty ignore list, and the workflow
  documents the policy for keeping it that way.
- The Qdrant server image is pinned by tag **and** digest
  (`qdrant/qdrant:v1.18.1@sha256:45f8e3…`) rather than tracking `:latest`:
  Qdrant's storage-format migrations are one-way, so an accidental
  `docker compose pull` across a version boundary must not be possible.

### Performance and cost

- One-LLM-call-per-email as a design rule: cheap `judge` for mail you will
  drop, or `summarize` for mail you keep, never both; the content-addressed
  cache makes re-runs free.
- Incremental indexing by default: deterministic point ids mean a re-run
  replaces rather than duplicates, and `--recreate` is an explicit opt-in.

### Developer experience

- ruff and mypy are now declared as exactly-pinned dev dependencies
  (`ruff 0.15.20`, `mypy 2.1.0`), matching the versions CI runs, so the
  local gates cannot disagree with the remote ones (#108).
- The two dependency files (`pyproject.toml` and `requirements.txt`) are kept
  expressing the same constraints, so a pip install cannot diverge from a
  Poetry one.

### Notable behaviour and deliberate constraints

- **qdrant-client is capped at `>=1.18.0,<1.19`.** 1.19.0 removed
  `IDF_EMBEDDING_MODELS` from `qdrant_client.qdrant_fastembed`, which
  `llama-index-vector-stores-qdrant` (0.10.2, its latest) still imports at
  module load — so an uncapped resolve turns every dense/hybrid/MCP code path
  into an `ImportError`. The cap is a single minor version, meaning a
  security fix shipped as 1.19.x would be unreachable until it is lifted;
  issue #106 tracks the trigger to act. 1.18 also matches the pinned server.
- `get_thread` still accepts its `mode` parameter so the tool signature does
  not break, but a key lookup has no ranking to vary, so it no longer
  influences the result; a test pins this.
- Deleting mail on the server does **not** remove it from the index. This is
  archive semantics by design: the collection keeps what it has seen, and
  there is no reconciliation sweep.
- The MCP server is read-only by design; build, ingest and interactive steps
  stay on the CLI.

## Known limitations

- The headline retrieval numbers come from a private mailbox and are
  author-reported; they cannot be independently reproduced from this
  repository. `make demo` reproduces the method on public data.
- Sync sources are IMAP and Maildir only — no Gmail API, Exchange/EWS or
  mbox-watching; scheduling helpers cover macOS (launchd) and Linux
  (systemd user timers) only.
- OCR depends on system binaries (Tesseract, Poppler) being installed; when
  they are absent the pipeline degrades cleanly rather than failing, which
  also means scans silently contribute no text.
- The mypy gate is deliberately lenient: it checks existing annotations
  rather than enforcing full typing, and twelve pre-annotation-era modules
  are excluded per-module.
- There is no packaged distribution: Poetry runs with
  `package-mode = false`, there is no PyPI artefact and no installed console
  script — the project runs from a checkout via the `./mailrag` shim.

## Deliberately not done yet

- Lifting the qdrant-client cap (waiting on upstream
  `llama-index-vector-stores-qdrant`; #106).
- Any web UI or hosted service — local-first is the point, and the agent
  surface is MCP.
- Delete reconciliation in sync (archive semantics are intentional, but a
  reconciliation mode may become an option).

## Versioning

This release is tagged **v0.9.0**, and the number is deliberate rather than
modest. mailrag is feature-complete for its own purpose and in daily use —
indexing two live accounts, syncing continuously under a launchd agent, and
serving an MCP client — but it does not yet commit to interface stability. The
CLI surface, corpus profile formats and MCP tool schemas may still change
without ceremony, which is precisely what a 0.x signals.

**v1.0.0 is the next release, not a distant one.** It follows once a named set
of gaps closes: the remaining bugs, a wider MCP capability surface, and the
work that lets a stranger verify the headline retrieval claim on their own
machine rather than taking it on trust. The concrete milestone list lives in
[ROADMAP.md](ROADMAP.md) — it ships when that list is done, not on a date.

With `package-mode = false` the version declared in `pyproject.toml` is
informational — nothing is published to a package index — so the tag, not the
file, is the release's identity.
