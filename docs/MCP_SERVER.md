# MCP server — query your mail from any agent

`mailrag` ships a small [Model Context Protocol](https://modelcontextprotocol.io)
server (`src/mcp_server/`) that exposes the **existing** email-RAG query/read
pipeline over stdio, so an MCP client (Claude Code, Claude Desktop, opencode, or
any MCP-capable agent) can **discover, search, question and read attachments**
from your indexed mail without touching the internals.

It is a single, **multi-collection** server: one running process can serve any
corpus you have indexed. Every query tool takes an optional `collection` argument,
so an agent can list what is available and then target a specific corpus per call.

The server is a thin wrapper — retrieval is the same hybrid (bge-m3 dense + learned
sparse, RRF) searcher used by `mailrag ask`, answers use the same grounded-answer
path, and attachment text comes straight from the same content-addressed store the
CLI uses. No ranking, fusion, answering, or extraction logic is reimplemented.

## When to use it

- **You want an agent to query your mail.** Point Claude Code / opencode at the
  server and it can search threads, ask grounded questions, and pull attachment
  text as part of a larger task.
- **You do _not_ use it to build or maintain the index.** Ingesting mail, the
  LLM cleaning passes, and the interactive wizard stay on the CLI (they are
  long-running, need `--model` / `--workers`, or have a human-in-the-loop
  calibration gate). See the [capability matrix](#cli--mcp-capability-matrix)
  below — nothing is lost, it just lives on the CLI.

## Tools

All five tools are registered on one server. Every query tool accepts an optional
`collection`; when omitted it falls back to the server's resolved default (see
[Collection discovery & selection](#collection-discovery--selection)).

### `list_collections()`

Discover the indexed corpora on the configured Qdrant instance.

- **Args:** none.
- **Returns:** a list of rows `{name: str, points_count: int | None, is_default: bool}`.
  `points_count` is `None` when Qdrant cannot cheaply report it. `is_default` marks
  the corpus the other tools use when no `collection` is passed.
- **Errors:** a clear `ValueError` (`cannot list collections from Qdrant at <url>: …`)
  if Qdrant is unreachable, rather than a crash.

```jsonc
// list_collections()
[
  { "name": "work-rag",     "points_count": 10432, "is_default": true  },
  { "name": "personal-rag", "points_count":  3110, "is_default": false }
]
```

### `search_email(query, collection=None, top_k=5, mode="hybrid")`

Retrieve the email **threads** most relevant to `query`. No LLM call — cheap raw
material for the agent to reason over itself.

- **Args:**
  - `query` (str, required) — natural-language search query.
  - `collection` (str, optional) — corpus to search; defaults to the resolved default.
  - `top_k` (int, default 5) — maximum threads to return.
  - `mode` (str, default `"hybrid"`) — retrieval leg: `hybrid` (dense+sparse RRF),
    `dense` (dense-only), or `sparse` (sparse-only).
- **Returns:** up to `top_k` rows `{thread_id, subject, num_emails, text}`.
- **Errors:** `ValueError` on a blank `query`, `top_k < 1`, an unknown `mode`, or an
  unconfigured corpus.

```jsonc
// search_email("invoice from acme in march", collection="work-rag", top_k=3)
[
  { "thread_id": "t-91af", "subject": "Acme invoice #4021", "num_emails": 4, "text": "..." },
  { "thread_id": "t-2c0d", "subject": "Re: March billing",   "num_emails": 2, "text": "..." }
]
```

### `answer_question(query, collection=None, k=3)`

The full RAG answer path: retrieve threads, then ground a single-LLM-call answer
over the top-`k`.

- **Args:**
  - `query` (str, required) — the question to answer.
  - `collection` (str, optional) — corpus to answer from; defaults to the resolved default.
  - `k` (int, default 3) — number of retrieved threads to ground the answer on.
- **Returns:** `{answer: str, sources: [{thread_id, subject}]}`.
- **Errors:** `ValueError` on a blank `query`, `k < 1`, or an unconfigured corpus.

```jsonc
// answer_question("how much was the March Acme invoice?", collection="work-rag")
{
  "answer": "The March Acme invoice (#4021) was $12,480, due 2026-04-15.",
  "sources": [{ "thread_id": "t-91af", "subject": "Acme invoice #4021" }]
}
```

Use `search_email` when the agent wants to reason over the raw threads itself
(no model call); use `answer_question` when you want mailrag to produce the
grounded natural-language answer for you.

### `list_attachments(thread_id=None, message_id=None, collection=None)`

List the attachments belonging to a thread or a message (parity with the CLI
`mailrag attachments list`).

- **Args:**
  - `thread_id` (str, optional) — thread whose attachments to list.
  - `message_id` (str, optional) — message whose attachments to list.
  - **At least one** of `thread_id` / `message_id` is required.
  - `collection` (str, optional) — accepted for API symmetry; the attachment store
    is corpus-wide, so it is not required to select a store.
- **Returns:** a row per attachment
  `{sha256, filename, mime, size, thread_id, message_id, inline}`.
- **Errors:** `ValueError` when neither identifier is supplied.

```jsonc
// list_attachments(thread_id="t-91af")
[
  { "sha256": "9f2c…", "filename": "invoice-4021.pdf",
    "mime": "application/pdf", "size": 84213,
    "thread_id": "t-91af", "message_id": "m-01", "inline": false }
]
```

### `get_attachment(sha256, ocr=None)`

Return the **extracted text** (and metadata) for one attachment (parity with the
CLI `mailrag attachments get --text`). Raw bytes are **never** returned over MCP.

- **Args:**
  - `sha256` (str, required) — content hash of the attachment (from `list_attachments`).
  - `ocr` (str, optional) — extraction backend: `llm` | `tesseract` | `cloud`
    (like the CLI `--extractor` flag). Defaults to `$RAG_ATTACH_EXTRACTOR` or `llm`.
- **Returns:** `{sha256, filename, mime, size, text, text_status}`. `text_status`
  reports how extraction went (e.g. `ok`, `ocr_unavailable`).
- **Errors:** `ValueError` on a blank `sha256` or an unknown attachment.

```jsonc
// get_attachment("9f2c…", ocr="tesseract")
{
  "sha256": "9f2c…", "filename": "invoice-4021.pdf", "mime": "application/pdf",
  "size": 84213, "text": "ACME CORP  Invoice #4021 …", "text_status": "ok"
}
```

## Collection discovery & selection

The server is multi-collection. The typical agent flow is:

1. Call **`list_collections()`** to see what is indexed and which is the default.
2. Pass the chosen `name` as the **`collection`** argument to `search_email` /
   `answer_question` (and, if you like, `list_attachments`).

When a tool is called **without** a `collection`, the corpus is resolved in this
precedence order:

1. the **`collection`** argument (when given);
2. the **`$MAILRAG_COLLECTION`** environment variable;
3. the **latest onboarding manifest** (`latest_manifest_collection` — the corpus
   your most recent `mailrag onboard` built).

If none of those resolve, the tool returns a clear error
(`no email collection configured: set MAILRAG_COLLECTION or run 'mailrag onboard' first`)
rather than crashing. The same precedence drives the `is_default` flag returned by
`list_collections`.

## Configuration

Config mirrors `mailrag ask` and is resolved from flags/environment:

| Setting | Resolution (highest precedence first) | Notes |
|---------|---------------------------------------|-------|
| Collection | `collection` arg → `$MAILRAG_COLLECTION` → latest onboarding manifest | Per-call `collection` overrides the env default. |
| Qdrant URL | `--qdrant-url` (CLI) → `$MAILRAG_QDRANT_URL` → `$QDRANT_URL` → `http://localhost:6333` | `MAILRAG_QDRANT_URL` lets a host server target `localhost` without inheriting a container-oriented `QDRANT_URL` from `.env` (the [issue #29](https://github.com/fmasi/mailrag/issues/29) gotcha). |
| Attachment store | `$RAG_ATTACH_STORE` → `~/.mailrag/attachments` | Same default and store the CLI `attachments` verbs use; corpus-wide. |
| Answer LLM | the unified `Settings.llm` stack via the usual `RAG_*` env vars | Used only by `answer_question`. See [`BACKENDS.md`](BACKENDS.md). |
| Attachment OCR backend | `ocr` arg → `$RAG_ATTACH_EXTRACTOR` → `llm` | `$RAG_ATTACH_MAX_PAGES` bounds how many PDF pages OCR renders. |

## Running it

The server needs an already-indexed Qdrant collection — build one first with
`mailrag onboard` (see [`QUICKSTART.md`](QUICKSTART.md)). Then launch either way:

```bash
# via the CLI verb (recommended)
mailrag mcp
mailrag mcp --collection work-rag --qdrant-url http://localhost:6333

# or as a module (no console script needed)
python -m src.mcp_server
```

The process speaks MCP over stdio and blocks until the client disconnects.

## Registering with an MCP client

The repo ships no console-script `mailrag` shim, so point clients at the conda
env's Python running the module, with `PYTHONPATH` set to the repo root so
`src` is importable. Adjust the paths to your checkout.

### Claude Code

```bash
claude mcp add mailrag \
  --env MAILRAG_COLLECTION=work-rag \
  --env MAILRAG_QDRANT_URL=http://localhost:6333 \
  --env PYTHONPATH=/Users/you/Git/mailrag \
  -- /opt/miniconda3/envs/mailrag/bin/python -m src.mcp_server
```

Everything after `--` is the launch command. `list_collections` then lets the agent
discover corpora even if you leave `MAILRAG_COLLECTION` unset.

### opencode

Add a `local` MCP entry to `opencode.jsonc`:

```jsonc
{
  "mcp": {
    "mailrag": {
      "type": "local",
      "command": [
        "/opt/miniconda3/envs/mailrag/bin/python",
        "-m",
        "src.mcp_server"
      ],
      "environment": {
        "PYTHONPATH": "/Users/you/Git/mailrag",
        "MAILRAG_COLLECTION": "work-rag",
        "MAILRAG_QDRANT_URL": "http://localhost:6333"
      }
    }
  }
}
```

### Claude Desktop

`claude_desktop_config.json` uses the same shape as Claude Code:

```jsonc
{
  "mcpServers": {
    "mailrag": {
      "command": "/opt/miniconda3/envs/mailrag/bin/python",
      "args": ["-m", "src.mcp_server"],
      "env": {
        "PYTHONPATH": "/Users/you/Git/mailrag",
        "MAILRAG_COLLECTION": "work-rag",
        "MAILRAG_QDRANT_URL": "http://localhost:6333"
      }
    }
  }
}
```

## CLI ↔ MCP capability matrix

mailrag's full capability set, and where each lives. The MCP server deliberately
exposes only the **query/read** surface; build, ingest, and interactive steps stay
on the CLI. **Nothing is lost** — the CLI is the complete surface, and the MCP
tools are the agent-facing subset.

| Capability | CLI verb | MCP tool | Why it lives where it does |
|------------|----------|----------|----------------------------|
| Discover indexed corpora | *(implicit)* | `list_collections` | Read-only lookup — cheap and safe for agents. |
| Search threads | `ask` / `query` (retrieval) | `search_email` | Pure query; the agent's raw-material path. |
| Grounded answer | `ask` / `query` | `answer_question` | Pure query + one LLM call; the agent's answer path. |
| List attachments | `attachments list` | `list_attachments` | Read-only metadata lookup. |
| Read attachment text | `attachments get --text` | `get_attachment` | Read-only text extraction (never raw bytes over MCP). |
| Build an assistant end-to-end | `onboard` | — CLI only | Long-running ingest + validation; needs models/flags, not an agent round-trip. |
| Full-screen guided pipeline | `wizard` | — CLI only | Interactive full-screen TUI — human-in-the-loop, not stdio-scriptable. |
| Embed & index a corpus | `index` / `build` | — CLI only | Long-running GPU/MPS embedding job; run deliberately, not per agent call. |
| Run a persona recipe | `run` | — CLI only | Orchestrates a multi-stage build with `--model` / `--workers`. |
| Choose folders/accounts | `scope` / `select` | — CLI only | Configuration step against local mail, done once up front. |
| Measure corpus / chunk size | `measure` / `profile` | — CLI only | Build-time profiling that feeds indexing. |
| Tag bulk noise (Pass 1) | `tag` / `pass1` | — CLI only | Bulk regex/header pass over the whole corpus. |
| Cluster to surface noise | `scan` / `explore` | — CLI only | Bulk embedding clustering, part of the cleaning pipeline. |
| Calibrate the cleaning rubric | `calibrate` | — CLI only | **Human-in-the-loop gate** — you review the rubric before a full pass. |
| Cheap LLM noise verdict | `judge` | — CLI only | Bulk LLM pass with `--model` / `--workers`. |
| LLM summary + noise pass | `summarize` / `pass2` | — CLI only | Long-running one-LLM-call-per-email pass; gated on a calibrated rubric. |
| Blacklist confident noise | `prune` | — CLI only | Destructive corpus edit before re-indexing. |
| Ingest attachments | `attachments build` | — CLI only | Long-running extraction/OCR over the profile's mail. |
| Run the MCP server | `mcp` | *(this server)* | Launches the stdio server itself. |

If you need any CLI-only capability, run it from a shell in the `mailrag` conda env
(e.g. `conda run -n mailrag mailrag onboard …`); it then becomes queryable through
the MCP tools above.
