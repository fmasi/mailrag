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
sparse, RRF) searcher used by `./mailrag ask`, answers use the same grounded-answer
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

All seven tools are registered on one server. Every query tool accepts an optional
`collection`; when omitted it falls back to the server's resolved default (see
[Collection discovery & selection](#collection-discovery--selection)).

> **Bounded output.** `search_email` returns a **snippet window + metadata** per
> hit, not the full thread body (a single call used to emit ~130 K chars). Pull a
> full thread on demand with `get_thread` (or `search_email(..., full=True)`).
> For exact needle hunts — a number, an ID, an email address, an error string —
> use `grep_email`, a literal/regex scan over the raw corpus that bypasses
> embeddings entirely (mind its [cost model](#cost-model--why-the-scan-is-bounded)).
> None of those three read inside attachments: for a figure in a spreadsheet or a
> PDF, go `list_attachments` → `get_attachment`.

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

### `search_email(query, collection=None, top_k=5, mode="hybrid", max_chars=500, full=False)`

Retrieve the email **threads** most relevant to `query`. No LLM call — cheap raw
material for the agent to reason over itself. **Output is bounded** (issue #84):
each hit is a snippet window centred on the query match plus metadata, never the
full thread body.

- **Args:**
  - `query` (str, required) — natural-language search query.
  - `collection` (str, optional) — corpus to search; defaults to the resolved default.
  - `top_k` (int, default 5) — maximum threads to return.
  - `mode` (str, default `"hybrid"`) — retrieval leg: `hybrid` (dense+sparse RRF),
    `dense` (dense-only), or `sparse` (sparse-only).
  - `max_chars` (int, default 500) — snippet window size per hit. Clamped to a hard
    cap of **4000** so a single call can never emit an unbounded payload.
  - `full` (bool, default `false`) — return the full thread `text` instead of a
    snippet. The opt-in escape hatch when you genuinely need the whole body.
- **Returns:** up to `top_k` **bounded** rows
  `{thread_id, subject, num_emails, snippet, date, last_date, from, to, message_ids, attachment_names}`.
  With `full=true`, the `snippet` field is replaced by `text` (the whole thread).
- **Errors:** `ValueError` on a blank `query`, `top_k < 1`, `max_chars < 1`, an
  unknown `mode`, or an unconfigured corpus.

```jsonc
// search_email("invoice from acme in march", collection="work-rag", top_k=2)
[
  { "thread_id": "t-91af", "subject": "Acme invoice #4021", "num_emails": 4,
    "snippet": "…the March Acme invoice #4021 was $12,480, due 2026-04-15…",
    "date": "2026-03-02T09:14:00+00:00", "from": "billing@acme.com",
    "to": "you@co.com", "message_ids": ["m-01", "m-02"], "attachment_names": [] }
]
```

### `get_thread(thread_id, collection=None, mode="hybrid")`

Fetch the **full text** of one thread by `thread_id` — the full-body companion to
the bounded `search_email` (issue #84). Given an id from a `search_email` hit,
returns that thread's complete attributed text plus metadata.

Resolution is an exact **payload-filter key lookup** on the stored `thread_id`
(`HybridSearcher.thread_by_id`), not a search — so any id `search_email` returns
resolves deterministically. (It used to re-run retrieval with the id as the query
and scan the hits, which resolved only ~25% of ids — fixed in issue #109.) Ids
are normalised before lookup, so `<abc@host>` and `abc@host` both work.

- **Args:**
  - `thread_id` (str, required) — the id from a `search_email` result row.
  - `collection` (str, optional) — corpus to read; defaults to the resolved default.
  - `mode` (str, **ignored**) — accepted for backward compatibility only; was the
    retrieval leg before #109. A key lookup has no ranking to vary, so passing it
    changes nothing.
- **Returns:** `{thread_id, subject, num_emails, text, date, last_date, from, to, message_ids, attachment_names}`.
- **Errors:** `ValueError` on a blank id, an unknown thread, or an unconfigured corpus.

### `grep_email(pattern, collection=None, max_matches=50, regex=False, max_files=None, max_seconds=60)`

Literal / regex search over the **raw email corpus** — no embeddings (issue #82).
Walks the raw `.eml` files, decodes each body (quoted-printable + base64, HTML
stripped to text), and returns matching lines plus message metadata. This is the
escape hatch for exact needle hunts (a number, an ID, an email address, an error
string) where dense/hybrid retrieval is blind to numerals and identifiers.

- **Args:**
  - `pattern` (str, required) — the string to find (literal by default).
  - `collection` (str, optional) — accepted for API symmetry; grep is
    corpus-directory based (see the `MAILRAG_EML_ROOT` config below).
  - `max_matches` (int, default 50) — maximum matching **messages** to return.
    Clamped to a hard cap of **500**. Set it to `1` for an existence check.
  - `regex` (bool, default `false`) — treat `pattern` as a Python regex.
  - `max_files` (int, optional) — stop after scanning this many messages.
  - `max_seconds` (float, default 60, hard cap 900) — wall-clock budget for the
    scan. `null` disables the deadline, which is only safe on a small corpus.
- **Returns:** `{matches, scanned, corpus_files, complete, stop_reason, elapsed_s, root}`.
  `matches` holds up to `max_matches` rows
  `{subject, from, to, date, message_id, attachment_names, matches, path}`, where
  the inner `matches` is a list of matched-line snippets.
- **Errors:** `ValueError` on a blank `pattern`, an invalid regex, a non-positive
  `max_seconds`, or a missing corpus (`MAILRAG_EML_ROOT` unset and `~/rag_eml`
  absent).

```jsonc
// grep_email("210,000,000")
{
  "matches": [
    { "subject": "Global Partnership Staff call recap",
      "from": "Dana.Reyes@northwind.example", "to": "team@northwind.example",
      "date": "Wed, 30 Jul 2025 …", "message_id": "<a1b2c3@northwind.example>",
      "attachment_names": ["Q3 MBO targets partner team.xlsx"],
      "matches": ["…20% of the $210 million annual plan…"],
      "path": "/Users/you/rag_eml/Inbox/Wind River/… .eml" }
  ],
  "scanned": 1841, "corpus_files": 73219, "complete": false,
  "stop_reason": "max_matches", "elapsed_s": 4.4, "root": "/Users/you/rag_eml"
}
```

#### Cost model — why the scan is bounded

There is no index behind `grep_email`: every call decodes raw `.eml` files one at
a time, at roughly **2ms per message**. On a real personal corpus (73k messages /
11 GB is typical) a full pass is minutes of CPU warm, and considerably worse cold.

The match cap alone does **not** bound that work — it stops the walk early only
when the pattern actually *hits*. So:

| Pattern | Behaviour |
|---------|-----------|
| Matches often | Returns in seconds — stops at `max_matches`. |
| Matches rarely / never | Scans the **entire** corpus. |

Which makes *"does this string appear anywhere?"* the most expensive question you
can ask here — and it is the common one. `regex=true` costs no more per message
than a literal (measured: 1.6 vs 1.8 ms/file); what costs is how rarely the
pattern hits. A loose regex that matches nothing is the worst case, and one such
call previously ran an agent into a 30-minute client timeout with nothing to show
for it.

Hence `max_seconds` (default 60) and `max_files`, and hence the scan report:

> **An empty `matches` means "not in this corpus" only when `complete` is
> `true`.** Otherwise the needle is merely absent from the first `scanned` of
> `corpus_files` messages — a weaker claim, and one worth stating as such rather
> than reporting the needle as missing. Re-run with a larger budget to settle it.

For a pure literal over the raw tree, shell `rg` is far faster than this tool —
its one blind spot is base64/quoted-printable bodies, which is exactly what
`grep_email` decodes.

> **Scope note.** `grep_email` searches the message **subject + decoded body**
> only. It does **not** yet read attachment *bytes* — a spreadsheet cell or PDF
> text buried in an `.xlsx`/`.pdf` will not match, though the attachment's
> filename is reported in `attachment_names`. Use `list_attachments` +
> `get_attachment` to read inside those. Attachment content indexing is tracked
> separately (issue #80).

### `answer_question(query, collection=None, k=3)`

The full RAG answer path: retrieve threads, then ground a single-LLM-call answer
over the top-`k`. Before the LLM call it runs a **one-shot healthcheck** on the
configured endpoint + key (issue #83), so a mis-configured LLM fails at init with
a clear, actionable message naming **`RAG_LLM_API_KEY`** — not a raw 401 on every
query. The healthcheck is scoped to this path only: `search_email` and `grep_email`
keep working even when the LLM is down.

- **Args:**
  - `query` (str, required) — the question to answer.
  - `collection` (str, optional) — corpus to answer from; defaults to the resolved default.
  - `k` (int, default 3) — number of retrieved threads to ground the answer on.
- **Returns:** `{answer: str, sources: [{thread_id, subject}]}`.
- **Errors:** `ValueError` on a blank `query`, `k < 1`, or an unconfigured corpus;
  `LLMHealthcheckError` (with a `RAG_LLM_API_KEY` / `RAG_LLM_API_BASE` /
  `RAG_LLM_MODEL` hint) when the LLM endpoint is unreachable or rejects the key.

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

List the files attached to a thread or a message (parity with the CLI
`./mailrag attachments list`) — and the way in to their contents.

> **The store must be built once, separately.** `mailrag onboard` / `index` /
> `sync` do **not** populate it: they extract attachment *text* for retrieval
> down a different path (`src/indexing/attachment_docs.py`), which is why
> attachment content can be fully searchable while `list_attachments` returns
> nothing for every thread. Run `./mailrag attachments build --profile
> <corpus.profile.json>` once to populate it. Both attachment tools now raise an
> actionable error naming that command when the store is empty, rather than
> answering like a thread that simply has no attachments.

> **Attachment contents are invisible to `search_email`, `answer_question` and
> `grep_email`.** Those index message *bodies* only. So when the answer lives in
> a document somebody emailed — an invoice PDF, a spreadsheet of figures, a
> signed contract, a scanned letter — body search finds the covering email and
> then looks as though the numbers are not there. They are; they are in the file.
> The route is: `search_email` / `grep_email` to find the thread → this tool for
> its `sha256`s → `get_attachment` to read one. A search hit's
> `attachment_names` is the hint that a document exists.

- **Args:**
  - `thread_id` (str, optional) — thread whose attachments to list.
  - `message_id` (str, optional) — message whose attachments to list.
  - **At least one** of `thread_id` / `message_id` is required.
  - `collection` (str, optional) — accepted for API symmetry; the attachment store
    is corpus-wide, so it is not required to select a store.
- **Returns:** a row per attachment
  `{sha256, filename, mime, size, thread_id, message_id, inline}`.

> **Decoration is filtered out by default** — signature blocks, legal
> disclaimers rendered as images, newsletter headers and spacer pixels, which
> are 61% of attachment rows on a real corpus. Pass `include_boilerplate=true`
> for the unfiltered list; the store keeps every row, only this tool takes the
> opinion.
>
> The verdict prefers **measured OCR signals** and falls back to a metadata
> heuristic for blobs not yet measured (see
> [Noise signals](#noise-signals-attachments-build)). Signals win because the
> heuristic is a guess about content made from metadata, and it misfires both
> ways — it hid a quarterly reporting-deadline table (small, quoted into 15
> threads because it is a useful reference) while a text-poor "access denied"
> screenshot is content someone pasted deliberately.

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

Read the **extracted text** of one attachment — PDF, spreadsheet, doc, scan
(parity with the CLI `./mailrag attachments get --text`). The only way to see
inside an emailed document: it extracts (or serves the cached) text, running OCR
when the file is a scan or an image. Raw bytes are **never** returned over MCP.

- **Args:**
  - `sha256` (str, required) — content hash of the attachment (from `list_attachments`).
  - `ocr` (str, optional) — extraction backend: `llm` | `tesseract` | `cloud`
    (like the CLI `--extractor` flag). Defaults to `$RAG_ATTACH_EXTRACTOR` or `llm`.
- **Returns:** `{sha256, filename, mime, size, text, text_status}`. Always check
  `text_status` (e.g. `ok`, `ocr_unavailable`): it reports when extraction failed,
  so an empty `text` is **not** evidence that the document is blank.
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
   your most recent `./mailrag onboard` built).

If none of those resolve, the tool returns a clear error
(`no email collection configured: set MAILRAG_COLLECTION or run 'mailrag onboard' first`)
rather than crashing. The same precedence drives the `is_default` flag returned by
`list_collections`.

## Configuration

Config mirrors `./mailrag ask` and is resolved from flags/environment:

| Setting | Resolution (highest precedence first) | Notes |
|---------|---------------------------------------|-------|
| Collection | `collection` arg → `$MAILRAG_COLLECTION` → latest onboarding manifest | Per-call `collection` overrides the env default. |
| Qdrant URL | `--qdrant-url` (CLI) → `$MAILRAG_QDRANT_URL` → `$QDRANT_URL` → `http://localhost:6333` | `MAILRAG_QDRANT_URL` lets a host server target `localhost` without inheriting a container-oriented `QDRANT_URL` from `.env` (the [issue #29](https://github.com/fmasi/mailrag/issues/29) gotcha). |
| Attachment store | `$RAG_ATTACH_STORE` → `~/.mailrag/attachments` | Same default and store the CLI `attachments` verbs use; corpus-wide. |
| Answer LLM endpoint | `$RAG_LLM_API_BASE` (alias `$RAG_LLM_BASE_URL`) → `http://localhost:1234/v1` | Used only by `answer_question`. See [`BACKENDS.md`](BACKENDS.md). |
| Answer LLM key | `$RAG_LLM_API_KEY` → `lm-studio` placeholder | **Set this in the MCP server config if your endpoint enforces auth** — the `lm-studio` placeholder is for auth-less local servers and is rejected with a 401 otherwise (issue #83). The startup healthcheck names it on failure. |
| Answer LLM model | `$RAG_LLM_MODEL` | Required for `answer_question`; the healthcheck names it when unset. |
| Raw corpus (grep) | `$MAILRAG_EML_ROOT` → `~/rag_eml` | The directory of raw `.eml` files `grep_email` scans. Must be the corpus you onboarded. |
| Usage log | `$MAILRAG_MCP_USAGE_LOG` → `~/.mailrag/mcp_usage.jsonl` | One JSON line per tool call. Set to `off` (or `""`/`0`/`none`) to disable. See [Usage logging](#usage-logging). |
| Usage log arguments | `$MAILRAG_MCP_USAGE_ARGS` → `values` | `values` logs truncated argument values; `names` logs only names and types. |
| Attachment OCR backend | `ocr` arg → `$RAG_ATTACH_EXTRACTOR` → `llm` | `$RAG_ATTACH_MAX_PAGES` bounds how many PDF pages OCR renders. |

> **`answer_question` startup healthcheck.** The first `answer_question` call
> verifies the LLM endpoint + key are reachable and authorized, and raises a
> clear `LLMHealthcheckError` (naming `RAG_LLM_API_KEY` / `RAG_LLM_API_BASE` /
> `RAG_LLM_MODEL`) if not — instead of leaking an opaque provider 401. The
> non-LLM tools (`search_email`, `grep_email`, `list_collections`,
> `list_attachments`, `get_attachment`) do **not** depend on the LLM and keep
> working when it is down.

## Noise signals (`attachments build`)

`./mailrag attachments build` measures each blob after ingesting it and records
the signals used to judge decoration. Roughly 4 minutes for a 45k-row corpus,
because measurement is keyed by **content hash** — the one 6.5 KB logo carried by
2,273 messages is a single blob — and bounded to the cheap tier.

```bash
./mailrag attachments build --profile ~/corpus.profile.json
# attachments: {'emails': 31969, 'attachments': 47956, 'skipped': 0}
# classified:  {'measured': 2570, 'skipped': 0, 'failed': 0}

./mailrag attachments build --profile … --no-classify        # skip it
./mailrag attachments build --profile … --classify-max-size 50000
```

**Why bulk, when extraction is otherwise lazy.** `get_attachment` runs OCR on
first fetch and caches it, which is right for real documents — a large PDF takes
minutes. But filtering happens when attachments are *listed*, while extraction
happens when one is *fetched*, so the blobs nobody has fetched are exactly the
ones polluting listings; lazy measurement never reaches them. The saving grace is
that decoration is small: ~0.09s per blob with tesseract. So the cheap tier is
measured in bulk and the expensive tier stays lazy.

**Engine choice.** The pass defaults to `tesseract`, not the global `llm`
default — it exists to answer "does this image contain words", and measured on
the same images the LLM path is 16–20× slower (8.5s vs 0.4s on a table). Override
with `--extractor` or `$RAG_ATTACH_CLASSIFY_EXTRACTOR`. LLM output is normalised
first: that provider prefixes a `DESCRIPTION:` preamble, which measured raw
inflates a three-word newsletter header from 22 to 159 characters — across the
text-rich threshold — so only the transcription is counted, and the two engines
produce identical signals for the same image.

**Signals, not verdicts.** The store records `chars`, `words`, `unique_words`,
`digits`, dimensions, status and extractor per blob; the verdict is computed from
them at read time. The measurement is corpus-agnostic ("a logo has no text, a
table does"); the thresholds are not — work mail is 41% bulk against personal
mail's 7%. Keeping them apart means re-calibrating for another corpus is a SQL
query, not a re-OCR of thousands of blobs.

The rule, calibrated against images inspected by eye:

| Condition | Verdict |
|---|---|
| Inline, in ≥20 threads | decoration — a disclaimer image sat in 829 threads, a signature block in 61 |
| ≥100 chars of text | content — rescues tables the metadata heuristic hid |
| <30 chars **and** in ≥5 threads | decoration |
| anything else | no opinion → metadata heuristic decides |

Known limit: the 15–25 thread band is ambiguous on recurrence alone — a real
reporting-deadline table sits at 15 threads, a disclaimer at 18. The cut errs
toward keeping, so some signature images survive rather than one real table being
hidden. Splitting that band properly needs a content rule (disclaimer phrasing,
contact-detail patterns), not a bigger threshold.

## Usage logging

Every tool call appends one JSON line to `~/.mailrag/mcp_usage.jsonl` (override
with `$MAILRAG_MCP_USAGE_LOG`, disable with `off`):

```jsonc
{"ts": "2026-08-19T08:22:21+00:00", "tool": "grep_email",
 "args": {"pattern": "support@…", "max_matches": 50, "regex": true, "max_seconds": 10},
 "duration_ms": 10088.1, "ok": true, "result_count": 0,
 "complete": false, "scanned": 4138, "stop_reason": "deadline"}
```

Arguments left unset are omitted, so the log shows what callers actually *chose*
to pass — which is the signal for whether a parameter earns its place in the tool
schema. Values are truncated to 200 characters; set `MAILRAG_MCP_USAGE_ARGS=names`
to log only names and types when even a search query is too sensitive to keep on
disk. The log lives outside the repo because it records your own queries against
your own mail.

What it is for: a tool nobody calls is usually a badly *described* tool rather
than an unwanted one, and the log is the only way to tell the difference — along
with which calls are slow enough to be abandoned, which fail, and which arguments
are dead weight. Logging failures are swallowed: a broken log must never break a
search.

```bash
# what gets used
jq -r .tool ~/.mailrag/mcp_usage.jsonl | sort | uniq -c | sort -rn
# slowest calls
jq -r 'select(.duration_ms > 5000) | "\(.duration_ms)ms \(.tool) \(.args)"' ~/.mailrag/mcp_usage.jsonl
# greps that ran out of budget rather than finding nothing
jq -r 'select(.tool=="grep_email" and .complete==false)' ~/.mailrag/mcp_usage.jsonl
```

## Running it

The server needs an already-indexed Qdrant collection — build one first with
`./mailrag onboard` (see [`QUICKSTART.md`](QUICKSTART.md)). Then launch either way:

```bash
# via the CLI verb (recommended — resolves its own interpreter, and loads .env)
./mailrag mcp
./mailrag mcp --collection work-rag --qdrant-url http://localhost:6333

# or as a module — NOT equivalent: this path does not load .env, so every
# setting it would have read (model, API key, corpus root) must be in the
# environment already
python -m src.mcp_server
```

The shim needs no activated environment; the module path must run in the
`mailrag` conda env. Set `HF_HUB_OFFLINE=1` once the bge-m3
weights are cached so `search_email` / `answer_question` embed the query from
cache without contacting the Hub (see [`SETUP.md § 2`](SETUP.md#2-the-mailrag-environment)).
`grep_email` uses no embeddings at all, so it works even before the weights are
cached. The process speaks MCP over stdio and blocks until the client disconnects.

## Registering with an MCP client

**Launch clients through the repo-root `./mailrag` shim.** It is self-locating:
it resolves the repo, the interpreter (`$MAILRAG_PYTHON` → a repo `.venv` → a
conda env named `mailrag` → `python`) and `PYTHONPATH` from its own path, so it
works with no activated environment, no `PYTHONPATH`, and whatever working
directory the client happens to use. There is no *installed* console command
(Poetry stays `package-mode = false`), so this shim is the stable entry point.

**Pick a scope first.** Client config comes in two scopes and they are not
interchangeable:

| | Reaches | Use when |
|---|---|---|
| **User scope** (`claude mcp add -s user`) | every session, any directory | You query your mail *from other projects* — the usual case, since the point of the server is reaching your mail while working on something else. |
| **Project scope** (`.mcp.json`, committed) | only sessions inside this repo | You work on mailrag itself, or you want a shareable zero-setup default for other people cloning it. |

User scope **overrides** project scope, so an entry in both means the user-scope
one runs everywhere and `.mcp.json` is never exercised. Point them at the same
command (below) and that costs you nothing — but be aware that agents outside
this repo see *only* the user-scope entry. Removing it silently cuts them off.

Launching via the shim (`./mailrag mcp`) also runs `load_dotenv()`, which the
bare `python -m src.mcp_server` module path does **not**. That is the difference
between one line of client config and duplicating your whole `.env` — model,
corpus root and API key included — into every client that wants the server.
Secret references (`keychain:` / `env:` / `file:`) resolve too, so no client
config need hold a plaintext key.

### Project scope (`.mcp.json`) — in-repo work and clean clones

The repo ships a project-scoped [`.mcp.json`](../.mcp.json), so any MCP client
that reads project config picks the server up with no per-machine setup and no
secrets in client config:

```jsonc
{
  "mcpServers": {
    "mailrag": {
      "type": "stdio",
      "command": "${CLAUDE_PROJECT_DIR:-.}/mailrag",
      "args": ["mcp"],
      "env": {
        "MAILRAG_QDRANT_URL": "${MAILRAG_QDRANT_URL:-http://localhost:6333}",
        "MAILRAG_EML_ROOT": "${MAILRAG_EML_ROOT:-~/rag_eml}",
        "RAG_LLM_API_BASE": "${RAG_LLM_API_BASE:-http://localhost:1234/v1}"
      }
    }
  }
}
```

Only three settings appear here, and each is an env lookup with a default you can
override from your shell. Everything else — collection, LLM model, API key —
comes from `.env` via the shim, which is why there is nothing to keep in sync.

The two `localhost` defaults are deliberate: a `.env` written for the container
path points `QDRANT_URL` / `RAG_LLM_API_BASE` at `host.docker.internal`, which a
host-side server cannot reach. `MAILRAG_QDRANT_URL` exists precisely to win that
fight without editing `.env` (the [issue #29](https://github.com/fmasi/mailrag/issues/29)
gotcha).

Claude Code asks you to approve a project-scoped server the first time it sees
it. **Approve it through the `/mcp` prompt, not by editing `~/.claude.json`** —
a running client rewrites that file from its own in-memory state, so hand-edited
approvals are silently discarded. The same applies to adding servers: use
`claude mcp add` rather than editing the file under a live session.

Remember this file is invisible to sessions outside the repo. If your agents
query mail from elsewhere, you need the user-scope entry below as well.

### Claude Code

Working *inside* the repo you need nothing — `.mcp.json` is picked up on
approval. For everywhere else, which is where a mail server usually earns its
keep, add it at user scope:

```bash
claude mcp add mailrag \
  --env MAILRAG_QDRANT_URL=http://localhost:6333 \
  --env MAILRAG_EML_ROOT=/Users/you/rag_eml \
  --env RAG_LLM_API_BASE=http://localhost:1234/v1 \
  -- /Users/you/Git/mailrag/mailrag mcp
```

Verify from a directory outside the repo — that is the case project scope cannot
cover:

```bash
cd ~ && claude mcp list
# mailrag: /Users/you/Git/mailrag/mailrag mcp - ✔ Connected
```

Everything after `--` is the launch command; an absolute path to the shim works
from any directory. `list_collections` lets the agent discover corpora even with
`MAILRAG_COLLECTION` unset. If your LLM endpoint enforces auth, set
`RAG_LLM_API_KEY` in `.env` (a `keychain:`/`env:`/`file:` reference keeps it out
of plaintext) — `answer_question` runs a startup healthcheck and fails loudly
naming that var rather than leaking a raw 401 (issue #83).

> **A running client pins the code it started with.** The shim always launches
> the current working tree, but a client that is already connected keeps the
> server process — and the tool schemas — it loaded at startup. After changing
> server code, reconnect (`/mcp` in Claude Code) or restart the client, or you
> will be testing yesterday's build.

### opencode

Add a `local` MCP entry to `opencode.jsonc`:

```jsonc
{
  "mcp": {
    "mailrag": {
      "type": "local",
      "command": [
        "/Users/you/Git/mailrag/mailrag",
        "mcp"
      ],
      "environment": {
        "MAILRAG_QDRANT_URL": "http://localhost:6333",
        "MAILRAG_EML_ROOT": "/Users/you/rag_eml",
        "RAG_LLM_API_BASE": "http://localhost:1234/v1"
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
      "command": "/Users/you/Git/mailrag/mailrag",
      "args": ["mcp"],
      "env": {
        "MAILRAG_QDRANT_URL": "http://localhost:6333",
        "MAILRAG_EML_ROOT": "/Users/you/rag_eml",
        "RAG_LLM_API_BASE": "http://localhost:1234/v1"
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
| Search threads (bounded) | `ask` / `query` (retrieval) | `search_email` | Pure query; snippet + metadata, the agent's raw-material path. |
| Fetch a full thread | *(implicit)* | `get_thread` | Full-body opt-in companion to the bounded `search_email`. |
| Literal / regex needle hunt | *(shell `grep` over the raw `.eml` tree)* | `grep_email` | No embeddings; exact match over the raw corpus (numbers, IDs, error strings). |
| Grounded answer | `ask` / `query` | `answer_question` | Pure query + one LLM call (with a startup healthcheck); the agent's answer path. |
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
(e.g. `conda run -n mailrag ./mailrag onboard …`); it then becomes queryable through
the MCP tools above.

## Advanced: driving retrieval from your own code

The MCP tools wrap the same in-process API the CLI uses, so an agent host (or any
Python you write) can skip the stdio layer and call retrieval directly. Run this in
the `mailrag` env, from the repo root, with the bge-m3 weights cached
(`HF_HUB_OFFLINE=1`):

```python
from dotenv import load_dotenv; load_dotenv()
from src.query.hybrid import build_hybrid_searcher
from src.llm.answer import answer_from_threads

searcher = build_hybrid_searcher(
    "work-rag",
    mode="hybrid",     # "hybrid" (best) | "dense" | "sparse"
    rerank=False,      # True = cross-encoder precision boost (slower, +~2 GB RAM)
    dense_top_k=20, sparse_top_k=20, top_n=5,
)

# Thread-aware retrieval — ThreadContext objects (whole threads, not fragments).
contexts = searcher.search_threads("What did we decide about the Q3 budget?")
for c in contexts[:5]:
    print(c.subject)      # thread subject
    print(c.text[:500])   # rendered thread (headers + bodies), ready for an LLM

# Optional grounded answer over the top-k threads (the answer_question path).
print(answer_from_threads("What did we decide about the Q3 budget?", contexts, k=3))
```

`searcher.search(query)` (no expansion) returns ranked `NodeWithScore` chunks if you
want raw passages instead of whole threads. `search_threads` + `answer_from_threads`
are exactly what `search_email` and `answer_question` call under the hood. See
[`RETRIEVAL_GUIDE.md` → How to call it](RETRIEVAL_GUIDE.md#how-to-call-it) for the full
`HybridSearcher` API and [`BACKENDS.md`](BACKENDS.md) for the answer LLM configuration.
