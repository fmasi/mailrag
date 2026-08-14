# Documentation index

The map of `mailrag`'s docs. Each doc has one job; read them in the order that
matches how deep you want to go. The root [`README.md`](../README.md) mirrors this
map — this is the canonical reading order.

## Reader journey

1. **[`README.md`](../README.md)** *(repo root)* — **start here.** What `mailrag` is, the
   one-command quickstart (`make demo`), the architecture sketch, and the case study
   (cleanup economics + the measured retrieval ladder).
2. **[`GUIDE.md`](GUIDE.md)** — the friendly walkthrough: the cleanup funnel, how to pick a
   **persona** (budget vs quality), and what the full-screen `wizard` looks like when you run it.
3. **[`QUICKSTART.md`](QUICKSTART.md)** — the 5-minute demo (`make demo`) and a minimal
   `./mailrag ask` against a built collection.
4. **[`SETUP.md`](SETUP.md)** — full setup: the single `mailrag` conda env, Qdrant, configuration,
   the **local `.eml` pipeline** (persona → `wizard`/`run` → `index`), and how to run the tests.
5. Deep dives (read any, in any order):
   - **[`BACKENDS.md`](BACKENDS.md)** — pointing mailrag at the LLM / embedder / vector store of
     your choice (LM Studio, Ollama, vLLM, NVIDIA NIM, OpenAI, Qdrant): the `RAG_*` variables,
     per-backend examples, and the dense-only "sparse caveat".
   - **[`VERBS.md`](VERBS.md)** — the CLI source of truth: every verb (including `ask` and `mcp`),
     the cost-ordered ladder, the alias table, and the persona recipes.
   - **[`SYNC.md`](SYNC.md)** — keeping a collection live: `accounts.yaml`, secret references
     (`keychain:` / `env:` / `file:`), folder **roles**, stage-skipping when a backend is down,
     launchd & systemd scheduling, and the `MessageSource` seam for adding a provider.
   - **[`MCP_SERVER.md`](MCP_SERVER.md)** — the multi-collection stdio MCP server: the seven tools
     (`list_collections` / `search_email` / `get_thread` / `grep_email` / `answer_question` /
     `list_attachments` / `get_attachment`), collection discovery, client setup, and the
     CLI↔MCP capability matrix.
   - **[`ARCHITECTURE.md`](ARCHITECTURE.md)** — design decisions and extension points.
   - **[`EMAIL_PREPROCESSING.md`](EMAIL_PREPROCESSING.md)** — reply-chain stripping, posting
     styles, and chunk-size tuning.
   - **[`CHUNKING.md`](CHUNKING.md)** — the two chunk kinds (body-with-summary vs summary-free
     attachment) and how `thread_id` stitches them back together at query time.
   - **[`RETRIEVAL_GUIDE.md`](RETRIEVAL_GUIDE.md)** — the retrieval stack end-to-end: dense vs
     learned-sparse, hybrid + RRF, reranking, and **thread-aware retrieval** (small→big expansion).
   - **[`EXPERIMENTS.md`](EXPERIMENTS.md)** — the measured findings: the cleanup funnel, the
     labeled-eval ladder (§9–§13), the corpus-portability result (§14), confound controls,
     and the negative results. Its
     **[terminology box](EXPERIMENTS.md#terminology-read-this-first)** defines the `C`/`C′`
     collection labels and the two senses of "thread-aware".
   - **[`CASE_STUDY.md`](CASE_STUDY.md)** — what each cleanup and retrieval choice actually bought
     on a real ~32k-email mailbox: the technique ladder, the cost/benefit of each stage, and the
     two results that went against expectation (reranking, and a +6pp win that was half a
     quantization artifact).
   - **[`CLAIMS.md`](CLAIMS.md)** — the claims register: every published number, the script that
     produces it, the corpus it came from, and when it was last verified. Start here if you want
     to know whether a figure is publicly reproducible or author-reported — and which two are
     currently unverifiable for want of an API key.
   - **[`BENCHMARK.md`](BENCHMARK.md)** — `make bench`: the *public* retrieval number you can
     regenerate yourself on Enron-QA, with no key and no private data. Covers what it measures,
     what it deliberately omits (thread reconstruction, rerank — neither reproducible on public
     data), and why the corpus is sized the way it is.

## Operations & reference

- **[`ARCHITECTURE_DIAGRAMS.py`](ARCHITECTURE_DIAGRAMS.py)** — runnable script that prints the
  data-lifecycle and query-flow diagrams as terminal ASCII, annotated with the module
  behind each stage.
- **[`POETRY_MIGRATION.md`](POETRY_MIGRATION.md)** — Poetry dependency-management notes (the
  conda + `requirements.txt` install path in [`SETUP.md`](SETUP.md) is the supported one).

## Live entry points (at a glance)

| You want to… | Use |
|--------------|-----|
| Run the public demo | `make demo` → [`scripts/quickstart.sh`](../scripts/quickstart.sh) → [`scripts/demo.py`](../scripts/demo.py) |
| Ask a question from the CLI | `./mailrag ask "…" --collection <name>` |
| Serve a collection to an agent | `./mailrag mcp` (see [`MCP_SERVER.md`](MCP_SERVER.md)) |
| Keep a collection fresh | `./mailrag sync` (see [`SYNC.md`](SYNC.md)) |
| Build a contextual index from loaded emails | `build_contextual_index(...)` in `src/indexing/contextual_index.py` |
| Query a collection from code | `build_hybrid_searcher(collection).search()` / `.search_threads()` in `src/query/hybrid.py` |
| Load emails from a source | `load_emails(source="enron" \| "mail_archive_x")` in `src/data/` |

> Older docs may mention `EmailIndexer` (`src/indexing/indexer.py`) and `EmailQueryEngine`
> (`src/query/engine.py`). Those classes have been **retired** — the live replacements are the
> rows above.
