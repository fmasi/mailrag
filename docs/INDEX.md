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
   - **[`MCP_SERVER.md`](MCP_SERVER.md)** — the multi-collection stdio MCP server: the five tools
     (`list_collections` / `search_email` / `answer_question` / `list_attachments` /
     `get_attachment`), collection discovery, client setup, and the CLI↔MCP capability matrix.
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

## Operations & reference

- **[`CLOUD_STORAGE_SETUP.md`](CLOUD_STORAGE_SETUP.md)** — Azure Blob Storage + Qdrant Cloud:
  batch indexing, cost estimates, validation, and reset.
- **[`ARCHITECTURE_DIAGRAMS.py`](ARCHITECTURE_DIAGRAMS.py)** — runnable script that prints the
  data-lifecycle and query-flow diagrams.
- **[`POETRY_MIGRATION.md`](POETRY_MIGRATION.md)** — Poetry dependency-management notes (the
  conda + `requirements.txt` install path in [`SETUP.md`](SETUP.md) is the supported one).

## Live entry points (at a glance)

| You want to… | Use |
|--------------|-----|
| Run the public demo | `make demo` → [`scripts/quickstart.sh`](../scripts/quickstart.sh) → `main.py::run_demo` |
| Ask a question from the CLI | `./mailrag ask "…" --collection <name>` |
| Serve a collection to an agent | `./mailrag mcp` (see [`MCP_SERVER.md`](MCP_SERVER.md)) |
| Build a contextual index from loaded emails | `build_contextual_index(...)` in `src/indexing/contextual_index.py` |
| Query a collection from code | `build_hybrid_searcher(collection).search()` / `.search_threads()` in `src/query/hybrid.py` |
| Load emails from a source | `load_emails(source="enron" \| "mail_archive_x" \| "azure_blob")` in `src/data/` |

> Older docs may mention `EmailIndexer` (`src/indexing/indexer.py`) and `EmailQueryEngine`
> (`src/query/engine.py`). Those classes have been **retired** — the live replacements are the
> rows above.
