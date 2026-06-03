# Documentation index

The map of `mailrag`'s docs. Each doc has one job; read them in the order that
matches how deep you want to go.

## Reader journey

1. **[`README.md`](../README.md)** *(repo root)* — **start here.** What `mailrag` is, the
   one-command quickstart (`make demo`), the architecture sketch, and the case study
   (cleanup economics + the measured retrieval ladder).
2. **[`QUICKSTART.md`](QUICKSTART.md)** — the 5-minute path and copy-paste usage patterns
   against a built collection.
3. **[`SETUP.md`](SETUP.md)** — full setup: the two conda environments, Qdrant, configuration,
   the **local `.eml` pipeline** (Pass-1 filter → LLM Pass-2 → build), and how to run the tests.
4. Deep dives (read any, in any order):
   - **[`ARCHITECTURE.md`](ARCHITECTURE.md)** — design decisions and extension points.
   - **[`EMAIL_PREPROCESSING.md`](EMAIL_PREPROCESSING.md)** — reply-chain stripping, posting
     styles, and chunk-size tuning.
   - **[`RETRIEVAL_GUIDE.md`](RETRIEVAL_GUIDE.md)** — the retrieval stack end-to-end: dense vs
     learned-sparse, hybrid + RRF, reranking, and **thread-aware retrieval** (small→big expansion).
   - **[`EXPERIMENTS.md`](EXPERIMENTS.md)** — the measured findings: the cleanup funnel, the
     labeled-eval ladder (§9–§13), confound controls, and the negative results. Its
     **[terminology box](EXPERIMENTS.md#terminology-read-this-first)** defines the `C`/`C′`
     collection labels and the two senses of "thread-aware".

## Operations & reference

- **[`AGENT_USAGE.md`](AGENT_USAGE.md)** — driving mailrag from code or another agent:
  the query API (`search` / `search_threads`), collection selection, retrieval knobs,
  the query-path env vars, and gotchas. Seeds a future MCP connector.
- **[`CLOUD_STORAGE_SETUP.md`](CLOUD_STORAGE_SETUP.md)** — Azure Blob Storage + Qdrant Cloud
  (Pinecone optional): batch indexing, cost estimates, validation, and reset.
- **[`POETRY_MIGRATION.md`](POETRY_MIGRATION.md)** — Poetry dependency-management notes.
- **[`ARCHITECTURE_DIAGRAMS.py`](ARCHITECTURE_DIAGRAMS.py)** — runnable script that prints the
  data-lifecycle and query-flow diagrams.

## Live entry points (at a glance)

| You want to… | Use |
|--------------|-----|
| Run the public demo | `make demo` → `main.py::run_demo` |
| Build a contextual index from loaded emails | `build_contextual_index(...)` in `src/indexing/contextual_index.py` |
| Query a collection | `build_hybrid_searcher(collection).search()` / `.search_threads()` in `src/query/hybrid.py` |
| Load emails from a source | `load_emails(source="enron" \| "mail_archive_x" \| "azure_blob")` in `src/data/` |

> Older docs may mention `EmailIndexer` (`src/indexing/indexer.py`) and `EmailQueryEngine`
> (`src/query/engine.py`). Those classes have been **retired** — the live replacements are the
> three rows above.
