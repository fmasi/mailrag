# Documentation index

The map of `mailrag`'s docs. Each one has a single job. Read them in the order that
matches how deep you want to go, and stop whenever you have what you came for. This
page is the canonical reading order, and the root [`README.md`](../README.md) points
here rather than repeating it.

## Reader journey

1. **[`README.md`](../README.md)** *(repo root)*. **Start here.** What `mailrag` is, the
   two public commands (`make demo` and `make bench`), the architecture sketch and the
   headline numbers.
2. **[`WHY_LOCAL.md`](WHY_LOCAL.md)**. Why this runs on your own hardware, why no
   consented email corpus exists, and what running locally actually costs you.
3. **[`GUIDE.md`](GUIDE.md)**. The friendly walkthrough: the cleanup funnel, how to pick a
   **persona** (budget against quality), and what the full-screen `wizard` looks like in use.
4. **[`QUICKSTART.md`](QUICKSTART.md)**. The 5-minute demo and a minimal `./mailrag ask`
   against a built collection.
5. **[`SETUP.md`](SETUP.md)**. Full setup: the single `mailrag` conda env, Qdrant,
   configuration, the local `.eml` pipeline (persona → `wizard`/`run` → `index`), and how to
   run the tests.
6. Deep dives, in any order:
   - **[`BACKENDS.md`](BACKENDS.md)**. Pointing mailrag at the LLM, embedder or vector store
     of your choice (LM Studio, Ollama, vLLM, NVIDIA NIM, OpenAI, Qdrant): the `RAG_*`
     variables, per-backend examples, and the dense-only sparse caveat.
   - **[`VERBS.md`](VERBS.md)**. The CLI source of truth. Every verb including `ask` and
     `mcp`, the cost-ordered ladder, the alias table, and the persona recipes.
   - **[`SYNC.md`](SYNC.md)**. Keeping a collection live: `accounts.yaml`, secret references
     (`keychain:` / `env:` / `file:`), folder **roles**, stage-skipping when a backend is
     down, launchd and systemd scheduling, and the `MessageSource` seam for adding a provider.
   - **[`MCP_SERVER.md`](MCP_SERVER.md)**. The multi-collection stdio MCP server. The seven
     tools (`list_collections` / `search_email` / `get_thread` / `grep_email` /
     `answer_question` / `list_attachments` / `get_attachment`), collection discovery, client
     setup, and the CLI-to-MCP capability matrix.
   - **[`ARCHITECTURE.md`](ARCHITECTURE.md)**. Design decisions, the module map, and
     extension points.
   - **[`EMAIL_PREPROCESSING.md`](EMAIL_PREPROCESSING.md)**. Reply-chain stripping, posting
     styles, and chunk-size tuning.
   - **[`CHUNKING.md`](CHUNKING.md)**. The two chunk kinds (body-with-summary against
     summary-free attachment) and how `thread_id` stitches them back together at query time.
   - **[`RETRIEVAL_GUIDE.md`](RETRIEVAL_GUIDE.md)**. The retrieval stack end to end: dense
     against learned-sparse, hybrid and RRF, reranking, why the answering message is the
     hardest one to retrieve, and **thread-aware retrieval** (small to big expansion).

## Evidence and claims

These four answer "should I believe any of this?", which is a different question from "how
does it work?".

- **[`CLAIMS.md`](CLAIMS.md)**. The claims register. Every published number, the script that
  produces it, the corpus it came from, and when it was last verified. Start here to find out
  whether a figure is publicly reproducible or author-reported, which two are currently
  unverifiable, and which one was withdrawn.
- **[`BENCHMARK.md`](BENCHMARK.md)**. The two public commands. `make bench` regenerates the
  retrieval number on Enron-QA with no key and no private data, `make demo` measures
  contextual embedding and thread reconstruction. Covers what each one deliberately omits.
- **[`CASE_STUDY.md`](CASE_STUDY.md)**. What each cleanup and retrieval choice bought on a
  real ~32k-email mailbox: the technique ladder, the cost and benefit of each stage, and the
  two results that went against expectation.
- **[`EXPERIMENTS.md`](EXPERIMENTS.md)**. The full log behind all of it. The cleanup funnel,
  the labeled-eval ladder (§9–§13), the corpus-portability result (§14), confound controls,
  and the negative results. Its
  **[terminology box](EXPERIMENTS.md#terminology-read-this-first)** defines the `C`/`C′`
  collection labels and the two senses of "thread-aware", which are easy to confuse.

A shorter, reader-facing version of the same material is published as
[`methods.html`](methods.html), which serves at
[fmasi.eu/mailrag/methods.html](https://fmasi.eu/mailrag/methods.html).

## Operations and reference

- **[`CI.md`](CI.md)**. The gates every PR runs, which two block a merge, how to run each
  locally, and the supply-chain pins with the reason each one exists.
- **[`ARCHITECTURE_DIAGRAMS.py`](ARCHITECTURE_DIAGRAMS.py)**. A runnable script that prints
  the data-lifecycle and query-flow diagrams as terminal ASCII, annotated with the module
  behind each stage.
- **[`ROADMAP.md`](ROADMAP.md)**. What ships in v1.0.0 and why each item gates it, plus what
  is deliberately deferred. Mirrored by the `v1.0.0` GitHub milestone.
- **[`RELEASE_NOTES.md`](RELEASE_NOTES.md)**. What changed between tagged releases.
- **[`POETRY_MIGRATION.md`](POETRY_MIGRATION.md)**. Poetry dependency-management notes. The
  conda plus `requirements.txt` install path in [`SETUP.md`](SETUP.md) is the supported one.

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
> (`src/query/engine.py`). Both classes are **retired**. The live replacements are the rows
> above.
