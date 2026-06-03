# Using mailrag from code or another agent

How to drive mailrag programmatically (or from another AI agent) to query an
email corpus. For human setup, start with [`QUICKSTART.md`](QUICKSTART.md) and
[`SETUP.md`](SETUP.md); for retrieval design, see [`RETRIEVAL_GUIDE.md`](RETRIEVAL_GUIDE.md).

## TL;DR — ask it a question

```bash
cd <repo root>
# run in the conda env that has FlagEmbedding/torch (query embeds with bge-m3 locally)
./mailrag query "What did we decide about the Q3 budget?" --collection <NAME> --k 3
```

`mailrag query` does: hybrid retrieval → **thread expansion** (returns whole
threads, not fragments) → an LLM answer grounded only in those threads. The
repo-root `.env` is auto-loaded.

## Choosing a collection

A query targets one Qdrant collection. Pick one of:
- **Explicit:** `--collection <NAME>` (CLI) or the first arg to `build_hybrid_searcher`.
- **Latest built:** omit `--collection` and the CLI uses `latest_manifest_collection()`
  (reads `~/.mailrag/*.json`, newest by mtime).

Collections are built by `mailrag onboard` (see below) or `make demo`:
- **`mailrag-demo`** — the public Enron demo collection (`make demo` / `python main.py`).
- A **production collection** is a contextual, thread-aware build — e.g.
  `work-rag-ctx-threadaware` (thread-context summaries + thread-aware retrieval),
  which is the highest-quality configuration. Query whichever collection you built.

## Prerequisites

- **Qdrant** reachable at `QDRANT_URL` (default `http://localhost:6333`). Start it with
  `docker compose up -d qdrant`; check `curl -s http://localhost:6333/readyz`.
- **An embedding-capable env**: queries embed the query string with **bge-m3 via
  FlagEmbedding** (local, MPS/CPU), so run in the env that has `FlagEmbedding`/`torch`
  (the project's build env), not a lightweight test env.
- **An OpenAI-compatible LLM** for the *answer* step (LM Studio at
  `http://localhost:1234/v1` by default). Retrieval alone does not need it.

## Two ways to query

### 1. CLI
```bash
./mailrag query "YOUR QUESTION" --collection <NAME> --k 3
```

### 2. Programmatic (retrieval-only, or custom answering)
```python
# from the repo root, in the embedding-capable env
from dotenv import load_dotenv; load_dotenv()
from src.query.hybrid import build_hybrid_searcher
from src.llm.answer import answer_from_threads

searcher = build_hybrid_searcher(
    "<COLLECTION>",
    mode="hybrid",     # "hybrid" (best) | "dense" | "sparse"
    rerank=False,      # True = cross-encoder precision boost (slower, +~2GB RAM)
    dense_top_k=20, sparse_top_k=20, top_n=5,
)

# Thread-aware retrieval — returns ThreadContext objects (whole threads).
contexts = searcher.search_threads("What did we decide about the Q3 budget?")
for c in contexts[:5]:
    print(c.subject)      # thread subject
    print(c.text[:500])   # rendered thread (headers + bodies), ready for an LLM

# Optional grounded answer over the top-k threads.
print(answer_from_threads("What did we decide about the Q3 budget?", contexts, k=3))
```

`searcher.search(query)` (no expansion) returns ranked `NodeWithScore` passage
chunks if you want raw passages instead of whole threads.

## Retrieval knobs that matter

- **`mode="hybrid"`** (best): fuses dense (bge-m3 semantic) + sparse (bge-m3
  learned-lexical) via Reciprocal Rank Fusion. Beats dense-only or sparse-only.
- **`search_threads()` over `search()`**: match a small chunk, then return the
  *whole thread* — so terse replies ("Approved") arrive with their context. This
  is the signature technique; production collections are tuned for it.
- **`rerank=True`**: adds a `BAAI/bge-reranker-v2-m3` cross-encoder pass — a clear
  win on content/literal questions, at the cost of latency and ~2 GB RAM.
- **`top_n`** = final result/thread count; **`--k` / `answer_from_threads(k=)`** =
  how many threads the answer model reads.
- Ask **natural-language questions**, not keywords — semantic retrieval is the
  point. There is no special query syntax (metadata filtering is Qdrant-level).

## Config env vars

The **query + answer** path (`src/llm/client.py`) reads:
- `RAG_LLM_BASE_URL` (default `http://localhost:1234/v1`)
- `RAG_LLM_API_KEY`
- `RAG_LLM_MODEL` — the chat model id used for answers; **must be non-empty**, and
  that model must be loaded/available on the endpoint, or the answer step fails.
- `QDRANT_URL` (default `http://localhost:6333`), `QDRANT_API_KEY` (cloud only).

Note: `src/config/settings.py` (`RAGConfig`) uses a *different* set
(`RAG_LLM_API_BASE`, `RAG_EMBEDDING_*`) for the LlamaIndex indexing path — those
do not affect querying.

## Building / refreshing a collection (optional)

```bash
./mailrag onboard /path/to/eml-dir --collection my-emails --chunk-size 512
```
or programmatically `src.indexing.contextual_index.build_contextual_index(...)`.
**Caution:** `onboard` and `build_contextual_index(recreate=True)` drop and
rebuild the named collection — don't run them against a collection you want to keep.

## Gotchas

- Run from the **repo root** (the `./mailrag` shim runs `python -m src.cli`, which
  needs `src/` on the path).
- Use the **embedding-capable env** — a test-only env without `FlagEmbedding` can't
  embed queries.
- If another process is using the LLM endpoint, **retrieve only** (use
  `search_threads` and read `c.text`) to avoid triggering a model swap.

## Verify

```bash
cd <repo root>
./mailrag query "budget approval" --collection <NAME> --k 3
# or retrieval-only:
python -c "
from dotenv import load_dotenv; load_dotenv()
from src.query.hybrid import build_hybrid_searcher
s=build_hybrid_searcher('<NAME>', mode='hybrid')
ctx=s.search_threads('budget approval')
print('retrieved threads:', len(ctx)); print(ctx[0].subject if ctx else 'none')
"
```

## Future directions seeded by this doc

- **MCP connector**: expose `search_threads` (and a grounded-answer tool) over MCP
  so any MCP client can query the corpus — this guide is the contract for that tool.
- **Streamlining**: the env/collection/knob surface here is the checklist for
  simplifying configuration and the query entry points.
