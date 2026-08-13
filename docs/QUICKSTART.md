# Quick Start Guide

*[← docs index](INDEX.md) · [README](../README.md) · full setup in [`SETUP.md`](SETUP.md)*

The fastest way to see `mailrag` work: build a thread-aware contextual index over 100
public Enron emails and query it. For the full local `.eml` pipeline, the single
`mailrag` conda env, and the test suite, see [`SETUP.md`](SETUP.md).

## The 5-minute demo

From a fresh clone:

```bash
pip install -r requirements.txt   # includes FlagEmbedding (bge-m3); first run downloads ~2 GB of weights
cp .env.example .env              # add an LLM key/endpoint for summaries + answers (see BACKENDS.md)
make demo                         # starts Qdrant, builds the contextual index, runs thread-aware queries
```

> Keep deps in a conda env or container, never the host — see [`SETUP.md`](SETUP.md)
> for the `mailrag` env. `pyproject.toml` / `poetry.lock` pin exact versions;
> `requirements.txt` is the supported install path.

`make demo` runs [`scripts/quickstart.sh`](../scripts/quickstart.sh): it brings up
Qdrant (Docker) and runs `python main.py`, which builds a **thread-aware contextual
index** over 100 Enron emails and answers example questions by retrieving and
assembling whole threads. This is the §13 stack from the
[case study](CASE_STUDY.md).

> **What the demo is and isn't.** It is a **walkthrough**, not a measurement: 100
> emails, no scoring, no gold answers. It shows you the shape of the pipeline —
> index, retrieve, expand to the thread, answer with citations — and it does spend
> LLM calls to do it.
>
> If you want a **number** rather than a walkthrough, run
> [`make bench`](BENCHMARK.md) instead: 2 000 documents, 360 committed queries,
> recall@k with confidence intervals and a paired significance test, and no LLM
> calls at all. Note that it scores the hybrid *retrieval layer* only — the
> thread-reconstruction and summary levers are not reproducible on public data, and
> `docs/BENCHMARK.md` says exactly what is left out.

## Ask a question

Once a collection is indexed, query it from the repo root with the `./mailrag` shim:

```bash
# once the ~2 GB of bge-m3 weights are cached, HF_HUB_OFFLINE=1 skips the Hub round-trip
HF_HUB_OFFLINE=1 ./mailrag ask "who approved the Q3 budget, and when?" --collection mailrag-demo --k 3
```

`ask` runs hybrid retrieval → **thread expansion** (whole threads, not fragments) →
a grounded LLM answer over the top-`k` threads. Omit `--collection` to use the latest
build. To expose the same collection to any agent over the Model Context Protocol,
see [`MCP_SERVER.md`](MCP_SERVER.md).

## Configuration essentials

Copy `.env.example` to `.env` and set the backend you want. The all-local default is
LM Studio (or any OpenAI-compatible server) for answers, local bge-m3 for embeddings,
and local Qdrant:

```bash
RAG_LLM_PROVIDER=lmstudio
RAG_LLM_API_BASE=http://localhost:1234/v1
RAG_LLM_MODEL=<your-local-model>          # must be non-empty and loaded on the endpoint
QDRANT_URL=http://localhost:6333
```

Full variable reference (LM Studio / Ollama / vLLM / NVIDIA NIM / OpenAI, and the
dense-only "sparse caveat") is in [`BACKENDS.md`](BACKENDS.md).

## Query it from code (retrieval-only)

The live query API is `build_hybrid_searcher(...).search_threads(query)`:

```python
from dotenv import load_dotenv; load_dotenv()
from src.query.hybrid import build_hybrid_searcher

# point at a built collection (see `make demo` / SETUP.md)
searcher = build_hybrid_searcher("mailrag-demo", mode="hybrid")

# plain retrieval — ranked NodeWithScore chunks
nodes = searcher.search("meeting schedule")

# thread-aware retrieval — ThreadContext objects (match a unit, get its whole thread)
contexts = searcher.search_threads("meeting schedule")
```

For the grounded-answer helper and the full `HybridSearcher` API, see
[`RETRIEVAL_GUIDE.md` → How to call it](RETRIEVAL_GUIDE.md#how-to-call-it). (The older
`EmailIndexer` / `EmailQueryEngine` / `engine.query` snippets in some historical notes
reference retired classes — they no longer exist.)

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `RAG_LLM_MODEL` empty / answer step fails | Set `RAG_LLM_MODEL` to a model actually loaded on the endpoint. |
| "QDRANT_URL environment variable is not set" | Set `QDRANT_URL` — Qdrant is required (`docker compose up -d qdrant` for the local one). |
| First run is slow | Normal — the first run downloads bge-m3 weights and embeds the corpus. Set `HF_HUB_OFFLINE=1` afterwards. |
| bge-m3 tries to reach the Hub offline | Weights aren't cached yet — run once online, then export `HF_HUB_OFFLINE=1`. |
| Out of memory | Lower the sample count (`run_demo(num_samples=…)`) or the chunk size. |

## Next steps

1. Run `make demo`, then modify the example queries in `main.py::run_demo`.
2. Read the root [`README.md`](../README.md) for the overview and case study.
3. Read [`SETUP.md`](SETUP.md) to run the full local `.eml` pipeline over your own mailbox.
4. Read [`GUIDE.md`](GUIDE.md) and [`VERBS.md`](VERBS.md) for the persona flow and the CLI.
5. Read [`RETRIEVAL_GUIDE.md`](RETRIEVAL_GUIDE.md) and [`EXPERIMENTS.md`](EXPERIMENTS.md)
   for the retrieval stack and the measured findings.
