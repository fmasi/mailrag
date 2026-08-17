# Quick start

*[← docs index](INDEX.md) · [README](../README.md) · full setup in [`SETUP.md`](SETUP.md)*

The fastest way to see `mailrag` work is `make demo`: two indexes over 1,200 public Enron
emails, the same questions asked of both, and a measured difference. It needs no API key and
touches no private data. For the full local `.eml` pipeline, the `mailrag` conda env and the
test suite, go to [`SETUP.md`](SETUP.md).

## The 5-minute demo

From a fresh clone:

```bash
pip install -r requirements.txt   # includes FlagEmbedding (bge-m3); first run downloads ~2 GB of weights
make demo                         # starts Qdrant, builds both indexes, scores them
```

No `.env` and no key. You need those only for `./mailrag ask` and for indexing your own mail,
which come later on this page.

> Keep dependencies in a conda env or a container rather than on the host. See
> [`SETUP.md`](SETUP.md) for the `mailrag` env. `pyproject.toml` and `poetry.lock` pin exact
> versions, and `requirements.txt` is the supported install path.

`make demo` runs [`scripts/quickstart.sh`](../scripts/quickstart.sh), which brings up Qdrant in
Docker and then runs `python -m scripts.demo`. That builds **two** indexes over the same
corpus, one embedding each message as it stands and one embedding it alongside a summary of
what came before it in its conversation, then asks both the same questions.

Everything it needs is committed under [`eval/demo/`](../eval/demo/): the 1,200-message corpus,
1,193 pre-computed summaries, 99 validated single-message questions and 73 spanning ones. It
runs offline and spends no LLM tokens. Regenerating those fixtures is documented in
[`BENCHMARK.md`](BENCHMARK.md).

Expect under two minutes on an Apple-silicon GPU and roughly 15 on CPU.

### What you get out of it

Two numbers, both with a paired significance test, because both arms answer identical queries.

**Findability**, on the 99 single-message questions: recall@5 goes from 60.6% on the plain
index to 73.7% with thread context, fixing 16 questions and breaking 3 (McNemar exact
p = 0.0044).

**Completeness**, on the 73 spanning questions whose answers need several messages: the right
conversation is found 97.3% of the time at top-5, the top-5 *messages* give you 52.6% of that
conversation, and expanding to the thread gives you all of it.

### How it differs from `make bench`

The two commands answer different questions, and it is worth keeping them apart.

| | `make demo` | `make bench` |
|---|---|---|
| asks | does the technique work? | how good is the retrieval? |
| corpus | 1,200 public Enron emails | 2,000-document Enron-QA slice |
| measures | contextual embedding, thread reconstruction | hybrid dense plus learned-sparse |
| LLM calls | none | none |
| runtime (GPU / CPU) | ~2 min / ~15 min | 1.6 min / 14.7 min |

Neither reproduces the private 45.6 → 93.3 ladder, which was measured on a 32,000-email
archive and stays author-reported. [`BENCHMARK.md`](BENCHMARK.md) lists every omission for
both, and [`CLAIMS.md`](CLAIMS.md) records which figures are reproducible.

## Ask a question

Once a collection is indexed, query it from the repo root with the `./mailrag` shim. This step
does need a model, so set up [`BACKENDS.md`](BACKENDS.md) first:

```bash
cp .env.example .env
# once the ~2 GB of bge-m3 weights are cached, HF_HUB_OFFLINE=1 skips the Hub round-trip
HF_HUB_OFFLINE=1 ./mailrag ask "who approved the Q3 budget, and when?" --collection mailrag-demo --k 3
```

`ask` runs hybrid retrieval, expands each hit into its whole thread rather than a fragment,
and writes a grounded answer over the top-`k` threads. Omit `--collection` to use the latest
build. To hand the same collection to any agent over the Model Context Protocol, see
[`MCP_SERVER.md`](MCP_SERVER.md).

## Configuration essentials

Copy `.env.example` to `.env` and set the backend you want. The all-local default is LM Studio,
or any OpenAI-compatible server, for answers, local bge-m3 for embeddings, and local Qdrant:

```bash
RAG_LLM_PROVIDER=lmstudio
RAG_LLM_API_BASE=http://localhost:1234/v1
RAG_LLM_MODEL=<your-local-model>          # must be non-empty and loaded on the endpoint
QDRANT_URL=http://localhost:6333
```

The full variable reference (LM Studio, Ollama, vLLM, NVIDIA NIM, OpenAI) and the dense-only
sparse caveat are in [`BACKENDS.md`](BACKENDS.md).

## Query it from code

The live query API is `build_hybrid_searcher(...).search_threads(query)`:

```python
from dotenv import load_dotenv; load_dotenv()
from src.query.hybrid import build_hybrid_searcher

# point at a built collection (see `make demo` / SETUP.md)
searcher = build_hybrid_searcher("mailrag-demo", mode="hybrid")

# plain retrieval: ranked NodeWithScore chunks
nodes = searcher.search("meeting schedule")

# thread-aware retrieval: ThreadContext objects, match a unit and get its whole thread
contexts = searcher.search_threads("meeting schedule")
```

For the grounded-answer helper and the full `HybridSearcher` API, see
[`RETRIEVAL_GUIDE.md` → How to call it](RETRIEVAL_GUIDE.md#how-to-call-it). Historical notes
that mention `EmailIndexer`, `EmailQueryEngine` or `engine.query` are referring to retired
classes that no longer exist.

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `RAG_LLM_MODEL` empty, or the answer step fails | Set `RAG_LLM_MODEL` to a model actually loaded on the endpoint. |
| "QDRANT_URL environment variable is not set" | Set `QDRANT_URL`. Qdrant is required, and `docker compose up -d qdrant` starts the local one. |
| First run is slow | Expected. The first run downloads bge-m3 weights and embeds the corpus. Export `HF_HUB_OFFLINE=1` afterwards. |
| bge-m3 tries to reach the Hub while offline | The weights are not cached yet. Run once online, then export `HF_HUB_OFFLINE=1`. |
| Out of memory | Lower the sample count (`run_demo(num_samples=…)`) or the chunk size. |

## Next steps

1. Run `make demo`, then edit the questions in [`eval/demo/questions.jsonl`](../eval/demo/questions.jsonl) and watch the numbers move.
2. Read [`WHY_LOCAL.md`](WHY_LOCAL.md) for why the project is built this way.
3. Read [`SETUP.md`](SETUP.md) to run the full local `.eml` pipeline over your own mailbox.
4. Read [`GUIDE.md`](GUIDE.md) and [`VERBS.md`](VERBS.md) for the persona flow and the CLI.
5. Read [`RETRIEVAL_GUIDE.md`](RETRIEVAL_GUIDE.md) and [`EXPERIMENTS.md`](EXPERIMENTS.md) for the retrieval stack and the measured findings.
