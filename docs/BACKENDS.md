# Backends & providers

How to point `mailrag` at the LLM and embedder of your choice, local or cloud,
and at a local or managed Qdrant. The short version:

- **LLM — fully backend-agnostic.** One set of `RAG_LLM_*` variables points both
  the answer side and the cleanup pipeline at *any* OpenAI-compatible server:
  LM Studio, Ollama, vLLM, NVIDIA NIM, OpenAI, Perplexity.
- **Embedder — local-first by design.** The default retrieval stack uses a
  **local bge-m3 dense + learned-sparse hybrid** (no API, and the thing that wins
  on real email — see [`EXPERIMENTS.md`](EXPERIMENTS.md)). You *can* swap in a
  remote OpenAI-compatible embedder, with **one honest caveat**: it's dense-only
  (see [The sparse caveat](#the-sparse-caveat)).
- **Vector store — Qdrant** (local Docker or managed), behind a single client
  seam. It is the only backend; see [`ROADMAP.md`](ROADMAP.md) for why.

`mailrag` is local-first: the defaults run entirely on your machine, and every
cloud option is opt-in. Copy [`.env.example`](../.env.example) to `.env` and set
the variables below.

---

## 1. The LLM

The LLM is used in two places, and **both go through one client,
`src/llm/client.py`** (a LlamaIndex `OpenAILike`), so configuring it once points
the whole system at one server:

| Where | What it does |
|-------|--------------|
| **Answer side** (`src/llm/answer.py`) | generates grounded answers from retrieved threads — for both `./mailrag ask` and the MCP [`answer_question`](MCP_SERVER.md#answer_questionquery-collectionnone-k3) tool |
| **Cleanup pipeline** | `summarize`/`judge`, HyDE, vision OCR |

### Variables

| Variable | Meaning |
|----------|---------|
| `RAG_LLM_API_BASE` | The OpenAI-compatible base URL. Default `http://localhost:1234/v1` (LM Studio). Any server that speaks `/v1/chat/completions` works. |
| `RAG_LLM_API_KEY` | API key / token for the endpoint. Many local servers don't need one; a placeholder is sent so the client doesn't complain. Accepts a `keychain:` / `env:` / `file:` reference, so the token need not sit in `.env`. |
| `RAG_LLM_MODEL` | The model id as the endpoint names it. Required: there is no default. |

Every call runs at temperature `0.0`. There is no provider switch: the base URL
alone decides which server answers.

### Pick your backend

| Backend | `RAG_LLM_API_BASE` | `RAG_LLM_API_KEY` | Example `RAG_LLM_MODEL` |
|---------|--------------------|-------------------|--------------------------|
| **LM Studio** (local) | `http://localhost:1234/v1` ¹ | LM Studio API token (if auth on) | `mistralai/magistral-small-2509` |
| **Ollama** (local) | `http://localhost:11434/v1` | any non-empty string | `llama3.1` |
| **vLLM** (self-hosted) | `http://<host>:8000/v1` | your token | the served model id |
| **NVIDIA NIM** (hosted) | `https://integrate.api.nvidia.com/v1` | your NVIDIA key | `meta/llama-3.1-8b-instruct` |
| **OpenAI** | `https://api.openai.com/v1` | your OpenAI key | `gpt-4o-mini` |
| **Perplexity** | `https://api.perplexity.ai` | your Perplexity key | `sonar` |

¹ Inside a devcontainer, reach the host with `http://host.docker.internal:1234/v1`.

> **Vision / OCR.** The attachment OCR path (`chat_vision`) sends an inline image
> to the chat endpoint, so it needs a **multimodal** model loaded (e.g. a Gemma-3/4
> vision build in LM Studio, or `meta/llama-3.2-11b-vision-instruct` on NIM). Text
> models may return an error or silently ignore the image.

---

## 2. The embedder

`mailrag` has **two embedding paths**. Know which one you're configuring.

### a. Local bge-m3 hybrid — the default, recommended path

The headline contextual stack (`just demo`, `build_contextual_index`,
`build_hybrid_searcher`) embeds with **bge-m3 via FlagEmbedding, locally**:
**dense + learned-sparse** in one model. This is the differentiator — the
learned-sparse leg is what beats dense-only retrieval on real email (see
[`RETRIEVAL_GUIDE.md`](RETRIEVAL_GUIDE.md) and
[`EXPERIMENTS.md`](EXPERIMENTS.md)). It needs no API and no key; it loads from
the local model cache.

```python
from src.ingest.embedder import make_embedder
emb = make_embedder("bge-m3")   # dense + learned sparse, produces_sparse=True
```

> **`HF_HUB_OFFLINE=1` for cached weights.** The first run downloads a few GB of bge-m3
> weights into your Hugging Face cache; after that, set `HF_HUB_OFFLINE=1` so
> FlagEmbedding loads from cache instead of contacting the Hub on every build/query
> (faster, works offline). Applies anywhere bge-m3 embeds — `./mailrag ask`,
> `./mailrag index`, and the MCP server. See [`SETUP.md § 2`](SETUP.md#2-the-mailrag-environment).

### b. Remote OpenAI-compatible dense embedder (legacy scripts only)

The LlamaIndex `Settings.embed_model` path points at any OpenAI-compatible embedding
endpoint. Only two legacy scripts read it (`scripts/reset_qdrant_index.py` and
`scripts/smoke_lmstudio_embedding.py`). The CLI, `just demo` and the MCP server do
not:

| Variable | Meaning |
|----------|---------|
| `RAG_EMBEDDING_PROVIDER` | `openai` or `lmstudio` (= any OpenAI-compatible server). |
| `RAG_EMBEDDING_API_BASE` | embedding endpoint URL. |
| `RAG_EMBEDDING_MODEL` | embedding model id. |
| `RAG_EMBEDDING_API_KEY` | key (often empty for local servers). |

So an LM Studio / Ollama / vLLM / OpenAI embedding model works for those scripts,
**dense only** (see the caveat below).

### c. NVIDIA NIM dense + reranker — the "C" experiment

The NVIDIA-native pattern: a dense embedding NIM **plus a reranking NIM** in
place of bge-m3's sparse leg. Dense-only embedder, hosted on NVIDIA's free tier
(needs `NVIDIA_API_KEY`). Requires the optional `nvidia` extra
(`poetry install --extras nvidia`).

```python
emb = make_embedder("nvidia-e5")        # nvidia/nv-embedqa-e5-v5, 1024-d, dense-only
emb = make_embedder("nvidia-nemotron")  # nvidia/llama-nemotron-embed-1b-v2, 2048-d
```

### The sparse caveat

> The OpenAI `/v1/embeddings` standard **has no field for sparse weights** —
> confirmed across NIM, vLLM, TEI, Infinity, and Ollama. So **any embedder reached
> over an OpenAI-compatible API is dense-only by construction** and cannot carry
> bge-m3's learned-sparse hybrid. On real email, the local bge-m3 hybrid measurably
> beats dense + rerank ([`EXPERIMENTS.md`](EXPERIMENTS.md)). The takeaways:
>
> - Want the **best retrieval**? Keep **bge-m3 local** (path *a*). It's the default.
> - Must embed **remotely**? You get dense-only — pair it with a **reranker**
>   (NVIDIA's own pattern, path *c*) to recover some of the lost ranking quality.

---

## 3. The reranker (optional)

Reranking is an opt-in cross-encoder postprocessor on top of retrieval:

- **Local (default):** `BAAI/bge-reranker-v2-m3` via FlagEmbedding — no API.
- **Hosted:** `nvidia/rerank-qa-mistral-4b` via `make_nim_reranker()` (the only
  hosted NVIDIA reranker; on the `ai.api.nvidia.com` host, needs `NVIDIA_API_KEY`).

The reranker is injectable, so you can supply either to `build_hybrid_searcher`.

---

## 4. The vector store

| Variable | Meaning |
|----------|---------|
| `QDRANT_URL` | `http://localhost:6333` (the quickstart's local Docker Qdrant) or a managed Qdrant Cloud URL. |
| `QDRANT_API_KEY` | set for managed Qdrant. |
| `QDRANT_COLLECTION_NAME` | read only by legacy scripts. The CLI takes `--collection` or the profile's collection. |

The Qdrant connection is built in one place (`src/config/qdrant.py::get_qdrant_client`),
so swapping local ↔ cloud is a single URL change. There is no provider switch:
Qdrant is the only backend, because storing learned-sparse vectors alongside
dense ones as named vectors on one point is a Qdrant-specific facility.

---

## Worked `.env` examples

**All-local (recommended):** LM Studio LLM, local bge-m3 hybrid embeddings, local Qdrant.

```bash
RAG_LLM_API_BASE=http://localhost:1234/v1
RAG_LLM_MODEL=mistralai/magistral-small-2509
RAG_LLM_API_KEY=sk-lm-...            # only if LM Studio auth is on
# embeddings: bge-m3 is local — nothing to configure for the contextual stack
QDRANT_URL=http://localhost:6333
```

**Cloud LLM via NVIDIA NIM, embeddings still local bge-m3:**

```bash
RAG_LLM_API_BASE=https://integrate.api.nvidia.com/v1
RAG_LLM_MODEL=meta/llama-3.1-8b-instruct
RAG_LLM_API_KEY=nvapi-...             # your build.nvidia.com key
```

**Ollama LLM (local):**

```bash
RAG_LLM_API_BASE=http://localhost:11434/v1
RAG_LLM_MODEL=llama3.1
RAG_LLM_API_KEY=ollama               # any non-empty placeholder
```

---

## See also

- [`SETUP.md`](SETUP.md) — full environment setup and the local `.eml` pipeline.
- [`.env.example`](../.env.example) — every variable, annotated.
- [`RETRIEVAL_GUIDE.md`](RETRIEVAL_GUIDE.md) — dense vs learned-sparse, hybrid + RRF, reranking.
- [`EXPERIMENTS.md`](EXPERIMENTS.md) — the measured bge-m3-hybrid-vs-dense+rerank result on real email.
