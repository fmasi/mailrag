# Setup, running & testing

*[← docs index](INDEX.md) · [README](../README.md)*

How to stand up `mailrag` from scratch, run the full local pipeline, and run the test
suite — enough for someone else to reach the same point we're at.

For a 60-second taste, see [`QUICKSTART.md`](QUICKSTART.md) (Enron demo). This guide covers
the **full** path including the local `.eml` hybrid pipeline and the LLM Pass-2.

---

## 1. Prerequisites

| need | for |
|------|-----|
| **Python 3.12+** + **conda/miniconda** | environments (never install project deps on the host) |
| **Docker** | Qdrant vector store |
| **Apple Silicon (MPS)** *or* CUDA *or* CPU | embeddings (bge-m3 via FlagEmbedding) — MPS/CUDA strongly preferred |
| **LM Studio** (or any OpenAI-compatible local LLM) | optional, for the LLM Pass-2 summarize/judge |

> **Environment rule:** all Python deps go in a **conda env or container**, never the host.

---

## 2. Two environments

We use two conda envs because the test suite and the GPU build need different deps.

**a) Test env** (no torch/FlagEmbedding needed — fast):

```bash
conda create -y -n mailrag-test python=3.12
conda run -n mailrag-test pip install -r requirements.txt \
    rich azure-storage-blob llama-index-vector-stores-pinecone openai anthropic
conda run -n mailrag-test python -m pytest tests/ -q     # expect: all pass, ~1 skipped
# cleanup when done: conda env remove -y -n mailrag-test
```

(The 5 extras are pyproject deps not in `requirements.txt`. The project is poetry
**non-package mode**, so `pip install -e .` will fail — don't use it.)

**b) Build env** (GPU/MPS hybrid embeddings + local-LLM pipeline):

```bash
conda create -y -n mailrag-build python=3.13
conda run -n mailrag-build pip install \
    FlagEmbedding torch transformers llama-index-core qdrant-client openai
```

Run host-side scripts with `conda run -n mailrag-build --no-capture-output python <script>`.

- **Hybrid query + reranker (`rag` env only):** the framework-native query path
  (`scripts/compare_retrieval.py`, `src/query/`) needs two LlamaIndex packages in the
  `rag` env, which is otherwise built ad-hoc with only `llama-index-core` + `qdrant-client`
  (the build talks to Qdrant via `qdrant-client` directly):
  `conda run -n rag pip install "llama-index-vector-stores-qdrant>=0.10,<0.11" llama-index-postprocessor-flag-embedding-reranker`.
  The reranker pulls FlagEmbedding (already present). Both are deliberately NOT core
  dependencies — the unit suite mocks/patches them, so `mailrag-test` does not need them.

---

## 3. Start Qdrant

```bash
docker compose up -d        # serves http://localhost:6333
```

Vector data persists outside the repo (set `QDRANT_STORAGE_PATH` in `.env`).

---

## 4. Configuration

```bash
cp .env.example .env                                   # fill in keys/paths
cp config/noise_rules.template.yaml  config/noise_rules.yaml
cp config/whitelist_domains.template.yaml config/whitelist_domains.yaml
# optional: append the portable starter rules
cat config/community_blocklist.template.yaml >> config/noise_rules.yaml   # review first
```

`.env`, `config/noise_rules.yaml`, `config/whitelist_domains.yaml` are **gitignored** —
they hold keys and sender-domain names. Only the `*.template.yaml` files are tracked.

---

## 5. Quick demo (public data, no GPU/LLM)

```bash
python main.py        # builds an index over 100 Enron emails and runs example queries
```

---

## 6. Full local `.eml` pipeline

Export your mailbox to a folder of `.eml` files (non-iCloud path), then:

```bash
# 1) pick which folders to index -> writes a selection JSON
python scripts/select_local_eml.py --root ~/rag_eml          # interactive picker

# 2) (optional) LLM Pass-2: summarize + judge each email into a resumable cache
#    needs a local OpenAI-compatible LLM (LM Studio) + RAG_LLM_MODEL / .env
conda run -n mailrag-build --no-capture-output python scripts/llm_pass2.py run \
    --selection ~/rag_eml.selection.json --cache ~/rag_pass2/pass2.db \
    --model <your-local-model> | tee ~/pass2.log
#    backfill stable identifiers (re-export resilience), once:
conda run -n mailrag-build python scripts/backfill_pass2_identity.py \
    --cache ~/rag_pass2/pass2.db --selection ~/rag_eml.selection.json

# 3) turn the Pass-2 noise verdicts into a blacklist
python scripts/llm_pass2.py apply --cache ~/rag_pass2/pass2.db \
    --blacklist ~/rag_pass2/noise.blacklist --min-confidence 0.0   # --dry-run first

# 4) build the hybrid collection (Pass-1 filter + dedup + embed + upsert)
conda run -n mailrag-build --no-capture-output python scripts/build_local_eml_rag.py \
    --collection my-rag --recreate \
    --blacklist ~/rag_pass2/noise.blacklist \
    --summary-cache ~/rag_pass2/pass2.db \
    [--embed-summary]            # add for contextual retrieval (summaries in the vector)
    | tee ~/build.log
```

`--summary-cache` injects each email's summary into the payload (and, with
`--embed-summary`, into the embedded text). The build prints per-stage counts and a
per-batch upsert rate.

---

## 7. Running tests

```bash
conda run -n mailrag-test python -m pytest tests/ -q          # full suite
# stdlib-only modules can run on the host without the env, e.g.:
PYTHONPATH=. python3 tests/test_llm_cache.py
```

Two skips are by design: the LM-Studio smoke test (opt-in via `RUN_LMSTUDIO_SMOKE=1`) and
a test that loads a real `config/noise_rules.yaml` (skips when only the template is present).

---

## 8. What you end up with

A Qdrant collection (dense bge-m3 + learned sparse, payload-indexed, optional per-email
summaries) you can query with hybrid RRF retrieval. See
[`EXPERIMENTS.md`](EXPERIMENTS.md) for the measured effect of each cleanup/retrieval step.

---

## 9. Attachment extraction

`mailrag attachments build` extracts text from attachments so their content is
searchable alongside email bodies. Extraction is handled by a subsystem in
`src/attachments/extract/`: one handler per content type (plaintext, html, docx,
xlsx, pptx, pdf, image) feeding into a swappable OCR-provider backend.

### OCR / vision backend

For images and image-only / scanned PDFs the default backend is a **local
vision-LLM (gemma-4 via LM Studio)** — fully on-device, no data leaves the machine.
If the LLM is unavailable it falls back automatically to local **tesseract**. Cloud
is opt-in and not yet implemented.

The LLM returns a structured description + transcription. Scanned-PDF pages read by
any OCR backend (vision LLM or tesseract) are capped by `RAG_ATTACH_MAX_PAGES`
(default `10`); pages past the cap are never rendered, and truncation is logged.

### Config

| Env var | Default | Values |
|---------|---------|--------|
| `RAG_ATTACH_EXTRACTOR` | `llm` | `llm` · `tesseract` · `cloud` |
| `RAG_ATTACH_MAX_PAGES` | `10` | integer — max scanned-PDF pages rendered/read per attachment (any OCR backend) |

Both can be overridden per-call with `--extractor <name>` on
`mailrag attachments get` and `mailrag attachments build`. Use `--force` to
re-extract even when a cached result already exists.

### Optional Python dependencies

```bash
conda run -n <your-env> pip install \
    pypdf python-docx openpyxl python-pptx pillow pytesseract pdf2image
```

Plus system packages for OCR: **tesseract** and **poppler**
(e.g. `brew install tesseract poppler` on macOS).

Without these, extraction degrades gracefully: unsupported types are stored with
status `binary` or `ocr_unavailable`, and the raw attachment file is always
served regardless.
