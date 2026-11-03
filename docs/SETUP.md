# Setup, running & testing

*[← docs index](INDEX.md) · [README](../README.md)*

How to stand up `mailrag` from scratch, run the full local pipeline, and run the test
suite — enough for someone else to reach the same point we're at.

For a 60-second taste, see [`QUICKSTART.md`](QUICKSTART.md) (Enron demo). This guide covers
the **full** path including the local `.eml` hybrid pipeline and the LLM `summarize` pass.

---

## 1. Prerequisites

| need | for |
|------|-----|
| **Python 3.11+** + **conda/miniconda** | the `mailrag` env (never install project deps on the host) |
| **Docker** | Qdrant vector store |
| **[`just`](https://github.com/casey/just)** (`brew install just`) | the task runner: `just demo`, `just bench`, `just ci` |
| **Apple Silicon (MPS)** *or* CUDA *or* CPU | embeddings (bge-m3 via FlagEmbedding) — MPS/CUDA strongly preferred |
| **LM Studio** (or any OpenAI-compatible local LLM) | optional, for the LLM `summarize`/`judge` pass and answers |

> **Environment rule:** all Python deps go in a **conda env or container**, never the host.

---

## 2. The `mailrag` environment

Everything runs in **one** conda env named `mailrag`. It carries the full bge-m3
runtime (FlagEmbedding + torch), the test suite, the `wizard` TUI, and the MCP
server — there is no separate build/test split:

```bash
conda create -y -n mailrag python=3.11
conda activate mailrag
just install                         # poetry install --with dev --all-extras --no-interaction --no-root
python -m pytest tests/ -q           # expect: all pass, a handful of opt-in skips
```

Poetry itself comes from `pipx install poetry` or from conda-forge into the env
(`conda install -n mailrag -c conda-forge poetry`). With the env active, `poetry install`
installs into it. `pyproject.toml` and `poetry.lock` pin every version, and this is the
same install CI runs. The main dependencies already include `FlagEmbedding` (bge-m3),
`textual` (the wizard), `mcp` (the server) and the attachment extractors, so this single
env can build, query, run the TUI, and serve MCP. The project is Poetry **non-package
mode** (`package-mode = false`), so `pip install -e .` will fail. Don't use it.

> **Cached bge-m3 needs `HF_HUB_OFFLINE=1`.** The first build/query downloads a few GB
> of bge-m3 weights into your Hugging Face cache. After that, set `HF_HUB_OFFLINE=1`
> so FlagEmbedding loads straight from cache instead of contacting the Hub on every
> run — faster, and it works with no network. Export it in your shell profile, or
> prefix the command: `HF_HUB_OFFLINE=1 ./mailrag ask "…"`.

All commands below assume you run them from the repo root via the `./mailrag` shim
(`exec python -m src.cli`), inside this env — e.g. `conda run -n mailrag ./mailrag …`.

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
For pointing the LLM / embedder / vector store at a specific backend, see
[`BACKENDS.md`](BACKENDS.md).

---

## 5. Quick demo (public data)

```bash
just demo        # starts Qdrant, builds a plain and a contextual index over 1,200 Enron emails, scores both
```

`just demo` runs [`scripts/quickstart.sh`](../scripts/quickstart.sh) (bring up Qdrant →
`python -m scripts.demo`). See [`QUICKSTART.md`](QUICKSTART.md) for the trimmed walkthrough.

---

## 6. Full local `.eml` pipeline

Export your mailbox to a folder of `.eml` files (a non-iCloud path), then drive the
clean-and-index pipeline through a **persona** — a named recipe for a
budget/quality tradeoff (`llm-none` / `llm-verify` / `llm-all`; see
[`GUIDE.md`](GUIDE.md) and [`VERBS.md`](VERBS.md)). You don't wire the verbs by hand;
you scope a profile, then let the wizard or a headless `run` walk the recipe:

```bash
# 1) scope + measure a profile (which folders, what chunk size)
./mailrag scope   --profile ~/rag_eml.profile.json     # interactive folder picker
./mailrag measure --profile ~/rag_eml.profile.json

# 2) (optional, free) surface the noise pockets and get a persona recommendation
./mailrag scan    --profile ~/rag_eml.profile.json

# 3a) full-screen guided run: pick a persona, review the plan, watch it live
HF_HUB_OFFLINE=1 ./mailrag wizard --profile ~/rag_eml.profile.json

# 3b) …or run a persona headlessly (same recipes, no prompts)
HF_HUB_OFFLINE=1 ./mailrag run --profile ~/rag_eml.profile.json \
    --persona llm-all --model <your-local-model>
#    add --limit N for a fast end-to-end test on a small sample
```

The persona owns the ordering (for `llm-all`: `scope → measure → calibrate → summarize →
prune → index → attach`. Every recipe in [`personas.yaml`](../personas.yaml) ends with
`index → attach`). `index` is the only step that actually drops anything, and your raw
`.eml` files stay on disk, so every choice is reversible. `--model` is required for any persona with LLM
steps. When it finishes you have a queryable collection (see §8) — ask it with
`./mailrag ask` or serve it over MCP (see [`MCP_SERVER.md`](MCP_SERVER.md)).

### One-shot build

For a zero-config build straight from an `.eml` directory (no profile, no persona
picking), use `onboard`:

```bash
HF_HUB_OFFLINE=1 ./mailrag onboard /path/to/eml-dir --collection my-rag --chunk-size 512
```

### Advanced: the by-hand verbs & scripts

Every persona step is also a standalone verb (`./mailrag tag`, `./mailrag summarize`,
`./mailrag index`, …), so you can run the pipeline one stage at a time — see
[`VERBS.md`](VERBS.md) for the full ladder and the cost of each verb. The older
standalone scripts under `scripts/` (`select_local_eml.py`, `llm_pass2.py`,
`build_local_eml_rag.py`) predate the verb interface and remain for reference only.

---

## 7. Running tests

```bash
conda run -n mailrag python -m pytest tests/ -q          # full suite
# stdlib-only modules can run on the host without the env, e.g.:
PYTHONPATH=. python3 tests/test_llm_cache.py
```

Two skips are by design: the LM-Studio smoke test (opt-in via `RUN_LMSTUDIO_SMOKE=1`) and
a test that loads a real `config/noise_rules.yaml` (skips when only the template is present).
CI runs the same suite with a coverage floor. See [`CI.md`](CI.md).

---

## 8. What you end up with

A Qdrant collection (dense bge-m3 + learned sparse, payload-indexed, optional per-email
summaries) you can query with hybrid RRF retrieval — from the CLI (`./mailrag ask`) or
over MCP. See [`EXPERIMENTS.md`](EXPERIMENTS.md) for the measured effect of each
cleanup/retrieval step.

---

## 9. Attachment extraction

`./mailrag index` puts attachment text into the search index alongside email bodies.
`./mailrag attachments build`, which is also the `attach` step every recipe ends with,
fills the collection's attachment store, the copy agents open files from over MCP. Its
noise-signal pass OCRs the small blobs (below `--classify-max-size`) to tell decoration
from documents. Extraction is handled by a subsystem in
`src/attachments/extract/`: one handler per content type (plaintext, html, docx,
xlsx, pptx, pdf, image) feeding into a swappable OCR-provider backend.

### OCR / vision backend

For images and image-only / scanned PDFs the default backend is a **local
vision-LLM** (the multimodal model `RAG_LLM_MODEL` names at `RAG_LLM_API_BASE`) —
fully on-device, no data leaves the machine.
If the LLM is unavailable it falls back automatically to local **tesseract**. Cloud
is opt-in and not yet implemented.

The LLM returns a structured description + transcription. Scanned-PDF pages read by
any OCR backend (vision LLM or tesseract) are capped by `RAG_ATTACH_MAX_PAGES`
(default `10`); pages past the cap are never rendered, and truncation is logged.

### Config

| Env var | Default | Values |
|---------|---------|--------|
| `RAG_ATTACH_EXTRACTOR` | `llm` | `llm` · `tesseract` · `cloud` (not implemented). Used by `index` and when a file is opened (`attachments get`, MCP `get_attachment`) |
| `RAG_ATTACH_CLASSIFY_EXTRACTOR` | `tesseract` | OCR backend for the noise-signal pass of `attachments build` |
| `RAG_ATTACH_STORE` | `~/.mailrag/attachments` | store root. Each collection gets its own subdirectory |
| `RAG_ATTACH_MAX_PAGES` | `10` | integer — max scanned-PDF pages rendered/read per attachment (any OCR backend) |

`--extractor <name>` overrides the backend per call: on `./mailrag attachments get` it
replaces `RAG_ATTACH_EXTRACTOR`, on `./mailrag attachments build` it replaces
`RAG_ATTACH_CLASSIFY_EXTRACTOR`. `attachments get --force` re-extracts even when a cached
result already exists.

### System packages for OCR

The Python extractors (`pypdf`, `python-docx`, `openpyxl`, `python-pptx`, `pillow`,
`pytesseract`, `pdf2image`) are main dependencies, so `just install` already has them.
OCR also needs two system packages: **tesseract** and **poppler**
(`brew install tesseract poppler` on macOS, or `just ocr-deps` to check).

Without these, extraction degrades gracefully: unsupported types are stored with
status `binary` or `ocr_unavailable`, and the raw attachment file is always
served regardless.

Note that the Python packages and the system binaries fail *differently*.
`pytesseract` imports happily with no `tesseract` on `PATH` and only fails when it
shells out, so a missing binary is caught at OCR time rather than at import. Both
cases report `ocr_unavailable`, which matters because that status is deliberately
**not cached** — a later run with a working `PATH` retries instead of inheriting a
stale verdict. `error` means the attachment itself could not be read, and *is*
cached. If a scheduled sync reports `ocr_unavailable` for everything, check the
`PATH` it inherits (launchd jobs get a minimal one) before suspecting the files.

Poppler can also go missing in *halves* — `pdfinfo` installed but `pdftoppm` not,
which is what a minimal container image or an interrupted `brew install` leaves
behind. Scanned PDFs still report `ocr_unavailable` in that case, so installing
the missing binary and re-running is enough; nothing needs clearing by hand.
