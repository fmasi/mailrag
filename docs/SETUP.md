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
conda run -n mailrag pip install -r requirements.txt
conda run -n mailrag python -m pytest tests/ -q     # expect: all pass, a handful of opt-in skips
```

`requirements.txt` already pulls `FlagEmbedding` (bge-m3), `textual` (the wizard),
and `mcp` (the server), so this single env can build, query, run the TUI, and serve
MCP. The project is Poetry **non-package mode** (`package-mode = false`), so
`pip install -e .` will fail — don't use it. `pyproject.toml` / `poetry.lock` are
the source of truth for exact pinned versions (and what CI installs), but the
conda + `requirements.txt` path above is the supported local install.

> **Cached bge-m3 needs `HF_HUB_OFFLINE=1`.** The first build/query downloads ~2 GB
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
make demo        # starts Qdrant, then builds an index over 100 Enron emails and answers example queries
```

`make demo` runs [`scripts/quickstart.sh`](../scripts/quickstart.sh) (bring up Qdrant →
`python main.py`). See [`QUICKSTART.md`](QUICKSTART.md) for the trimmed walkthrough.

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

The persona owns the ordering (`tag → scan → calibrate → summarize → prune → index`);
`index` is the only step that actually drops anything, and your raw `.eml` files stay
on disk, so every choice is reversible. `--model` is required for any persona with LLM
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
CI runs the same suite with a coverage floor — see the
[CI / quality gates](../README.md#ci--quality-gates) table.

---

## 8. What you end up with

A Qdrant collection (dense bge-m3 + learned sparse, payload-indexed, optional per-email
summaries) you can query with hybrid RRF retrieval — from the CLI (`./mailrag ask`) or
over MCP. See [`EXPERIMENTS.md`](EXPERIMENTS.md) for the measured effect of each
cleanup/retrieval step.

---

## 9. Attachment extraction

`./mailrag attachments build` extracts text from attachments so their content is
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
`./mailrag attachments get` and `./mailrag attachments build`. Use `--force` to
re-extract even when a cached result already exists.

### Optional Python dependencies

```bash
conda run -n mailrag pip install \
    pypdf python-docx openpyxl python-pptx pillow pytesseract pdf2image
```

Plus system packages for OCR: **tesseract** and **poppler**
(e.g. `brew install tesseract poppler` on macOS).

Without these, extraction degrades gracefully: unsupported types are stored with
status `binary` or `ocr_unavailable`, and the raw attachment file is always
served regardless.
