# mailrag

> A pluggable, multi-backend **Email RAG** engine built on LlamaIndex — load emails
> from multiple sources, clean and chunk them, embed with hybrid dense+sparse
> retrieval, and query them with an LLM.

[![Test Suite](https://github.com/fmasi/mailrag/actions/workflows/test-suite.yml/badge.svg)](https://github.com/fmasi/mailrag/actions/workflows/test-suite.yml)
![Python](https://img.shields.io/badge/python-3.11+-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## What it does

`mailrag` turns a mailbox into a queryable knowledge base:

- **Pluggable loaders** — public Enron corpus (HuggingFace), local `.eml` archives,
  or Azure Blob Storage, behind one `EmailLoader` interface.
- **Email-aware preprocessing** — reply-chain stripping, calendar-invite collapsing,
  noise/newsletter filtering, exact-text chunk dedup.
- **Hybrid retrieval** — bge-m3 dense + sparse vectors stored in Qdrant (also
  supports local persistence and Pinecone).
- **LLM "Pass-2"** — optional local-LLM summarization/judging of each email,
  content-addressed and cached.
- **Source-agnostic API** — `load_emails(source="enron"|"mail_archive_x"|"azure_blob")`.

## Quickstart (runs against the public Enron dataset)

```bash
git clone https://github.com/fmasi/mailrag.git
cd mailrag
poetry install            # or: pip install -r requirements.txt
cp .env.example .env      # add your OPENAI_API_KEY
python main.py            # builds an index over 100 Enron emails and runs demo queries
```

`main.py` initializes config, loads 100 Enron emails, builds the index, and runs
three example queries (pure retrieval, RAG-with-LLM, metadata-filtered).

## Architecture

```
                       ┌─────────────────────────────┐
   sources             │      EmailLoader (ABC)       │
  ┌─────────┐          ├──────────┬─────────┬─────────┤
  │  Enron  │──────────│  enron   │ mail_   │  azure  │
  │ .eml    │          │          │ archive │  blob   │
  │ Azure   │          └────┬─────┴────┬────┴────┬────┘
  └─────────┘               │ NormalizedEmail     │
                            ▼                      ▼
                  preprocess (noise filter, dedup, reply-chain strip)
                            ▼
                  chunk (SentenceSplitter, bge-m3 tokenizer)
                            ▼
                  embed (bge-m3 dense + sparse)
                            ▼
        ┌───────────────────┼────────────────────┐
        ▼                   ▼                     ▼
   local persist        Qdrant (hybrid)       Pinecone
                            ▼
                  query engine (retrieval / RAG / metadata filter)
```

## Project layout

| Path | Responsibility |
|------|----------------|
| `src/config/` | Configuration + LlamaIndex `Settings` |
| `src/data/` | `NormalizedEmail` model, multi-source `load_emails` API |
| `src/data/loaders/` | Pluggable source loaders (enron, mail_archive_x, azure_blob) |
| `src/ingest/` | Embedding (bge-m3), sparse vectors, hybrid Qdrant upsert |
| `src/indexing/` | Index creation/management |
| `src/storage/` | Persistence (local / Pinecone / Qdrant) |
| `src/query/` | Retrieval + RAG query engine |
| `src/llm/` | Optional LLM "Pass-2" summarization + cache |
| `scripts/` | Build / index / maintenance utilities |
| `tests/` | Test suite (pytest) |
| `docs/` | Architecture, quickstart, preprocessing guides |

## Documentation

- [`docs/QUICKSTART.md`](docs/QUICKSTART.md) — 5-minute setup
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — design decisions & extension points
- [`docs/EMAIL_PREPROCESSING.md`](docs/EMAIL_PREPROCESSING.md) — reply-chain stripping & chunk tuning

## License

[MIT](LICENSE)
