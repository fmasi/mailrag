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

## Case study: what the cleanup & retrieval choices actually bought

> The numbers below come from running `mailrag` on a real ~32,000-email corporate
> mailbox (all references anonymized). They're included so the repo doubles as a
> worked example — *why* each step exists, what it saves, and what it costs.

### Cleanup pipeline — measured savings (and an honest cost/benefit)

The corpus is filtered in stages before anything gets embedded:

| stage | what it does | effect on this corpus |
|------|--------------|-----------------------|
| **Scope** | keep only the work-account folders | 70,016 exported → **31,969** selected |
| **Pass-1 (regex)** | cheap sender/subject rules drop obvious bulk (newsletters, social, automated senders) *before* any expensive work | flags **10.4%** (3,332) |
| **Pass-2 (local LLM)** | summarize + judge each email's content | flags **37.9%** (12,123) as noise |
| **Calendar-collapse + chunk-dedup** | one-line calendar summaries; drop byte-identical chunks | 22,613 → **21,590** chunks (−1,023) |
| **Net** | | 31,969 emails → **19,820 kept** → 21,590 embedded chunks |

**The honest part — most of the LLM's noise removal did not justify its cost.** Of the
12,123 emails Pass-2 called noise, **97.9% (11,872)** were automated mail identifiable by
sender/subject alone — i.e. *cheap regex rules could drop them for free.* Only **2.1%
(251)** were genuine judgment calls needing content understanding. And the headline
embedding-time win — a first-run estimate of **~48 h → under 10 min** — came from the
**inference method** (FlagEmbedding on Apple-Silicon MPS) plus volume reduction, **not**
from the LLM.

So the ~13-hour local-LLM pass is **not** paid for by noise removal or speed. It earns
its keep through one durable output: the per-email **summaries**, which power the
retrieval gains below (contextual retrieval, reranking) and human-readable results.
**Lesson: use cheap regex for noise; reserve the LLM for the summaries only it can
produce.**

### Retrieval methodology — what each technique adds (and its trade-off)

| technique | what it adds | trade-off (observed) |
|-----------|--------------|----------------------|
| **Dense (semantic) only** | matches meaning & paraphrase | misses rare exact tokens (acronyms, IDs); returns redundant near-duplicate chunks |
| **+ learned sparse + RRF fusion** (bge-m3) | exact-token / acronym precision, fused with semantics | needs a sparse-capable embedder + fusion; more storage |
| **+ LLM noise removal** | precision — junk can't surface (≈1,000 spam-quarantine digests and ≈1,500 calendar notifications removed from results) | one-time LLM cost (see above) |
| **+ contextual retrieval** (prepend each email's summary before embedding) | short/terse emails match by *gist* — e.g. a 43-character reply surfaced via its summary | **topic drift**: dilutes literal matches and can pull in adjacent-but-off results; best paired with a reranker |
| **+ cross-encoder reranker** *(planned)* | reorders the fused candidates, removing contextual drift | extra per-query latency |

**Worked example.** Searching for a partner certification program by its acronym
(`"ACP"`) mixes a *semantic* concept (certification readiness) with a *rare exact token*
(`ACP`). Dense-only finds the concept but ranks the literal acronym low; sparse-only
finds the token but misses paraphrases; **hybrid + RRF gets both.** Multi-query expansion
(searching several phrasings and fusing with RRF) further bridges acronym ↔ expansion
("ACP" ↔ "Acme Certified Partner") at the cost of extra queries per search.

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
