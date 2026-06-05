# mailrag

> A pluggable, multi-backend **Email RAG** engine built on LlamaIndex: load mail from
> several sources, clean and chunk it, embed it with hybrid dense+sparse retrieval, and
> query it with an LLM.

[![Test Suite](https://github.com/fmasi/mailrag/actions/workflows/test-suite.yml/badge.svg)](https://github.com/fmasi/mailrag/actions/workflows/test-suite.yml)
![Python](https://img.shields.io/badge/python-3.11+-blue)
![License](https://img.shields.io/badge/license-Apache%202.0-blue)

> **What it buys you.** On a real ~32k-email corporate mailbox, the full stack takes answer
> coverage@3 from **45%** (plain email RAG) **to 84%**, and recall@1 from **36% to 70%** — most
> of it from thread-aware expansion, the rest from preceding-context summaries. These are
> author-reported numbers on a *private* mailbox; the public `make demo` reproduces the
> *method*, not the exact figures. How they were measured, and the honest caveats, are in the
> [case study](#case-study-what-the-cleanup--retrieval-choices-actually-bought) below.

## What it does

`mailrag` turns a mailbox into a queryable knowledge base:

- **Pluggable loaders** — public Enron corpus (HuggingFace), local `.eml` archives,
  or Azure Blob Storage, behind one `EmailLoader` interface.
- **Email-aware preprocessing** — reply-chain stripping, calendar-invite collapsing,
  noise/newsletter filtering, exact-text chunk dedup.
- **Hybrid retrieval** — bge-m3 dense + sparse vectors (RRF-fused) in Qdrant (also
  supports local persistence and Pinecone), with optional **thread-aware expansion**
  (match a small unit, answer from its full conversation).
- **LLM "Pass-2"** — optional local-LLM summarization and noise-judging of each email,
  content-addressed and cached, so re-runs are free.
- **A measured methodology** — a 360-query retrieval eval that prices each technique,
  controls for confounds, reports significance, and in several cases *overturned* the
  intuitive choice. The core techniques were later confirmed against a public,
  human-judged benchmark.
- **Source-agnostic API** — `load_emails(source="enron"|"mail_archive_x"|"azure_blob")`.

## Quickstart (thread-aware contextual RAG over the public Enron dataset)

```bash
git clone https://github.com/fmasi/mailrag.git
cd mailrag
pip install -r requirements.txt        # includes FlagEmbedding (bge-m3); first run downloads ~2 GB of weights
cp .env.example .env                    # add an LLM key/endpoint (used for summaries + answers)
make demo                               # starts Qdrant, builds the contextual index, runs thread-aware queries
```

`make demo` brings up Qdrant (Docker), builds a thread-aware contextual index over 100 Enron
emails — per-email preceding-context summaries embedded with bge-m3 hybrid vectors — then
answers example questions by retrieving and assembling whole threads. This is the
[§13 stack](docs/EXPERIMENTS.md#13-doc-side-thread-aware-summaries--the-evolution-ladder-1113-2026-06-01);
a small amount of LLM usage goes to the Pass-2 summaries and the answers.

## Architecture

```
                       ┌─────────────────────────────┐
   sources             │      EmailLoader (ABC)       │
  ┌─────────┐          ├──────────┬─────────┬─────────┤
  │  Enron  │──────────│  enron   │ mail_   │  azure  │
  │ .eml    │          │          │ archive │  blob   │
  │ Azure   │          └────┬─────┴────┬────┴────┬────┘
  └─────────┘               │   NormalizedEmail    │
                            ▼                      ▼
            Pass-1: regex noise filter — tag bulk/newsletters   (no LLM)
                            ▼
            Pass-2: local LLM — summarize + judge noise, cached  (optional)
                            ▼
            drop noise · dedup · reply-chain strip
              └ drop stage is tunable: Pass-1 = save LLM budget · Pass-2 = best quality
                            ▼
            chunk (SentenceSplitter, bge-m3 tokenizer)
              └ optional: prepend each email's summary  → contextual retrieval
                            ▼
            embed (bge-m3 dense + sparse)
                            ▼
        ┌───────────────────┼────────────────────┐
        ▼                   ▼                     ▼
   local persist        Qdrant (hybrid)       Pinecone
                            ▼
            query engine (hybrid RRF · thread-aware expansion · optional rerank)
```

Pass-1 only *tags* by default, so nothing is lost before the LLM sees it; the confident
drop happens at Pass-2. Where you drop is a deliberate budget-vs-quality knob — drop at
Pass-1 to skip the LLM cost, or at Pass-2 for the cleaner result.

## Case study: what the cleanup & retrieval choices actually bought

> Most numbers below come from running `mailrag` on a real ~32,000-email corporate mailbox
> (all references anonymized); the portability check at the end uses a second, personal
> archive. They're here so the repo doubles as a worked example — *why* each step exists,
> what it saves, and what it costs.

### Cleanup pipeline — measured savings, and an honest cost/benefit

The corpus is filtered in stages before anything gets embedded:

| stage | what it does | effect on this corpus |
|------|--------------|-----------------------|
| **Scope** | keep only the work-account folders | 70,016 exported → **31,969** selected |
| **Pass-1 (regex)** | cheap sender/subject rules drop obvious bulk (newsletters, social, automated senders) *before* any expensive work | flags **10.4%** (3,332) |
| **Pass-2 (local LLM)** | summarize + judge each email's content | flags **37.9%** (12,123) as noise |
| **Calendar-collapse + chunk-dedup** | one-line calendar summaries; drop byte-identical chunks | 22,613 → **21,590** chunks (−1,023) |
| **Net** | | 31,969 emails → **19,859 kept** → 21,590 embedded chunks |

**How much of this actually needed an LLM?** We measured it. Regex rules derived from the
corpus (high-noise sender domains, calendar/out-of-office subject patterns) catch ~65% of
the LLM's noise at high precision, but miss ~35% (≈4,200 emails). The miss is structural:
the work domain itself is 29% noise — 24k emails interleaving real correspondence with
compliance reminders, calendar churn, AMAs, and internal newsletters — and you can't write
a sender rule for your own domain. That ~35% is the LLM's unique contribution. Two
qualifiers keep it honest:

- **Rule *discovery* didn't need a full pass.** The dominant noise senders (LinkedIn, Zoom,
  SharePoint, …) fall straight out of a sender-frequency table; a small sample finds the
  rules without the 32k run.
- **The 48 h → under-10-min embedding win was the *inference method*** (FlagEmbedding on
  Apple-Silicon MPS) plus volume reduction, not the LLM.

So the local-LLM pass earns its keep twice: the ~35% mixed-domain noise cheap rules can't
reach, and the per-email summaries that power the retrieval gains below (contextual
retrieval, reranking) and make results human-readable. The lesson: use cheap regex for the
obvious bulk, reserve the LLM for the interleaved noise and the summaries only it can write.

**And the rubric itself doesn't port across corpora.** Run the same pipeline over a
~25,000-email *personal* archive and the point makes itself: the *corporate* noise rubric
flagged **87.6%** of personal mail as noise — it would have deleted real receipts, bank
statements, and correspondence — while a rubric calibrated for the personal corpus flagged
**61.5%**. A cheap ~200-email calibration caught the gap before the ~6 h run, and a
spot-check of the dropped pile confirmed it, all on a local model with no cloud spend. Full
write-up:
[EXPERIMENTS §14](docs/EXPERIMENTS.md#14-a-second-corpus--the-llm-rubric-is-not-portable-30-2026-06-05).

### Retrieval methodology — what each technique adds (and its trade-off)

| technique | what it adds | trade-off (observed) |
|-----------|--------------|----------------------|
| **Dense (semantic) only** | matches meaning & paraphrase | misses rare exact tokens (acronyms, IDs); returns redundant near-duplicate chunks |
| **+ learned sparse + RRF fusion** (bge-m3) | exact-token / acronym precision, fused with semantics | needs a sparse-capable embedder + fusion; more storage |
| **+ LLM noise removal** | precision — catches the ~⅓ of noise regex can't, and clears junk out of the top results (measured below) | one-time LLM cost (see above) |
| **+ contextual retrieval** (prepend each email's summary before embedding — the `C′` / `work-rag-ctx-*` collection) | short/terse emails match by *gist*; the best ranked arm *and* the end-to-end winner | one extra embedded collection to build/maintain |
| **+ cross-encoder reranker** | *(intuition: reorder candidates for precision)* | **measured to HURT** — under an LLM judge it demotes answer-bearing emails (§9); off by default |
| **+ thread-aware expansion** (pull the full conversation of each top hit) | **~doubles answer-coverage** (terse replies 33% → ~80%) — match a small unit, answer from its thread | larger context per query (tunable: expand top-N threads) |

**How the eval was run.** The eval set is **360 synthetic queries** — generated from corpus
bodies and graded by a local LLM judge. To trust that judge, it was calibrated against a
stronger reference model (Cohen's κ = 0.52, Spearman 0.74), and both pre-registered
decisions held under either judge. The core techniques were then cross-checked on the
**TREC Legal Track's real human relevance judgments**, which broadly agreed. Results are
scored as an evolution ladder — body-only → +thread expansion → +summary → +thread-aware
summary — with significance tests and confound controls (full write-ups in
[`EXPERIMENTS.md` §9–§13](docs/EXPERIMENTS.md#9-labeled-eval--retrieval-metrics-coverage-and-end-to-end-answer-quality-2026-05-29)):

- **Thread expansion is the biggest single win — and needs no LLM.** Matching a small unit
  and returning its whole conversation lifts recall@1 from 36% → 60% (terse answer-coverage
  33% → ~80%).
- **Thread-aware *summaries* help where they're designed to — terse replies.** *(Note:
  "thread-aware" names two things — the retrieval expansion above, and this
  summary-conditioning step; see the
  [terminology box](docs/EXPERIMENTS.md#terminology-read-this-first).)* Conditioning each
  email's embedded summary on its *preceding* thread context lifts terse-reply retrieval
  from covered@3 75% → 81% (p = 0.035). The corpus-wide effect is real but modest (+3pp),
  and we report it as such.
- **A confound caught and reported.** An early +6pp headline turned out to be half a
  *quantization* artifact; re-running the control at matched quant split it into +3pp (quant)
  + +3pp (method). Holding the summarizer fixed is the difference between a result and a mirage.
- **Cleanup pays in precision, not recall.** Leaving the noise a regex can't catch barely
  dents gold recall (the DB still finds the answer), but then **21% of queries surface noise
  in their top-3** (~11% of slots) — junk the LLM removes for free in the pass that also
  writes the summary.
- **Two intuitive ideas, measured and rejected.** A cross-encoder reranker *hurt* under
  LLM-judged relevance; query-side HyDE never beat the raw query on this entity-rich corpus.
  Both are kept in-tree, off by default, for corpora where they'd pay off.
- **The ceiling is retrieval, not the model.** With the answer in context, even a 4 B model
  answered ~88% correctly; the lost points are queries where retrieval never surfaced the
  thread. Model size was second-order.

**The compound effect.** Stacked, that ladder is what produces the headline: coverage@3
**45% → 84%**, recall@1 **36% → 70%**, MRR **.43 → .78** — most of it from thread-awareness,
the rest from the contextual summary, each increment individually measured above. The value
isn't any single trick; it's the disciplined stack and the rigor to prove every layer earns
its place.

**Worked example.** Searching for a partner certification program by its acronym (`"ACP"`)
mixes a semantic concept (certification readiness) with a rare exact token (`ACP`).
Dense-only finds the concept but ranks the literal acronym low; sparse-only finds the token
but misses paraphrases; hybrid + RRF gets both. Multi-query expansion (searching several
phrasings and fusing with RRF) further bridges acronym ↔ expansion ("ACP" ↔ "Acme Certified
Partner"), at the cost of extra queries per search.

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

Full map and reading order: **[`docs/INDEX.md`](docs/INDEX.md)**. The reader journey is
**this page → quickstart → setup → deep dives**:

1. **You are here** (`README.md`) — overview, quickstart, and the case study.
2. [`docs/QUICKSTART.md`](docs/QUICKSTART.md) — 5-minute setup and copy-paste usage patterns.
3. [`docs/SETUP.md`](docs/SETUP.md) — full setup, the local `.eml` pipeline, and how to run the tests.
4. Deep dives:
   - [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — design decisions & extension points.
   - [`docs/EMAIL_PREPROCESSING.md`](docs/EMAIL_PREPROCESSING.md) — reply-chain stripping & chunk tuning.
   - [`docs/RETRIEVAL_GUIDE.md`](docs/RETRIEVAL_GUIDE.md) — the retrieval stack end-to-end: hybrid fusion, contextual retrieval, reranking, and thread-aware *retrieval* (small→big expansion).
   - [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) — the measured findings behind the case study: cleanup economics, regex-vs-LLM, the labeled-eval ladder (§9–§13), and the corpus-portability result (§14). Start with its [terminology box](docs/EXPERIMENTS.md#terminology-read-this-first) for the `C`/`C′` labels and the two senses of "thread-aware".

Reference: [`config/community_blocklist.template.yaml`](config/community_blocklist.template.yaml) — portable starter noise rules (~1/3 of corporate-mail noise, corpus-independent).

## License

[Apache 2.0](LICENSE) — see also [`NOTICE`](NOTICE). Copyright © 2026 Frederic Masi.
If you build on this work (code or method), please preserve the attribution in `NOTICE`.
