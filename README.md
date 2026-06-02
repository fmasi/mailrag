# mailrag

> A pluggable, multi-backend **Email RAG** engine built on LlamaIndex — load emails
> from multiple sources, clean and chunk them, embed with hybrid dense+sparse
> retrieval, and query them with an LLM.

[![Test Suite](https://github.com/fmasi/mailrag/actions/workflows/test-suite.yml/badge.svg)](https://github.com/fmasi/mailrag/actions/workflows/test-suite.yml)
![Python](https://img.shields.io/badge/python-3.11+-blue)
![License](https://img.shields.io/badge/license-Apache%202.0-blue)

> **Headline.** On a real ~32k-email corporate mailbox, the full stack takes answer coverage from
> **45% (plain email RAG) → 84%** (coverage@3) — nearly doubling recall@1 (36% → 70%) — as the
> *compound* effect of thread-aware expansion and preceding-context contextual summaries. The point
> isn't only the number; it's the **methodology and rigor** behind it: **360 validated queries, an evolution ladder
> that prices every increment, significance tests, and confounds caught and corrected** (a +6pp
> headline that proved half a quantization artifact), **with all negative results kept in.**
> A reproducible, documented worked example.

## What it does

`mailrag` turns a mailbox into a queryable knowledge base:

- **Pluggable loaders** — public Enron corpus (HuggingFace), local `.eml` archives,
  or Azure Blob Storage, behind one `EmailLoader` interface.
- **Email-aware preprocessing** — reply-chain stripping, calendar-invite collapsing,
  noise/newsletter filtering, exact-text chunk dedup.
- **Hybrid retrieval** — bge-m3 dense + sparse vectors (RRF-fused) in Qdrant (also
  supports local persistence and Pinecone), with optional **thread-aware expansion**
  (match a small unit, answer from its full conversation).
- **LLM "Pass-2"** — optional local-LLM summarization/judging of each email,
  content-addressed and cached.
- **A measured methodology** — a labeled, LLM-judged retrieval eval (360 validated
  queries) that quantifies each technique, controls for confounds, reports
  significance, and in several cases *overturned* the intuitive choice.
- **Source-agnostic API** — `load_emails(source="enron"|"mail_archive_x"|"azure_blob")`.

## Quickstart (thread-aware contextual RAG over the public Enron dataset)

```bash
git clone https://github.com/fmasi/mailrag.git
cd mailrag
pip install -r requirements.txt        # includes FlagEmbedding (bge-m3); first run downloads ~2 GB of weights
cp .env.example .env                    # add an LLM key/endpoint (used for summaries + answers)
make demo                               # starts Qdrant, builds the contextual index, runs thread-aware queries
```

`make demo` brings up Qdrant (Docker), builds a **thread-aware contextual** index over 100 Enron
emails — per-email preceding-context summaries embedded with bge-m3 hybrid vectors — then answers
example questions by retrieving and assembling whole threads. This is the §13 stack (see the case
study); a small amount of LLM usage is spent on the Pass-2 summaries and the answers.

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
| **Net** | | 31,969 emails → **19,859 kept** → 21,590 embedded chunks |

**The honest part — how much of this needed an LLM?** We measured it. Regex rules
*derived from the corpus* (high-noise sender domains + calendar/out-of-office subject
patterns) catch **~65%** of the LLM's noise at high precision, but **miss ~35%
(≈4,200 emails)**. The miss is structural: the work domain itself is **29% noise**
(24k emails interleaving real correspondence with compliance reminders, calendar churn,
AMAs, internal newsletters) — you can't write a sender rule for your own domain, and the
noise inside it isn't cleanly separable by sender/subject. That ~35% is the LLM's *unique*
cleaning contribution. Two further findings:

- **Rule *discovery* did not need a full pass.** The dominant noise senders
  (LinkedIn, Zoom, SharePoint, …) jump straight out of a sender-frequency table — a small
  sample (or non-LLM frequency analysis) reveals them; the 32k pass wasn't required to
  *find* the rules.
- **The 48 h → under-10-min embedding win was the *inference method*** (FlagEmbedding on
  Apple-Silicon MPS) plus volume reduction — **not** the LLM.

So the local-LLM pass earns its keep two ways: the **~35% mixed-domain noise** that cheap
rules can't reach, **and** the per-email **summaries** that power the retrieval gains
below (contextual retrieval, reranking) and human-readable results. **Lesson: use cheap
regex for the obvious bulk; reserve the LLM for the interleaved noise and the summaries
only it can produce.**

### Retrieval methodology — what each technique adds (and its trade-off)

| technique | what it adds | trade-off (observed) |
|-----------|--------------|----------------------|
| **Dense (semantic) only** | matches meaning & paraphrase | misses rare exact tokens (acronyms, IDs); returns redundant near-duplicate chunks |
| **+ learned sparse + RRF fusion** (bge-m3) | exact-token / acronym precision, fused with semantics | needs a sparse-capable embedder + fusion; more storage |
| **+ LLM noise removal** | precision — catches the ~⅓ of noise that regex can't; *without it*, **21% of queries surface noise in their top-3 and ~11% of retrieval slots are junk** (measured) | one-time LLM cost (see above) |
| **+ contextual retrieval** (prepend each email's summary before embedding — the `C′` / `work-rag-ctx-*` collection) | short/terse emails match by *gist*; in the labeled eval, the **best ranked arm** *and* the end-to-end winner | one extra embedded collection to build/maintain |
| **+ cross-encoder reranker** | *(intuition: reorder candidates for precision)* | **measured to HURT** — under an LLM judge it demotes answer-bearing emails (§9); **off by default** |
| **+ thread-aware expansion** (pull the full conversation of each top hit) | **~doubles answer-coverage** (terse replies 33% → ~80%) — match a small unit, answer from its thread | larger context per query (tunable: expand top-N threads) |

**What the labeled evals settled.** The eval set grew to **360 validated, LLM-screened queries**,
scored as an **evolution ladder** — body-only → +thread expansion → +summary → +thread-aware summary
— with significance tests and confound controls (full write-ups in
[`EXPERIMENTS.md` §9–§13](docs/EXPERIMENTS.md)):

- **Thread expansion is the biggest single win — and it needs no LLM.** Matching a small unit and
  returning its whole conversation lifts recall@1 from 36% → 60% (terse answer-coverage 33% → ~80%):
  match-small, answer-from-the-thread.
- **Thread-aware *summaries* help where they're designed to — terse replies.** *(Note: "thread-aware"
  names two distinct things — the **retrieval** expansion above, and this **summary-conditioning**
  step; see the [terminology box](docs/EXPERIMENTS.md#terminology-read-this-first).)* Conditioning each
  email's embedded summary on its *preceding* thread context significantly improves terse-reply
  retrieval (covered@3 75% → 81%, p = 0.035). The corpus-wide effect is real but modest (+3pp), and
  we report it as such rather than rounding up.
- **A confound caught and reported.** An early +6pp headline turned out to be half a *quantization*
  artifact; re-running the control at matched quant split it into +3pp (quant) + +3pp (method).
  Holding the summarizer fixed is the difference between a result and a mirage.
- **Cleanup pays in precision, not recall.** Leaving the noise a regex can't catch in the corpus
  barely dents gold recall (the DB still finds the answer), but then **21% of queries surface noise
  in their top-3** (~11% of slots) — junk the LLM removes for free in the same pass that writes the
  summary.
- **Two intuitive ideas, measured and rejected.** A cross-encoder reranker *hurt* under LLM-judged
  relevance; query-side HyDE never beat the raw query on this entity-rich corpus. Both are kept
  in-tree, off by default, for corpora where they'd pay off.
- **The ceiling is retrieval, not the model.** With the answer in context, even a **4 B** model
  answered ~88% correctly; the lost points are queries where retrieval never surfaced the thread.
  Model size was second-order.

**Compound effect — the whole point.** End to end, these layers take coverage@3 from **45% (plain
email RAG) to 84%**, recall@1 from **36% → 70%**, and MRR **.43 → .78** — most of it from
thread-awareness, the remainder from the contextual summary, each increment individually measured
above. The value isn't any single trick; it's the **methodology** — a disciplined stack, with the
**rigor** to prove every layer earns its place.

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
- [`docs/SETUP.md`](docs/SETUP.md) — full setup, the local `.eml` pipeline, and how to run the tests
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — design decisions & extension points
- [`docs/EMAIL_PREPROCESSING.md`](docs/EMAIL_PREPROCESSING.md) — reply-chain stripping & chunk tuning
- [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) — measured findings & trade-offs: cleanup economics, regex-vs-LLM, and the **labeled-eval ladder (§9–§13)** — thread-aware retrieval, the contextual-summary result and its quant-confound control, the cleanup precision finding, the reranker and HyDE reversals, and "retrieval is the ceiling, not the model" — all with real, anonymized numbers
- [`docs/RETRIEVAL_GUIDE.md`](docs/RETRIEVAL_GUIDE.md) — the retrieval stack end-to-end: hybrid fusion, contextual retrieval, reranking, and thread-aware expansion
- [`config/community_blocklist.template.yaml`](config/community_blocklist.template.yaml) — portable starter noise rules (~1/3 of corporate-mail noise, corpus-independent)

## License

[Apache 2.0](LICENSE) — see also [`NOTICE`](NOTICE). Copyright © 2026 Frederic Masi.
If you build on this work (code or method), please preserve the attribution in `NOTICE`.
