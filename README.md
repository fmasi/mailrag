# mailrag

> Private, self-hosted **Email RAG**: turn your own mail archive into a queryable knowledge
> base that runs on your hardware, on open models, with nothing required to leave your network.
> A faithful, private record of what you've written and received — an AI memory you actually own,
> one half of a context stack you own. (The other half is
> [parley](https://github.com/fmasi/parley), for calls and meetings.)

[![Test Suite](https://github.com/fmasi/mailrag/actions/workflows/test-suite.yml/badge.svg)](https://github.com/fmasi/mailrag/actions/workflows/test-suite.yml)
![Python](https://img.shields.io/badge/python-3.11+-blue)
![License](https://img.shields.io/badge/license-Apache%202.0-blue)

> **What it buys you.** On a real ~32k-email corporate mailbox, stacking the retrieval
> techniques takes **recall@5 from 46% (plain dense) to 93%** — and the two biggest levers
> aren't a fancier model: a per-email contextual summary (**+13**) and reconstructing the whole
> thread instead of hunting one message (**+29**). As a yardstick the email-tuned hybrid is
> benchmarked against NVIDIA's general-purpose retrieval stack: it wins on email, while NVIDIA's
> stack wins on broad legal e-discovery (TREC) — same systems, opposite winners, task-dependent.
> Numbers are author-reported on a *private* mailbox (cross-checked on public Enron-QA, same
> ordering); the 93% is thread-level recall (see caveats) and the public `make demo` reproduces
> the *method*. Full write-up and reproducible scripts in the
> [case study](#case-study-what-the-cleanup--retrieval-choices-actually-bought) below.

## Why this exists

The first time I pointed cloud AI at my inbox it felt like a superpower — until I thought about
what it actually required: handing my entire email history to someone else's servers to make it
searchable. For real correspondence — contracts, receipts, the record of who agreed to what —
that's a non-starter.

So I built the opposite. mailrag runs on your own hardware, on open models, with nothing required
to leave your network — no mailbox upload, no vendor to trust with the whole archive.

Then the real point clicked. These aren't just emails, they're **context**. A faithful, private
record of what was said and written is exactly what an AI agent needs to be useful about *your*
work — kept private and self-owned, so you get total recall without renting your memory to anyone.
mailrag is one private context source, for **email**; [parley](https://github.com/fmasi/parley) is
another, for **calls and meetings** — different domain, different machinery (on-device audio +
diarization). They don't talk to each other; my agents know about both and reach for whatever fits.
The point was never a single app — it's a private, open stack of context I own.

## What it does

`mailrag` turns a mailbox into a queryable knowledge base, built on LlamaIndex:

- **Pluggable loaders** — public Enron corpus (HuggingFace), local `.eml` archives,
  or Azure Blob Storage, behind one `EmailLoader` interface.
- **Email-aware preprocessing** — reply-chain stripping, calendar-invite collapsing,
  noise/newsletter filtering, exact-text chunk dedup.
- **Thread-aware answers** (the flagship) — match a single small unit, then answer from
  its *entire* conversation. It roughly **doubles** answer coverage (terse replies 33% →
  ~80%), it's the biggest single retrieval win, and it needs no LLM.
- **Hybrid retrieval** — bge-m3 dense + sparse vectors (RRF-fused) in Qdrant (also
  supports local persistence and Pinecone). Gets both the concept and the rare exact
  token — acronyms, IDs, reference numbers.
- **Local-LLM `summarize`** — optional per-email summary + noise judgement from a local
  LLM, content-addressed and cached, so re-runs are free.
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

Between the two there's an optional, **no-LLM** triage: `mailrag explore` clusters the
corpus embeddings at thread level and ranks the densest "noise pockets" (bulk and
automated mail) by pass-1-tag enrichment, sender concentration, and tightness. It spends
no LLM budget — it reuses the already-embedded vectors when a collection exists, else
embeds once — and writes a JSON artifact (thread → `.eml` paths) so you can see where the
noise concentrates before deciding how much Pass-2 to run.

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
| **+ cross-encoder reranker** | small precision lift on pointed queries (**+2.5 R@5**) | **demotes the answer on thread-spanning queries** (and hurt outright under the earlier LLM-judged eval, §9); off by default |
| **+ thread reconstruction** (pull the full conversation of each top hit) | **recall@5 62% → 93%** — match a small unit, answer from its whole thread | larger context per query (tunable: expand top-N threads) |

**How the eval was run.** The eval set is **360 synthetic queries** (144 terse / 144 content /
72 spanning), each generated from a known email so the **recall ladder is scored against hard
gold labels — no LLM judge in the loop**. A *separate* answer-quality lens does use a local LLM
judge, calibrated against a stronger reference model (Cohen's κ = **0.52** on the 0–3 scale,
**0.80** binary at the relevance threshold actually used; Spearman 0.74). The core techniques
were cross-checked on the **TREC Legal Track's real human judgments** and on public **Enron-QA**,
which agreed on ordering. Significance tests and confound controls are in
[`EXPERIMENTS.md` §9–§13](docs/EXPERIMENTS.md#9-labeled-eval--retrieval-metrics-coverage-and-end-to-end-answer-quality-2026-05-29):

- **Thread reconstruction is the biggest single win — and needs no LLM.** Matching a small unit
  and returning its whole conversation lifts **recall@5 from 62% → 93%** (+29) — it trades
  "find the needle" for "find the right thread," which the conversation then answers.
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
- **Reranking helps pointed questions but hurts thread-spanning ones.** A cross-encoder reranker
  adds only **+2.5 recall@5** overall and *demotes* the answer on multi-email questions (no single
  message looks like the whole answer) — and it hurt outright under the earlier LLM-judged
  answer-quality eval. Query-side HyDE never beat the raw query on this entity-rich corpus. Both
  stay in-tree, off by default, for corpora where they'd pay off.
- **The ceiling is retrieval, not the model.** With the answer in context, even a 4 B model
  answered ~88% correctly; the lost points are queries where retrieval never surfaced the
  thread. Model size was second-order.

**The compound effect — the canonical recall@5 ladder.** Each technique added one at a time,
scored on the 360 queries against hard gold labels (no LLM judge), reproducible via
`scripts/eval/bench_avc.py` + `bench_thread_reconstruction.py`:

| step | recall@5 | gain |
|------|----------|------|
| plain dense | 46% | — |
| + learned sparse | 49% | +3 |
| + contextual summary | 62% | +13 |
| + reranking | 64% | +2 |
| **+ thread reconstruction** ★ | **93%** | **+29** |

★ The last step switches from "find the exact email" to "find its *thread*" — a legitimately
easier, more useful target (thread-recall). The two biggest levers (thread reconstruction +29,
contextual summary +13) are both about understanding the conversation, not a fancier embedding
model. Same ordering on public Enron-QA; the NVIDIA dense+rerank yardstick trails on email
(57% R@5 vs the hybrid's 62%) but wins on TREC legal e-discovery — task-fit, not brand. The
value isn't any single trick; it's the disciplined stack and the rigor to prove every layer.

**Worked example.** Searching for the salon partner programme by its acronym (`"SPP"`)
mixes a semantic concept (partnership onboarding) with a rare exact token (`SPP`).
Dense-only finds the concept but ranks the literal acronym low; sparse-only finds the token
but misses paraphrases; hybrid + RRF gets both. Multi-query expansion (searching several
phrasings and fusing with RRF) further bridges acronym ↔ expansion ("SPP" ↔ "Salon Partner
Programme"), at the cost of extra queries per search.

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

## CI / quality gates

Every PR runs the following checks (all pinned to commit-SHA'd actions). Each is
an independently named status so it can be wired into branch protection:

| Gate | What it enforces | Run locally |
|------|------------------|-------------|
| `pytest` | Full test suite + a coverage floor of **85%** (current ~88%) | `poetry run python -m pytest tests/ --cov=src --cov-fail-under=85 -q` |
| `ruff (lint + format)` | Import order + pyflakes/pycodestyle (`E,F,I,W`) and formatting | `ruff check .` and `ruff format --check .` |
| `mypy (type check)` | Type-checks `src/` (lenient: `ignore_missing_imports`, per-module opt-outs for legacy modules) | `poetry run mypy src/` |
| `pip-audit (dependency CVEs)` | Known CVEs in the locked deps (OSV) | `poetry run pip-audit --vulnerability-service osv` |
| `dependency-review` | Blocks PRs that add deps with `moderate`+ advisories | (PR-only; runs on GitHub) |

Config lives in `pyproject.toml` (`[tool.ruff]`, `[tool.mypy]`) and the workflows
under `.github/workflows/` (`ci.yml`, `dependency-review.yml`, `test-suite.yml`).
`ruff check --fix .` and `ruff format .` auto-fix most lint/format findings.

Two transitive advisories with no upstream fix (`nltk` path-traversal, `torch`
`jit.script` memory corruption) are ignored by ID in `ci.yml` with links; every
*fixable* advisory was resolved by bumping the lockfile instead.

## Documentation

Full map and reading order: **[`docs/INDEX.md`](docs/INDEX.md)**. The reader journey is
**this page → quickstart → setup → deep dives**:

1. **You are here** (`README.md`) — overview, quickstart, and the case study.
2. [`docs/GUIDE.md`](docs/GUIDE.md) — the friendly walkthrough: the cleanup funnel diagram, how to pick a **persona** (budget vs quality), and what the `wizard` looks like when you run it.
3. [`docs/QUICKSTART.md`](docs/QUICKSTART.md) — 5-minute setup and copy-paste usage patterns.
4. [`docs/SETUP.md`](docs/SETUP.md) — full setup, the local `.eml` pipeline, and how to run the tests.
5. Deep dives:
   - [`docs/BACKENDS.md`](docs/BACKENDS.md) — point mailrag at the LLM / embedder / vector store of your choice (LM Studio, Ollama, vLLM, NVIDIA NIM, OpenAI, Qdrant), with the dense-only "sparse caveat".
   - [`docs/VERBS.md`](docs/VERBS.md) — the verb ladder (cost of each step) and the persona recipes; the source of truth for the CLI.
   - [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — design decisions & extension points.
   - [`docs/EMAIL_PREPROCESSING.md`](docs/EMAIL_PREPROCESSING.md) — reply-chain stripping & chunk tuning.
   - [`docs/RETRIEVAL_GUIDE.md`](docs/RETRIEVAL_GUIDE.md) — the retrieval stack end-to-end: hybrid fusion, contextual retrieval, reranking, and thread-aware *retrieval* (small→big expansion).
   - [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) — the measured findings behind the case study: cleanup economics, regex-vs-LLM, the labeled-eval ladder (§9–§13), and the corpus-portability result (§14). Start with its [terminology box](docs/EXPERIMENTS.md#terminology-read-this-first) for the `C`/`C′` labels and the two senses of "thread-aware".

Reference: [`config/community_blocklist.template.yaml`](config/community_blocklist.template.yaml) — portable starter noise rules (~1/3 of corporate-mail noise, corpus-independent).

## Roadmap

mailrag is built to be one node in a private context stack — so the next steps make it
easier for agents to reach, and keep its memory current:

- **MCP server** ([#32](https://github.com/fmasi/mailrag/issues/32)) — expose `search`/`ask`
  and attachment fetch over the Model Context Protocol, so any agent can query your mail
  without touching the internals.
- **Live ingestion** — move from one-time imports to incremental ingest of incoming mail, so
  the index stays current: a *living* context source, not a static snapshot. (The
  `EmailLoader` interface is already source-agnostic to make this clean.)
- **Guided TUI** ([#36](https://github.com/fmasi/mailrag/issues/36)) — a full-screen terminal
  UI for the cleanup pipeline (pick a persona, watch the funnel, approve the calibrate gate),
  replacing today's prompt-by-prompt flow.

## Built by Frédéric Masi

I build private, self-hosted context tools for AI agents — software that gives an agent (and me)
total recall over my own work without renting my memory to a vendor. mailrag covers email;
[parley](https://github.com/fmasi/parley) covers calls and meetings.

I care about retrieval quality you can actually measure, email and information-retrieval systems,
and engineering claims backed by numbers and honest caveats. If that's useful to you, or you're
hiring, I'd like to hear from you.

- **LinkedIn** — https://www.linkedin.com/in/fmasi/
- **GitHub** — https://github.com/fmasi

## License

[Apache 2.0](LICENSE) — see also [`NOTICE`](NOTICE). Copyright © 2026 Frederic Masi.
If you build on this work (code or method), please preserve the attribution in `NOTICE`.
