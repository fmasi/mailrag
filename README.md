# mailrag

> Ask questions of your own email — on your own hardware, on open models, with nothing
> required to leave your network.

[![Test Suite](https://github.com/fmasi/mailrag/actions/workflows/test-suite.yml/badge.svg)](https://github.com/fmasi/mailrag/actions/workflows/test-suite.yml)
![Python](https://img.shields.io/badge/python-3.11+-blue)
![License](https://img.shields.io/badge/license-Apache%202.0-blue)

## Why this exists

The first time I pointed cloud AI at my inbox it felt like a superpower — until I thought about
what it actually required: handing my entire email history to someone else's servers to make it
searchable. For real correspondence — contracts, receipts, the record of who agreed to what —
that's a non-starter.

So I built the opposite. mailrag runs on your own hardware, on open models, with nothing required
to leave your network — no mailbox upload, no vendor to trust with the whole archive.

**How sensitive is email? Look at what's public.** Nobody has ever donated their inbox. Every
public email corpus exists because someone *lost control* of a mailbox — Enron entered the record
through a federal investigation, the Avocado collection came out of a company's liquidation, the
FOIA sets came out of public-records law. There is no consented, open corpus of real
correspondence, because nobody consents. That is why a dataset from **2001** is still the field
standard twenty-five years later, and why the best available alternative is another dead company's
mail.

That scarcity *is* the argument for building it this way. If the data is too sensitive to leave
its owner — and the entire history of this field says it is — then you don't bring the mailbox to
the model. You bring the model to the mailbox.

Then the real point clicked. These aren't just emails, they're **context**. A faithful, private
record of what was said and written is exactly what an AI agent needs to be useful about *your*
work — kept private and self-owned, so you get total recall without renting your memory to anyone.
mailrag is one private context source, for **email**;
[parley](https://github.com/fmasi/parley) is another, for **calls and meetings** — different
domain, different machinery (on-device audio + diarization). They don't talk to each other; my
agents know about both and reach for whatever fits. The point was never a single app — it's a
private, open stack of context I own.

## The idea: email is conversations, not documents

A generic RAG treats every email as an isolated document, and that is the mistake. Most single
messages are unanswerable alone — *"sounds good, go ahead"* means nothing without the three
messages above it. So the unit of truth for a mailbox is the **thread**, not the message.

Three ideas carry most of the value:

1. **Match small, answer big.** Retrieve one message, then answer from its *entire* reconstructed
   conversation. It is the single biggest win in the system — and it needs no LLM at all.
2. **Hybrid retrieval.** bge-m3 dense **+ learned-sparse**, RRF-fused in Qdrant: the concept *and*
   the rare exact token that dense vectors lose — an invoice number, a ticket ID, an acronym.
3. **Local by default.** Open models, your machine, your disk. Cloud is a swappable option at two
   seams (the LLM and the embedder), never a requirement.

Around that core sit the unglamorous parts that decide whether it works on a real mailbox:
attachment extraction with OCR, reply-chain stripping, noise filtering, and continuous IMAP sync
so the index doesn't quietly rot. Details in [what's in the box](#whats-in-the-box).

## Try it

Prerequisites: Python 3.11+ and Docker (for the Qdrant container).

```bash
git clone https://github.com/fmasi/mailrag.git
cd mailrag
pip install -r requirements.txt        # includes FlagEmbedding (bge-m3); first run pulls ~2 GB of weights
cp .env.example .env                    # an LLM key/endpoint — only needed for the demo's answers
make demo                               # see the pipeline work, end to end
make bench                              # check the retrieval numbers yourself
```

**`make demo` shows you the shape of it.** It brings up Qdrant, builds a thread-aware contextual
index over 100 public Enron emails, and answers example questions by retrieving a message and
assembling its whole conversation. It is a **walkthrough, not a measurement** — no scoring, no
gold answers — and it does spend a few LLM calls on summaries and answers.

**`make bench` is the one that proves something.** It scores 360 committed queries against a
fixed 2 000-document slice of public Enron-QA and prints recall@k with confidence intervals and a
paired significance test. It spends **zero LLM calls** — no summaries, no answer generation, no
reranker — so it needs no API key and no private data. Roughly 1.6 min on an Apple-silicon GPU,
14.7 min CPU-only. The corpus manifest and query set are committed under
[`eval/public/`](eval/public/), so you can read exactly what is being scored.

Once a collection is indexed, query it from the CLI or hand it to an agent over the
[Model Context Protocol](docs/MCP_SERVER.md):

```bash
./mailrag ask "who approved the Q3 budget, and when?"
./mailrag mcp                     # stdio MCP server, read-only, multi-collection
```

The MCP server exposes seven tools — `list_collections`, `search_email`, `get_thread`,
`grep_email`, `answer_question`, `list_attachments`, `get_attachment` — so an agent can discover
your corpora, search them, pull a whole thread by id (an exact key lookup, not a search), grep for
literal needles embeddings miss, and read attachment text. See
[`docs/MCP_SERVER.md`](docs/MCP_SERVER.md) for the reference and client setup.

## What the numbers say

Two different things get measured here, and it is worth keeping them apart.

**On a real mailbox (author-reported).** On a ~32k-email corporate archive, a plain-dense baseline
finds the right message **45.6%** of the time at recall@5. Answering from the whole thread instead
of the single message takes that to **93.3%** — the metric shifts from message-level to
thread-level *on purpose*, because for a conversation the thread is the right unit of truth.
Thread reconstruction is the biggest lever (**+29.1**, from a 64.2% message-level base), just
ahead of per-email contextual summaries (**+12.8**). Neither is a fancier embedding model. The
full technique-by-technique ladder is in the
[case study](#case-study-what-the-cleanup--retrieval-choices-actually-bought), with the reasoning
written up in the [benchmark post](https://fmasi.eu/blog/email-rag-retrieval/).

These figures come from a private mailbox, so you cannot re-run them. Treat them as
author-reported.

**What you can check yourself.** `make bench`, on public Enron-QA, no key, no private data:

| arm | R@1 | R@5 | R@10 |
|---|---|---|---|
| dense only | 87.5 [83.7, 90.5] | 94.4 [91.6, 96.4] | 95.3 [92.6, 97.0] |
| **dense + learned-sparse** | **90.0 [86.5, 92.7]** | **97.5 [95.3, 98.7]** | **98.6 [96.8, 99.4]** |

Brackets are 95% Wilson score intervals. They overlap, so the benchmark also reports the **paired**
test — the right one here, since both arms answer identical queries: at R@5 learned-sparse **fixes
12 queries and breaks 1**, McNemar exact **p = 0.0034**. Run `make bench SIZE=large` and the
distractor pool grows 5×, the task gets harder, and the sparse advantage *widens* to **+4.4pp**
(p = 0.0001). That direction is the real result: learned-sparse earns more as retrieval gets
harder.

**What that benchmark covers.** The hybrid retrieval layer, and only that. Thread reconstruction,
contextual summaries, reranking and noise cleanup are all switched off, because public email
corpora do not carry the conversation structure the first two depend on, and the third needs a
paid endpoint. So `make bench` shows the foundation is sound; it does not reproduce the ladder
above. [`docs/BENCHMARK.md`](docs/BENCHMARK.md) lists every exclusion, and
[`docs/CLAIMS.md`](docs/CLAIMS.md) maps every published figure to the script that produced it and
the date it was last re-run.

**Closing the gap (in progress).** Thread reconstruction can be made publicly checkable, and the
data supports it better than expected. Measured over 19,530 real Enron messages: the corpus is an
Outlook/Exchange export, so it carries **no `In-Reply-To` or `References` headers at all** (0%) —
but 64% of messages are replies or forwards by subject, and deriving conversations from normalised
subject plus shared participants puts **50.2% of messages into a multi-message thread** (largest:
59 messages). So the conversations are there; they just have to be reconstructed rather than read
off a header. That is [#123](https://github.com/fmasi/mailrag/issues/123). A contextual-summary
arm is planned alongside it, opt-in via a local LLM so the default stays key-free
([#125](https://github.com/fmasi/mailrag/issues/125)). Reranking and the TREC comparison will stay
author-reported — one needs a paid endpoint, the other needs TREC Legal data that cannot be
redistributed.

## What's in the box

`mailrag` turns a mailbox into a queryable knowledge base, built on LlamaIndex:

- **Pluggable loaders** — local `.eml` archives and the public Enron corpus
  (HuggingFace), behind one `EmailLoader` interface.
- **Email-aware preprocessing** — reply-chain stripping, calendar-invite collapsing,
  noise/newsletter filtering, exact-text chunk dedup.
- **Thread-aware answers** (the flagship) — match a single small unit, then answer from
  its *entire* conversation. It roughly **doubles** answer coverage (terse replies 33% →
  ~80%), it's the biggest single retrieval win, and it needs no LLM.
- **Hybrid retrieval** — bge-m3 dense + sparse vectors (RRF-fused) in Qdrant. Gets
  both the concept and the rare exact token — acronyms, IDs, reference numbers.
- **Attachments, extracted and indexed** — text is pulled from PDFs, Office files
  (Word/Excel/PowerPoint), HTML and images, with OCR for scans and screenshots (local
  Tesseract, or a local vision model as the privacy-first default). Each attachment is then
  chunked by its *own* structure: spreadsheets by row-group with the header repeated in every
  chunk, PDFs by page, decks by slide. A figure buried in a 500-row sheet stays searchable
  instead of being truncated at the embedder's token limit. Chunks carry a back-reference to
  their email and thread, and agents can pull the raw text over MCP. On by default for the
  `.eml` path. OCR is optional and falls through cleanly when its reader isn't installed.
- **Local-LLM `summarize`** — optional per-email summary + noise judgement from a local
  LLM, content-addressed and cached, so re-runs are free.
- **Continuous sync** — a collection built from a backup is a snapshot that quietly rots.
  `./mailrag sync` fetches new mail from a live account (IMAP or Maildir) and indexes only
  the delta, so the index stays 1–2 days fresh instead of frozen at its export date. Point
  ids are deterministic, so re-indexing replaces rather than duplicates; the content-addressed
  cache means only *new* mail costs an LLM call. Behind a provider-agnostic seam — any
  account, any provider, any collection (see [`docs/SYNC.md`](docs/SYNC.md)).
- **Agent-ready over MCP** — `./mailrag mcp` runs a read-only, multi-collection stdio
  server with seven tools (search, exact grep, thread fetch, Q&A, attachments), so an MCP
  client such as Claude Code can use the archive as context without touching the internals
  (see [`docs/MCP_SERVER.md`](docs/MCP_SERVER.md)).
- **A measured methodology** — a 360-query retrieval eval that prices each technique,
  controls for confounds, and reports significance, overturning the intuitive choice more
  than once. It also caught its own headline overstating: an early +6pp gain was half a
  quantization artifact, worth only **+3pp** once the control re-ran at matched precision.
  Every published figure is tracked in [`docs/CLAIMS.md`](docs/CLAIMS.md) with the script
  that produces it and the date it was last verified.
- **Source-agnostic API** — `load_emails(source="enron"|"mail_archive_x")`.

## Architecture

```
                       ┌────────────────────────┐
   sources             │    EmailLoader (ABC)    │
  ┌─────────┐          ├───────────┬─────────────┤
  │  .eml   │──────────│   mail_   │    enron    │
  │  Enron  │          │  archive  │             │
  └─────────┘          └─────┬─────┴──────┬──────┘
                             │ NormalizedEmail   │
                             ▼                   ▼
            tag: regex noise filter — flag bulk/newsletters   (no LLM)
                            ▼
            summarize: local LLM — summary + noise judgement, cached  (optional)
                            ▼
            drop noise · dedup · reply-chain strip
              └ drop stage is tunable: tag = save LLM budget · summarize = best quality
                            ▼
            chunk (SentenceSplitter, bge-m3 tokenizer)
              └ optional: prepend each email's summary  → contextual retrieval
                            ▼
            embed (bge-m3 dense + sparse)
                            ▼
                    Qdrant (hybrid: named dense + sparse vectors)
                            ▼
            query engine (hybrid RRF · thread-aware expansion · optional rerank)
```

The `tag` step only *tags* by default, so nothing is lost before the LLM sees it; the
confident drop happens at `summarize` (the LLM pass). Where you drop is a deliberate
budget-vs-quality knob — drop at `tag` to skip the LLM cost, or at `summarize` for the
cleaner result. *(These map onto the eval labels used in the case study below and in
[`EXPERIMENTS.md`](docs/EXPERIMENTS.md): **Pass-1 = `tag`**, **Pass-2 = `summarize`**.)*

Between the two there's an optional, **no-LLM** triage: `./mailrag scan` clusters the
corpus embeddings at thread level and ranks the densest "noise pockets" (bulk and
automated mail) by `tag` enrichment, sender concentration, and tightness. It spends
no LLM budget — it reuses the already-embedded vectors when a collection exists, else
embeds once — and writes a JSON artifact (thread → `.eml` paths) so you can see where the
noise concentrates before deciding how much of the `summarize` pass to run.

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
| **+ thread reconstruction** (pull the full conversation of each top hit) | **message-level recall@5 64.2% → thread-level 93.3%** — match a small unit, answer from its whole thread | larger context per query (tunable: expand top-N threads) |

**How the eval was run.** The eval set is **360 synthetic queries** (144 terse / 144 content /
72 spanning), each generated from a known email so the **recall ladder is scored against hard
gold labels — no LLM judge in the loop**. Generating a query from its answer email risks
circularity (the query could echo the target's exact tokens), so the queries come from body
content under a rule that bans artifact/metadata questions plus a validation pass, and the
load-bearing guard is external: the ordering holds on public **Enron-QA** (questions written
independently of this generator) and on TREC's real human judgments, below. A *separate*
answer-quality lens does use a local LLM judge, calibrated against a stronger reference
model on 514 pooled pairs (Cohen's κ = **0.52** on the 0–3 scale, Spearman 0.74 — and,
the decisive check, both pre-registered decisions came out identical under both judges).
Significance tests and confound controls are in
[`EXPERIMENTS.md` §9–§13](docs/EXPERIMENTS.md#9-labeled-eval--retrieval-metrics-coverage-and-end-to-end-answer-quality-2026-05-29):

- **Thread reconstruction is the biggest single win — and needs no LLM.** Matching a small unit
  and returning its whole conversation lifts **recall@5 from 64.2% (message-level) → 93.3%
  (thread-level)** (+29.1) — it trades "find the needle" for "find the right thread," which the
  conversation then answers. The target shifts from one message to its thread by design: for a
  conversation, the thread is the right unit of truth.
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
- **The ceiling is retrieval, not the model.** With the answer in context the answer model
  was right ~88% of the time, and at matched precision a 4 B model essentially tied a
  6×-larger one; the lost points are queries where retrieval never surfaced the thread.
  Model size was second-order.

**The compound effect — the canonical recall@5 ladder.** Each technique added one at a time,
scored on the 360 queries against hard gold labels (no LLM judge), reproducible via
`scripts/eval/bench_avc.py` + `bench_thread_reconstruction.py`:

| step | recall@5 | gain |
|------|----------|------|
| plain dense | 45.6% | — |
| + learned sparse | 48.9% | +3.3 |
| + contextual summary | 61.7% | +12.8 |
| + reranking | 64.2% | +2.5 |
| **+ thread reconstruction** ★ | **93.3%** | **+29.1** |

★ The last step switches from "find the exact email" to "find its *thread*" — a legitimately
easier, more useful target (thread-recall). The two biggest levers (thread reconstruction +29,
contextual summary +13) are both about understanding the conversation, not a fancier embedding
model. Same ordering on public Enron-QA. The NVIDIA dense+rerank yardstick trails on email
(57% R@5 vs the reranked hybrid's 64% — a like-for-like, both-reranked comparison) but wins on
TREC legal e-discovery — task-fit, not brand. The
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
| `src/data/loaders/` | Pluggable source loaders (mail_archive_x, enron) |
| `src/ingest/` | Embedding (bge-m3), sparse vectors, hybrid Qdrant upsert |
| `src/indexing/` | Index creation/management + structure-aware attachment chunking |
| `src/attachments/` | Attachment extraction (PDF/Office/HTML/image) + OCR (Tesseract / local vision model) |
| `src/query/` | Retrieval + RAG query engine |
| `src/mcp_server/` | Multi-collection MCP (stdio) server: discovery, search, grep, threads, answer, attachments |
| `src/llm/` | Optional LLM `summarize` pass (per-email summary + noise verdict) + cache |
| `src/sync/` | Continuous sync: provider-agnostic `MessageSource` seam (IMAP / Maildir), state, scheduling |
| `src/pipeline/` | The CLI verbs' pipeline stages (tag, scan, calibrate, judge, index, …) |
| `src/persona/` | Persona recipes: named budget-vs-quality presets for `run` / `wizard` |
| `src/tui/` | The full-screen `wizard` (Textual) |
| `src/cluster/` | `scan`: embedding-cluster noise-pocket triage (no LLM) |
| `src/eval/` | Pure-logic eval modules behind `scripts/eval/` |
| `scripts/` | Build / index / eval / maintenance utilities |
| `tests/` | Test suite (pytest, ~1,500 tests) |
| `docs/` | Architecture, quickstart, preprocessing guides |

## CI / quality gates

Every PR runs these checks. Two are **required** branch-protection gates; the rest
are advisory signals. The actions in our workflow files are pinned to commit SHAs
(CodeQL is the exception — it runs via GitHub's managed default setup, with no
workflow file to pin):

| Gate | Required? | What it enforces | Run locally |
|------|-----------|------------------|-------------|
| `pytest` | ✅ required | Full test suite (~1,500 tests) + a coverage floor of **85%** (currently ~88%) | `poetry run python -m pytest tests/ --cov=src --cov-fail-under=85 -q` |
| `CodeQL` | ✅ required | Static security analysis — GitHub **default setup** (managed, no workflow file) | (runs on GitHub) |
| `ruff (lint + format)` | advisory | Import order + pyflakes/pycodestyle (`E,F,I,W`) and formatting | `ruff check .` and `ruff format --check .` |
| `mypy (type check)` | advisory | Type-checks all of `src/`, including the bodies of unannotated functions (`check_untyped_defs`). No per-module opt-outs. Lenient only about third-party imports (`ignore_missing_imports`; CI runs deps-free so they resolve to `Any` and results stay deterministic) | `poetry run mypy src/` |
| `pip-audit (dependency CVEs)` | advisory | Known CVEs in the locked deps (OSV) — with **zero** `--ignore-vuln` entries | `poetry run pip-audit --vulnerability-service osv` |
| `dependency-review` | advisory | Blocks PRs that add deps with `moderate`+ advisories | (PR-only; runs on GitHub) |
| Claude review | advisory | Automated PR review + `@claude` mentions (`claude.yml`, `claude-code-review.yml`; skipped until the app token is set) | (runs on GitHub) |

Config lives in `pyproject.toml` (`[tool.ruff]`, `[tool.mypy]`) and the workflows
under `.github/workflows/` (`ci.yml`, `test-suite.yml`, `dependency-review.yml`,
`claude.yml`, `claude-code-review.yml`); CodeQL has no file (GitHub default setup).
`ruff check --fix .` and `ruff format .` auto-fix most lint/format findings.

Supply-chain state at the time of writing: **zero open Dependabot alerts** and a
pip-audit with **no ignore entries** — every advisory that reached us is resolved by
constraint floors in `pyproject.toml`, not waived. The Qdrant server image is pinned
by digest (not `:latest`), and `qdrant-client` is deliberately capped `<1.19`
because that release removed a symbol `llama-index-vector-stores-qdrant` still
imports ([#106](https://github.com/fmasi/mailrag/issues/106) tracks lifting it).

## Documentation

Full map and reading order: **[`docs/INDEX.md`](docs/INDEX.md)**. The reader journey is
**this page → quickstart → setup → deep dives**:

1. **You are here** (`README.md`) — overview, quickstart, and the case study.
2. [`docs/GUIDE.md`](docs/GUIDE.md) — the friendly walkthrough: the cleanup funnel diagram, how to pick a **persona** (budget vs quality), and what the `wizard` looks like when you run it.
3. [`docs/QUICKSTART.md`](docs/QUICKSTART.md) — 5-minute setup and copy-paste usage patterns.
4. [`docs/SETUP.md`](docs/SETUP.md) — full setup, the local `.eml` pipeline, and how to run the tests.
5. Deep dives:
   - [`docs/BACKENDS.md`](docs/BACKENDS.md) — point mailrag at the LLM / embedder / vector store of your choice (LM Studio, Ollama, vLLM, NVIDIA NIM, OpenAI, Qdrant), with the dense-only "sparse caveat".
   - [`docs/VERBS.md`](docs/VERBS.md) — the CLI source of truth: every verb (including `ask` and `mcp`), the cost-ordered ladder, the alias table, and the persona recipes.
   - [`docs/SYNC.md`](docs/SYNC.md) — continuous sync: account config, secret references, folder *roles* (so "everything but junk" means the same on every provider), what happens when the network / LLM / Qdrant is down, launchd & systemd scheduling, and how to add a provider.
   - [`docs/MCP_SERVER.md`](docs/MCP_SERVER.md) — the multi-collection stdio MCP server: the seven tools (`list_collections` / `search_email` / `get_thread` / `grep_email` / `answer_question` / `list_attachments` / `get_attachment`), collection discovery & selection, config, client setup (Claude Code / opencode), and the CLI↔MCP capability matrix.
   - [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — design decisions & extension points.
   - [`docs/EMAIL_PREPROCESSING.md`](docs/EMAIL_PREPROCESSING.md) — reply-chain stripping & chunk tuning.
   - [`docs/CHUNKING.md`](docs/CHUNKING.md) — how one email becomes two kinds of chunk: a body chunk with its summary baked into the vector, and summary-free attachment chunks split by their own structure (spreadsheet rows, PDF pages, slides), stitched back together at query time by `thread_id`.
   - [`docs/RETRIEVAL_GUIDE.md`](docs/RETRIEVAL_GUIDE.md) — the retrieval stack end-to-end: hybrid fusion, contextual retrieval, reranking, and thread-aware *retrieval* (small→big expansion).
   - [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) — the measured findings behind the case study: cleanup economics, regex-vs-LLM, the labeled-eval ladder (§9–§13), and the corpus-portability result (§14). Start with its [terminology box](docs/EXPERIMENTS.md#terminology-read-this-first) for the `C`/`C′` labels and the two senses of "thread-aware".

Reference: [`config/community_blocklist.template.yaml`](config/community_blocklist.template.yaml) — portable starter noise rules (~1/3 of corporate-mail noise, corpus-independent).

## Status

mailrag is built to be one node in a private context stack — reachable by agents, with a
memory that stays current. The three pillars that make that true have all shipped:

- **MCP server** ([#67](https://github.com/fmasi/mailrag/pull/67), [#74](https://github.com/fmasi/mailrag/pull/74)) — a single,
  multi-collection stdio server (MCP SDK v2) exposing the full query/read surface over the
  Model Context Protocol — seven tools spanning discovery, hybrid search, exact thread
  fetch, raw-corpus grep, Q&A and attachments — so any agent can discover, query and read
  your mail without touching the internals (see
  [`docs/MCP_SERVER.md`](docs/MCP_SERVER.md)).
- **Live ingestion** ([#101](https://github.com/fmasi/mailrag/issues/101)) —
  `./mailrag sync` moves mailrag from one-time imports to incremental ingest, so the index is a
  *living* context source rather than a static snapshot. New mail is fetched from any
  configured account, spooled as `.eml`, and run through the *same* pipeline — so the cleaning
  rubric stays consistent and only new mail costs an LLM call. Providers sit behind a
  `MessageSource` seam with opaque cursors (IMAP UID/MODSEQ today; Gmail `historyId`, JMAP and
  Graph delta tokens need no schema change), and each stage degrades independently: mail is
  still fetched when the LLM is down, still judged when Qdrant is down (see
  [`docs/SYNC.md`](docs/SYNC.md)).
- **Guided TUI** ([#36](https://github.com/fmasi/mailrag/issues/36)) —
  `./mailrag wizard` is a full-screen terminal app ([Textual](https://textual.textualize.io/)):
  pick a persona, scope folders on a tree, review the plan, and watch the run live — with the
  calibrate and confirm-before-spend gates as dialogs (see
  [`docs/GUIDE.md`](docs/GUIDE.md#what-to-expect-from-the-wizard)). The old prompt flow
  remains as `--classic`; `./mailrag run` stays the headless path.

  ![The mailrag wizard persona picker — a cost-ordered persona list on the left and a live preview of the highlighted recipe with colour-coded cost badges on the right](docs/images/tui/persona.svg)

  *The persona picker; the [full six-screen walkthrough](docs/GUIDE.md#what-to-expect-from-the-wizard) lives in the guide. Screenshots are auto-generated via `scripts/gen_tui_screenshots.py`.*

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
