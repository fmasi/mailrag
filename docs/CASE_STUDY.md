# Case study: what the cleanup & retrieval choices actually bought

The technique-by-technique breakdown behind the headline numbers — what each step
cost, what it bought, and where the intuitive choice turned out to be wrong.

Moved out of the README so the front page stays readable; nothing has been cut.
For which figures are publicly reproducible and when each was last re-run, see
[`CLAIMS.md`](CLAIMS.md). For the full experiment log with the §-numbered ladder,
see [`EXPERIMENTS.md`](EXPERIMENTS.md).

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

