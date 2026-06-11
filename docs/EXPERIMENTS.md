# Experiments & findings

*[← docs index](INDEX.md) · [README](../README.md) · related: [`RETRIEVAL_GUIDE.md`](RETRIEVAL_GUIDE.md) (the live stack), [`EMAIL_PREPROCESSING.md`](EMAIL_PREPROCESSING.md) (cleanup mechanics)*

A running log of what we measured building `mailrag` against a real ~32,000-email
corporate mailbox (all references anonymized). The point is to document the *why* and
the *trade-offs* with real numbers — including the places where the obvious assumption
turned out wrong.

> **Corpus:** 70,016 exported emails → **31,969** selected (work-account folders).
> All identifying names replaced with placeholders (`SPP` = a salon partner
> programme; "the work domain" = the employer's own email domain).

---

## Terminology (read this first)

This log was written incrementally and uses two pieces of shorthand. They are defined
once here and used consistently throughout.

**Collection labels (`C`, `C′`).** Early sections (§6–§12) compare two collections by a
short label; §13 onward names the live collections directly. They map as follows:

| label | what it is | live collection name |
|-------|------------|----------------------|
| `C`  | cleaned body-only collection (no embedded summary) | `work-rag-bodyonly` |
| `C′` | contextual-retrieval collection: each email's LLM **summary is prepended before embedding** | `work-rag-ctx-iso-8bit` (isolated summary) → `work-rag-ctx-threadaware` (preceding-context summary, **prod default**); `work-rag-ctx-whole-8bit` is the whole-thread-summary variant |

So "keep `C′`" means "keep the summary-embedded contextual collection"; the production
collection that realises it is `work-rag-ctx-threadaware`.

**"Thread-aware" means two distinct things** — disambiguated explicitly wherever it appears:

1. **Thread-aware *retrieval*** (a.k.a. *small→big* / *thread expansion*, §8) — a **retrieval**
   step: match any one unit in a thread, then return/answer from the **whole thread**. No LLM.
2. **Thread-aware *summary*** (a.k.a. *preceding-context summary*, §13) — a **summary-conditioning**
   step: each email's embedded summary is written with its *preceding* thread messages as context
   (causal, append-only), rather than in isolation. Uses the LLM.

The `work-rag-ctx-threadaware` collection is named for sense (2) — it stores thread-aware
*summaries*. Sense (1), thread-aware *retrieval*, is a query-time expansion applied on top of
**any** collection.

---

## 1. The cleanup funnel — measured savings

| stage | what it does | effect |
|------|--------------|--------|
| Scope | keep only work-account folders | 70,016 → **31,969** |
| Pass-1 (regex) | cheap sender/subject rules, *before* any embedding | flags **10.4%** (3,332) standalone |
| Pass-2 (local LLM) | summarize + judge each email | flags **37.9%** (12,123) noise |
| Calendar-collapse + chunk-dedup | 1-line calendar summaries; drop byte-identical chunks | 22,613 → **21,590** chunks |
| **Net** | | 31,969 emails → **19,859 kept** → 21,590 chunks |

The big embedding-time win — a first-run estimate of **~48 h → under 10 min** — came from
the **inference method** (FlagEmbedding on Apple-Silicon MPS) plus volume reduction, **not**
from the LLM. Keep that separate from the cleaning question below.

---

## 2. How much of the cleaning actually needed an LLM?

**First (wrong) answer:** keyword-matching the LLM's *reason text* suggested ~98% of its
noise was "automated, regex-catchable." That was a post-hoc artifact — it never proved a
rule would fire.

**Tested answer.** We derived regex rules *from the corpus* (sender domains that are
≥90% noise, + calendar/out-of-office subject patterns) and compared to the LLM's labels:

| | count | |
|---|---|---|
| caught (rules ∩ LLM-noise) | 7,875 | **65%** of LLM noise |
| missed (LLM-only) | 4,248 | **35%** — the LLM's *unique* catch |
| false-drop (genuine mail rules would remove) | 120 | the quality cost |

**Why rules can't reach the LLM — mixed-sender domains.** The work domain itself is
**29% noise** (≈24k emails interleaving real correspondence with compliance reminders,
calendar churn, internal newsletters). You can't write a sender rule for your own domain,
and the noise inside it isn't cleanly separable by sender/subject. That interleaved tail
is what only per-email judgment can clean.

**Lesson:** use cheap regex for the obvious bulk; reserve the LLM for the interleaved
noise *and* the summaries (§5–6). Rule **discovery** itself never needed a full pass — a
sample or a sender-frequency table reveals the obvious senders (§3).

---

## 3. Does regex *harm* (drop genuine mail), or just miss noise? → both, but small & tunable

Sweeping the domain-noise threshold:

| threshold | rules | catches | false-drops (genuine mail) |
|---|---|---|---|
| 0.70 | 86 | 67.0% | 193 |
| 0.90 | 79 | 65.0% | 120 |
| 0.95 | 74 | 64.0% | 109 |

So it's not pure incompleteness — there *is* a misclassification risk (~0.4%), and it's a
precision/recall knob. The false-drops cluster on **mixed channels** — domains that are
mostly automated but occasionally carry a real message (a recruiter via LinkedIn, a chat
"X sent a message", a doc-update notification referencing real work). A blanket domain
rule can't tell those apart; the LLM can, because it reads the message. (Amusingly, a few
"false-drops" were regex being *more* correct than a lenient LLM — e.g. auto-replies.)

---

## 4. Portability — a shared starter blocklist

Of the noise, **31.8%** comes from **71 "pure-notification" domains** (≥98% noise) that
are entirely corpus-independent: LinkedIn, Zoom, DocuSign, SharePoint, Notion, Salesforce,
Navan, RingCentral, KnowBe4, … — the SaaS/notification senders in *every* corporate inbox.

That means a brand-new user gets **~1/3 of the cleanup for free** by reusing a shared list,
with zero LLM and zero knowledge of their own corpus — much like shared spam blocklists. A
curated, anonymized starter set ships as
[`config/community_blocklist.template.yaml`](../config/community_blocklist.template.yaml).
(The other ~68% — calendar churn, internal mixed-domain mail — is corpus-specific.)

---

## 5. Targeted LLM — spend the budget where it matters

Idea: decide the confident cases by cheap stats, send only the uncertain ones to the LLM.

| strategy | LLM needed | errors vs full-LLM |
|---|---|---|
| auto-decide domains ≤5% / ≥95% noise | **82%** of corpus | 0.23% |
| + subject-noise patterns on the uncertain band | **68%** of corpus | 0.54% |

Domain-ratio alone barely helps (82% still needs the LLM) because **75% of the mail is one
mixed domain** that lands in the uncertain band. Adding a **subject-level signal** cuts the
LLM budget to 68% at a small accuracy cost — a finer signal is the lever. And for rule
*discovery* (not full labeling), a sampled LLM run is plenty: the full pass was overkill
for building filters; its payoff is per-email labels + summaries.

---

## 6. Retrieval methodology — what each technique adds (and its trade-off)

Compared four collections built from the same corpus (`A` original dense-only/uncleaned,
`B` hybrid+cleaned, `C` = B + LLM noise-removal + payload summaries, `C′` = C with summaries
*embedded*):

| technique | adds | trade-off (observed) |
|---|---|---|
| dense only (A) | semantic / paraphrase match | misses rare exact tokens; redundant near-dup chunks |
| + learned sparse + RRF (B) | exact-token / acronym precision | needs sparse-capable embedder + fusion; more storage |
| + LLM noise removal (C) | precision — junk can't surface (≈1,000 spam-quarantine digests, ≈1,500 calendar notices gone) | one-time LLM cost (§2) |
| + contextual retrieval (C′): prepend each email's summary before embedding | short/terse emails match by *gist* — a 43-char reply surfaced via its summary | **topic drift**: dilutes literal matches, pulls in adjacent-but-off results |
| + cross-encoder reranker (measured §7) | reorders candidates — clear win on content/literal queries, removes C′ drift | per-query latency; can *demote* contentless terse emails (reads the empty body) |

Key honest finding: **C vs C′ is a real trade-off, not a free win.** Contextual embedding
helps intent/short-email queries but *hurt* a precise technical query (it pulled in
off-topic training webinars). Contextual retrieval is meant to be paired with a reranker —
cast a wide net, then rerank to drop the drift. **(Updated by §7:** once measured, the cleaner
resolution turned out to be *thread-aware retrieval over a single collection `C`* — likely
retiring `C′` — rather than pairing C′ with a reranker. Read §7 for the verdict.)

**Worked example.** Searching for the salon partner programme by its acronym (`SPP`)
mixes a *semantic* concept (partnership onboarding) with a *rare exact token* (`SPP`):
dense-only ranks the literal token low, sparse-only misses paraphrases, **hybrid + RRF gets
both**; multi-query expansion bridges acronym ↔ expansion at the cost of extra queries.

---

## 7. Cross-encoder reranking & terse-email recall — measured (2026-05-28)

Ran `bge-reranker-v2-m3` as an opt-in stage over the hybrid candidates (LlamaIndex-native),
plus an 8-config matrix to test the one case contextual retrieval (C′) is *designed* for:
**terse, contentless emails**.

**Setup.** Two probes: (a) a 3-way mode comparison (dense → hybrid → hybrid+rerank) on 12
content-rich queries; (b) an 8-config matrix — `{C, C′} × {hybrid, +rerank(body),
+rerank(summary+body)} × fetch {20, 50}` — on 8 deliberately terse replies (one-word bodies
like "Done" / "Tks" / "+Name" on partner/program threads), measuring recall@10 and the rank
of the exact target email (keyed on Message-ID). Queries were drafted from subjects/threads,
never the summaries, to avoid biasing toward C′.

**Content / literal queries — reranking is a clear, consistent win.** Biggest where the query
terms are *not* in the subject line: relevant-in-top-5 went `0 → 2 → 5` (dense → hybrid →
hybrid+rerank) on several queries. On well-named entity queries every mode already finds
on-topic mail, but rerank fixes the *order* — surfacing the precisely-relevant thread above
generic same-topic mail. This is the fix for C′'s drift.

**Terse emails — the opposite story, and the more interesting one:**
- Recall wasn't the crisis expected: plain hybrid found 7/8 terse targets, because the
  *subjects* (`RE: <topic>`) carry the query terms even when the body is "Tks".
- **C′ (summary embedded) ranks the terse target far higher than C** — e.g. rank `7→1`,
  `5→1`, `10→2`, `9→4`. Contextual retrieval *does* deliver on its designed purpose; the
  content-query probe simply couldn't show it.
- **Body-only reranking can *hurt* terse emails:** the cross-encoder reads the empty body and
  scores it low — one target fell from rank 5 out of the top-10 entirely.
- **Summary-aware reranking** (feed the cross-encoder `summary + body` instead of body) was
  *not* a clear win — comparable to body-rerank on terse, and on content it shifted results
  (~0.34 top-5 overlap with body-rerank): sometimes more diverse, sometimes mildly drifting
  toward generic topics (the summary is broader than the body).
- **Thread-sibling effect:** terse replies compete with the *substantive* emails on the same
  thread/topic; the reranker legitimately prefers those, and a high-frequency topic (~900
  emails) buried its terse reply entirely. So "exact terse-email recall" is partly the wrong
  lens — the information is reachable via the substantive siblings.

**Honest verdict — no single config wins everything, but the fix is probably *architectural*,
not a second collection.** C′ wins terse/recall (its designed purpose); C + reranker wins
literal/technical precision, where C′ *hurts* via drift. Rather than maintain two collections
plus a query router, the cleaner resolution is **thread-aware retrieval over a single
collection**: a terse reply never needs to be found *in isolation* — it lives in a thread, and
pulling the thread (via its substantive emails) covers the topic. This is the classic
**parent-document / thread-reconstruction** pattern, and `thread_id` is already on every email.
That likely lets us **retire C′** and keep one collection (`C`) + reranker, with summaries as
*payload* (not embedded). It also subsumes the duplicate-results issue (#2) — grouping by
thread *is* the dedup. A terse reply that is the *only* match for a topic is low-value anyway,
so missing it in isolation is an acceptable price.

To validate next: (a) does pulling the thread actually surface the terse email's info; (b)
thread-size bounding for long threads — likely an LLM-generated thread summary and/or breaking
long threads into parent-id segments (its own research thread). Directional only — 8 hand-picked
terse queries, not proof; a labelled eval would quantify the trade-off.

---

## 8. Thread-aware retrieval — design and implementation (2026-05-29)

### Motivation

About **9.6% of emails in the kept corpus** are terse-but-valid short replies: one-line
acknowledgements, brief sign-offs, forwarding notes. These embed poorly in isolation — the
dense retriever has almost no semantic signal to latch onto, and the sparse side sees only
high-frequency function tokens. The §7 matrix showed that contextual retrieval (C′) rescues
them by prepending a summary before embedding, but at a measurable cost: **literal and
technical queries drift** as the summary dilutes the exact-token signal. That is a
bi-directional trade-off, not a free win, and it is the finding that drove the architectural
shift here.

### Design

The resolution is **parent-document / small-to-big retrieval**, applied at the thread level:

- Keep **one collection** (`C`) with summaries as payload (not embedded) and one reranker.
- After hybrid retrieve + optional rerank, group the top hits by `thread_id` and fetch all
  emails in each matched thread directly from Qdrant (a scroll + filter per thread).
- **Match on small units, answer from the thread.** A terse reply lives in a thread with
  substantive emails; those substantive siblings are the ones the retriever finds, and pulling
  the whole thread exposes the terse reply as context without needing to embed it well.

This is implemented in `src/query/thread_expand.py` and wired into `HybridSearcher` via
`search_threads()`. Multi-chunk emails (body split across several Qdrant points) are
reconstructed by joining their body chunks (best-effort order — there is no `chunk_index`
field yet; adding one at ingest is a filed follow-up). Each email is rendered with an
explicit From/To/Cc/Date header so the LLM can attribute who said what.

### Key implementation facts

- **93.3% of emails produce a single chunk**; 6.7% span multiple chunks and require
  reconstruction. The body-rejoining step handles both cases uniformly.
- **`message_id` deduplication** happens inside the expansion step: each unique `message_id`
  contributes exactly one `ThreadEmail`, regardless of how many chunks matched it. This
  subsumes the standalone dedup work item from §7.
- **`thread_id` is already on every stored payload** — no re-indexing needed; the feature
  is a pure query-time operation.
- A `bound_thread` helper (off by default) accepts a `max_tokens` limit and a pluggable
  summarizer for environments where very large threads would overflow a small context window.

### Verdict

Thread-aware retrieval **retires C′**: terse replies are reached via their thread siblings,
not by embedding tricks. The single collection (`C`) + reranker architecture is cleaner —
no routing logic, no second index to maintain — and the dedup issue from §7 is resolved as
a side effect of grouping. Confirmed directionally correct; a labelled eval would quantify
recall for the terse-reply subset now that the full thread is surfaced.

---

## 9. Labeled eval — retrieval metrics, coverage, and end-to-end answer quality (2026-05-29)

Turned §6–§8's directional reads into measured numbers. **45 synthetic-from-corpus
queries** (LLM-generated from thread *bodies*, never subjects/summaries, to avoid biasing
toward C′; hypothesis-weighted ~40% terse / ~40% content / ~20% thread-spanning, each with
a known answer-email target). Harness: `scripts/eval/{gen_queries,run_arms,judge,calibrate,
report,sweep_thread_n,e2e_context,e2e_answer}.py` + pure-logic modules `src/eval/`. All raw
artifacts are gitignored under `eval/out/`; only aggregate numbers appear here.

**Judge calibration (local Gemma-31B vs an Opus reference on 514 pooled pairs):** Cohen's
κ = 0.52 (moderate), Spearman = 0.74 (strong rank agreement), and — the decisive check —
**both pre-registered decisions came out identical under both judges**, so local judging is
decision-adequate (route: all-local).

### Lens 1 — ranked-list metrics (LLM-judge pooled, flatten-to-emails, 5 arms)

nDCG@10: **C′ 0.851 > C 0.804 > C′+rerank 0.665 > C+rerank 0.628 > C+rerank+thread 0.419.**
Two surprises that *overturn* the §7 eyeballing:
- **Reranking HURT every category.** The cross-encoder demoted answer-bearing emails the
  LLM-judge rated relevant (concrete case: a query's top-8 grades went `3 3 3 3 0 3 1 0` →
  `1 0 3 2 0 0 1 0` after rerank). It optimizes query↔body similarity; the judge rewards
  answer content — they disagree. **⇒ rerank off by default.**
- **C′ (embedded summaries) was the *best* ranked arm**, including on content queries — the
  §6 "C′ drifts on literal queries" worry did not reproduce under LLM-judge.
- Thread-aware scored *worst* here — but this is the wrong ruler for it (see Lens 2): it
  returns whole threads in chronological order (~44 emails), not a relevance-ranked top-10.

### Lens 2 — answer-coverage (does the known answer email get returned at all; no judging)

Pulling the full thread roughly **doubles** answer-coverage. Critically, the original
thread arm was seeded from the *rerank'd* (now-known-worst) hits; re-measured on clean seeds:

| arm | answer found | terse |
|---|---|---|
| C / C′ (no thread) | 44% / 42% | 33% / 33% |
| C+rerank+thread (rerank'd seeds) | 58% | 50% |
| **C+thread (clean)** | **82%** | **78%** |
| **C′+thread (clean)** | **84%** | **83%** |

Thread-expansion **top-N sweep** (coverage / avg emails): sharp diminishing returns — most
of the win is in the first ~3 threads. C+thread N=1 56% (8 emails) / N=3 71% (28) / all 82%
(113); C′+thread N=1 **67%** / N=3 76% / all 84%. C′ ranks the answer-thread higher, so it
wins at *tight* budgets (N=1: 67% vs 56%); the two converge by N≈5.

### Lens 3 — end-to-end answer quality (the arbiter)

Fed each setup's retrieved context to an answer model; an Opus judge graded the answer's
correctness (0–3) against the **full source thread** as ground truth. Two answer models;
mean grade (higher = better):

| context setup | 26B@4bit | e4b@128k |
|---|---|---|
| no context | 0.00 | 0.00 |
| answer email only | 1.27 | 1.38 |
| plain C (~10 emails) | 1.69 | 1.93 |
| C + 1 thread | 1.58 | 1.62 |
| C + 3 threads | 1.78 | 1.76 |
| C + all (~113) | 1.76 | 1.95 |
| **C′ + 1 thread** | 1.89 | **2.02** |
| C′ + 3 threads | 1.89 | 1.82 |
| **C′ + all** | 1.96 | **2.09** |

terse-only: e4b@128k **C′+1thread = 2.24** vs 26B 1.72.

Findings:
- **The answer model is *not* the bottleneck — retrieval coverage is.** When the answer was
  present in context the 26B answered correctly ≈88% of the time (67% correct ÷ 76% coverage);
  most lost points are questions where retrieval never surfaced the answer thread (~24%).
- **A 4B model at its full window matches or beats the 26B almost everywhere** (`C′+all`
  e4b 2.09 vs 26B 1.96; `C′+1thread` 2.02 vs 1.89). Bigger model buys little; better
  retrieval is the lever.
- **Handing the AI only the pinpoint answer email (1.27/1.38) is the *worst* real option** —
  worse than any thread context. A terse answer email alone is insufficient; the surrounding
  thread is needed. (Sanity floor: no-context = 0.00.)
- **More context did not confuse the model** (no "lost-in-the-middle" collapse 1→3→all);
  diminishing returns, not damage.
- **C′ beats C at every depth** for both models — embedded summaries earn their keep by
  ranking the answer thread into a tight context window.

### Verdict / recommended stack
- **`C′` + expand the top ~1–3 threads, reranker OFF.** Sweet spot: **C′ + top-1 thread**
  (~2k tokens) ≈ best quality at smallest context; `C′+all` is marginally higher (2.09 vs
  2.02) at ~15× the context — not worth it.
- **Keep C′** — the end-to-end arbiter says its summaries improve answers; the §7/§8 lean
  toward retiring it was based on a ranked-list lens that mis-models thread delivery.
- **A small model (e4b-class) appears sufficient** — but this is **quantization-confounded**
  later de-confounded by a same-context 3-way run (see the model/quant note below): at *fair*
  precision the 4B essentially **ties** the 6×-larger model, and model choice is second-order
  to retrieval here.
- **The ceiling (~62–69% correct) is retrieval coverage (~76%)** → see #11–#14.

### Caveats & a practical gotcha
- Directional: 45 queries, single Opus judge (±~0.1 run-to-run noise), one quant per model.
- **Always load the model at its full context window.** LM Studio silently loaded e4b at
  24k (max 128k); the big-context setups *overflowed* → empty answers graded 0
  (`C′+all` read 0.44). Reloaded at 128k via `lms load -c 131072`, the same setup scored
  **2.09**. The "small models can't do thread-aware" reading was entirely this artifact.

### Model size / quantization / speed (de-confounded, MLX on a 48GB Mac)

The first end-to-end run compared e4b@**8bit** vs 26b@**4bit** — a 2× precision gap. A
same-context (128k) 3-way re-run de-confounds size from quant. Mean answer-correctness (0–3)
across the 7 retrieval setups, plus the speed/memory benchmark
(`scripts/eval/bench_models.py`, LM Studio native stats):

| model | quant | RAM | gen tok/s | TTFT | end-to-end quality (overall) | terse (best setup) |
|---|---|---|---|---|---|---|
| e4b | 8bit | **9.0 GB** | ~50 | **~1.0 s** | **1.88** | **2.24** |
| 26b-a4b (MoE) | 4bit | 15.6 GB | **~75** | ~1.9 s | 1.79 | 1.72 |
| 26b-a4b (MoE) | 6bit | 21.8 GB | ~60 | ~2.2 s | **1.91** | 2.11 |
| 31b (dense) | 4bit | 28.9 GB | ~9.6 ⚠️ | ~10 s | — (too slow / OOMs on long ctx) | — |

Findings:
- **Quant ≈ size, in effect.** 4→6 bit on the *same* 26B lifted quality **+0.12** (1.79→1.91)
  — about the magnitude of the entire 6× size jump. The original "e4b > 26B" was *real but
  partly a 4-bit handicap*: at fair precision **26b@6bit (1.91) ties e4b@8bit (1.88)** (~noise).
- **e4b@8bit wins the hardest case (terse, 2.24)** and the quality-per-resource frontier: it
  matches the 6×-larger model at **<½ the RAM** and ~2× better latency.
- **More bits = more RAM + slower + slightly better** (26b@6bit 21.8 GB/60 tok/s vs 26b@4bit
  15.6 GB/75 tok/s) — the three move together. The 31B *dense* is impractical here (9.6 tok/s,
  OOMs on long context).
- **This task is retrieval-gated, not intelligence-gated** — all usable models cluster at
  ~1.8–1.9 because the ceiling is *whether the answer is in the retrieved context*, not model
  smarts. A harder, reasoning-heavy task might separate them. **For this use case, e4b wins;
  reach for 26b@6bit only if a future task actually needs the intelligence.**

**Net: model choice is second-order; the lever is retrieval coverage (#12).**

---

## 10. Coverage-miss diagnostic — why the answer thread isn't retrieved (#12, 2026-05-30)

§9 located the ceiling at **retrieval coverage** (~76% on C′ + top-3 threads): for ~24% of
queries the answer-bearing thread never reaches the answer model. This diagnostic traces
*where* each gold thread sinks and *why*. Method: for all 45 labeled queries, find the best
rank of the gold thread under **dense-only, sparse-only, and hybrid** retrieval on C and C′
(deep top-200), bucket the cause, and on the hard core run an **oracle escalation** —
re-query with the gold email's own subject/body to test whether the email is retrievable *at
all*. The classification logic is unit-tested (`src/eval/coverage_diag.py`); driver is
`scripts/eval/diagnose_coverage.py`. The diagnostic reproduces the §9 number exactly
(covered 34/45 = 76%), which validates the buckets.

**Cause split (C′, top-3 threads, n=45):**

| Bucket | n | % | meaning | lever |
|---|---|---|---|---|
| covered | 34 | 76% | gold thread in top-3 | — |
| budget | 3 | 7% | in the top-10 pool but past the 3rd thread | widen N / rerank |
| fusion | 2 | 4% | dense **or** sparse ranks it well; RRF buries it | fusion tuning |
| hard | 6 | 13% | deep/absent in **both** single modes | representation |

- **The ceiling is a query→document *matching* problem, not an indexing/chunking one.** All 6
  hard misses are `vocab_gap`; **zero** are `index_or_chunking`. The oracle is unambiguous:
  querying with the gold email's own *body* surfaces its thread at **rank 0 in 5 of 6** hard
  misses (rank 3 in the 6th). The answer email is perfectly indexed and findable — the
  natural-language *question* simply fails to rank it. **→ de-prioritises #14 (chunk size):
  chunking is not the lever for these misses.**
- **It's discriminability, not vocabulary absence.** Hard misses have *high* query↔thread
  lexical overlap (mean **0.62** vs 0.75 for covered) — the query words *are* in the thread.
  They are also in many *other* threads (recurring terms like "confirmed", customer/product
  names), so vocabulary-similar siblings out-rank the gold thread. The email body wins as a
  query because it is far more specific than the question.
- **Budget misses are one-rank near-misses.** The 3 sit at distinct-thread ranks 3, 3, 5 (the
  budget is 3). Widening top-3 → top-5 recovers 2 of 3 immediately — this *is* the 76%→84%
  gap the §9 top-N sweep showed. Near-free coverage (cost: a few k tokens).
- **Fusion misses are RRF dilution.** One: sparse ranks the gold thread at 6 but dense never
  returns it (>200) and RRF pushes the fused result to 58. The other: sparse 10 / dense 29 →
  fused 14. When one modality has a strong hit and the other is blind, RRF dilutes it; a
  min-rank fallback or weight tuning recovers both.
- **C′ net-helps coverage.** Embedded summaries cover **34 vs C's 32** (C′ rescues 3 — two
  fusion→covered, one hard→covered — and costs 1; net +2 / +4%). Corroborates §9's "keep C′".
- **Eval-hygiene caveat.** ≥2 of the 6 hard misses are meta/degenerate *generated* queries
  ("what was the subject of the email thread / of the meeting") — not real product questions.
  The automated `bad_query` guard flagged 0 because it keys on the oracle *failing*, and here
  the oracle *succeeds*; these are mislabeled `vocab_gap`. The genuinely retrieval-fixable hard
  core is therefore **~4, not 6**. N=45 is small — treat sub-bucket percentages as directional.

**Verdict — where the leverage is:**
- **Cheap, non-ML wins first (~5 of 11 misses, ~76%→~87% coverage):** widen the thread budget
  (3→5) and tune/guard RRF fusion. No model or index changes. → **#17.**
- **The real ceiling is query↔document matching.** The oracle proves a better *query* retrieves
  the gold thread at rank 0, so the highest-leverage real fix is **query-side**: query expansion
  / HyDE / multi-query — make the query look like the answer. → **#16** (filed as the new lead).
  Doc-side queryable summaries (#11/#13) attack the same gap from the other end.
- **De-prioritise #14 (chunk size)** for coverage — 0/6 hard misses are chunking defects.
- **Tighten the eval** — filter meta/degenerate generated queries and extend the `bad_query`
  guard before the next coverage measurement. → **#18.**

**Eval refresh (#18, 2026-05-30).** Replaced the 45-query set with **120 validated** queries
(48 terse / 48 content / 24 spanning) generated with a hardened prompt + an LLM validator gate
(`src/eval/query_validator.py`); the validator rejected ~24% of candidates. On the clean set the
hard-miss bucket **no longer contains any meta/artifact queries**, and the picture sharpens:
coverage **82%** (C′, top-3) · budget 6 (5%) · fusion 8 (7%) · hard 8 (7%). The hard core is now
**overwhelmingly terse-reply discriminability** — 7 of 8 hard misses are *terse* questions
("what did X decide/say"), all `vocab_gap`, **zero index/chunking**, oracle body-rank 0 every
time (the answer email is findable; the short reply just can't rank). C′'s coverage edge also
**grew on the cleaner set** (covers 98 vs C's 89, +9 — was +2 on the 45-set), reinforcing "keep
C′". This 82% is the trustworthy baseline for #17 (widen-N recovers ~4 of 6 budget misses) and
#16 (query-side help for the terse hard core).

---

## 11. Tunable fusion (power-mean) + widen-N — end-to-end (#17, 2026-05-30)

§10 pinpointed RRF-**sum** burying strong single-modality hits (7/8 fusion misses = gold thread
at sparse rank 0, drowned by mediocre-in-both siblings). We implemented a **tunable power-mean
fusion** over the per-list RRF terms `1/(k+rank)` — `p=1` is exactly RRF-sum, `p→∞` is CombMAX,
the continuum is a single tunable knob (`src/query/fusion.py: make_rank_fusion`). Evaluated it +
widen-N **end-to-end** on the clean 120-query set (§ eval-refresh): coverage (cheap p-sweep),
context tokens, and answer quality (local **26b@6bit** answer-gen, **26b@8bit** judge,
answer-vs-gold 0–3).

**Stage-1 coverage p-sweep (C′, N=3, n=120):** coverage rises only modestly with p — `p=1` 82%
→ `p=∞` **84% (+2pts)**; `p=∞` *regresses* the tight N=1 budget (69%→66%) — the tie/churn
CombMAX warning, realised. Widen-N is a comparable lever (N3→N5 ≈ +3pts at any p).

**Stage-2 end-to-end arm table:**

| arm | coverage | avg tokens | grade (0–3) |
|---|---|---|---|
| no_context (anchor) | – | 0 | 0.10 |
| answer_only (anchor) | – | 172 | 1.99 |
| C′ thread N3, RRF sum (baseline) | 82% | 6,150 | 1.54 |
| C′ thread N3, max-fusion | 84% | 6,730 | 1.59 |
| C′ thread N5, max-fusion | 87% | 11,910 | 1.61 |

**Paired (per-query) analysis — the gains are within noise:**
- **max-fusion vs baseline (N3):** only **8 of 120** queries changed grade — 5 better, 3 worse
  (**net +2**, meanΔ +0.05). It *reshuffles* rather than uniformly lifts: rescues some buried
  hits (3 queries +3 grade) but **demotes some terse answer-threads** (2 queries −2 grade).
- **widen-N5 vs N3:** **net −1 on grade** (2 wins / 3 losses) for ~2× the tokens; spanning grade
  drops (1.83→1.79). The extra threads add distraction. Counterproductive.

**Verdict — a negative result:**
- **Retrieval-knob tuning is second-order on this corpus** (reconfirms §9/§10). Neither
  max-fusion nor widen-N gives a meaningful end-to-end answer-quality gain: the fusion fix is a
  near-wash (helps and hurts roughly equally), and widening N to 5 is net-negative for double the
  tokens.
- **Keep RRF (`p=1`) as the default; reject widen-N→5.** We do not flip the default on a
  within-noise, partly-regressive (N=1) change.
- **The tunable combiner ships as available, corpus-tunable infrastructure** (not the default):
  `make_rank_fusion(p)` is a clean, tested knob anyone tuning mailrag for *their own* corpus can
  sweep — on a more lexical/heterogeneous corpus the buried-hit fix may pay off more than here.
- **The real lever remains query-side (#16).** The threads max-fusion *demoted* were terse —
  the exact class query expansion / HyDE targets.
- Caveat: the grade ceiling is low (`answer_only` = 1.99) — even gold-email-only doesn't grade as
  fully complete (judge strictness + terse-email self-containment), so treat absolute grades as
  relative, not calibrated.

---

## 12. Query-side retrieval — HyDE / anchored query expansion (#16, 2026-05-30)

§10/§11 left the query→document **matching** gap as the lead. The query-side bet (HyDE): instead
of searching with the user's question, generate a *hypothetical answer* and search with that — it
"looks like" a real answer email, so it should match the gold better, especially for terse
request→reply threads where no single email carries the query's vocabulary. Pure logic in
`src/query/hyde.py` (prompt + fail-safe `combine_query`); hypotheticals pre-generated once
(`scripts/eval/gen_hyde.py`) and consumed by the coverage diagnostic via `--hyde {off,pure,augment}`
(`pure` = search the hypothetical alone; `augment` = query + hypothetical).

**This stopped at Stage-1 (coverage), because Stage-1 is decisively negative** — no configuration
beats the raw-query baseline, so there is nothing for an end-to-end Stage-2 to recover (you cannot
answer from a thread retrieval never surfaced).

**Two prompts, two failure modes.** The first finding was that a *from-scratch* hypothetical
**fabricates competing specifics** (invented names/dates/times) that drag retrieval toward sibling
threads. So we added an **anchored** prompt (`build_hyde_prompt_anchored`): reshape the query into
answer-surface form, keep every real anchor verbatim, invent nothing. To rule out
"the generator is just too weak," we ran a **generator-quality ladder** spanning ~3 orders of
magnitude of model size — local **e4b** (4B), **Sonnet**, **Opus** (cloud, ~$1.50 total for 120×2;
the only cloud spend, query strings only).

**Stage-1 coverage (C′, N=3, n=120; terse = the 48 request→reply queries HyDE targets):**

| generator · prompt · mode | cov@N3 | cov@N5 | terse@N3 |
|---|---|---|---|
| **off — raw query (baseline)** | **82%** | **85%** | **77%** |
| e4b · from-scratch · pure | 56% | 63% | 50% |
| e4b · from-scratch · augment | 73% | 75% | 65% |
| e4b · anchored · pure | 75% | 79% | 73% |
| e4b · anchored · augment | 77% | 81% | 77% |
| Sonnet · anchored · pure | 71% | 76% | 69% |
| Sonnet · anchored · augment | 74% | 78% | 71% |
| **Opus · anchored · pure** (best HyDE) | **78%** | 81% | 77% |
| Opus · anchored · augment | 76% | 81% | 73% |

**Verdict — a negative result, but a sharply diagnostic one:**
- **No HyDE arm beats the raw query.** Best (Opus · anchored · pure) is **78% (−4 pts)** and only
  *ties* terse (77%). The conclusion is robust across generator (4B → frontier), prompt, and mode.
- **The anchored prompt is monotonically better than from-scratch** at every generator (e4b: 56→75
  pure, 73→77 augment) — it removes the fabrication drift and restores terse to baseline. The idea
  was right; it just asymptotes *to* the baseline, not past it.
- **Faithfulness, not capability, is what matters.** Sonnet (anchored) *underperformed* local e4b
  (anchored) — because it still invented concrete values (a made-up time, a made-up duration); Opus
  followed "invent nothing" and wrote around unknowns generically, reclaiming the top spot. A
  better model only helps insofar as it fabricates *less*.
- **Mechanism (airtight):** this corpus is entity-rich and **specific-fact**. The user's query
  *already contains the optimal retrieval anchors* (the real names/terms). A hypothetical can only
  (1) fabricate competing specifics → drift → harm, or (2) faithfully echo the query's anchors plus
  generic answer-vocab → *approach but never exceed* the raw query, because it adds no new *correct*
  signal, only dilution. The remaining gap is specific-fact **discriminability**, which the query
  side cannot manufacture — you would need the actual answer.
- **Ship decision: do not adopt HyDE on this corpus** (default stays raw-query hybrid retrieval).
  `build_hyde_prompt` / `build_hyde_prompt_anchored` / `combine_query` / `gen_hyde.py --anchored`
  stay in-tree as tested apparatus — HyDE may still pay off on a *lexical/open-domain* corpus where
  queries are keyword-sparse (the opposite regime), aligning with the clone-and-tune-your-own-corpus
  vision.
- **This justifies the doc-side lever (#11/#13).** Two opposite-end levers attacked the same gap;
  the cheap query-side one is now ruled out *with a mechanism*. The fix must make the **documents**
  more retrievable (thread-aware embedded summaries), not dress up the query.

---

## 13. Doc-side thread-aware summaries — the evolution ladder (#11/#13, 2026-06-01)

§12 ruled out the query side *with a mechanism*: this corpus is entity-rich, so the query already
holds the optimal anchors. The remaining lever is the **document** side — make a terse reply carry
the answer's vocabulary *at the unit that gets matched*. We do this with a per-email LLM summary,
**prepended to the body and embedded** (contextual retrieval), where the summary is conditioned on
the email's **preceding thread context** (causal, append-only). The whole point is to read this as
an **evolution ladder** — what each increment adds, and at what cost — not a single hero number.

### The ladder (each row adds one technique; retrieval is otherwise identical)

| Row | Increment | LLM? |
|---|---|---|
| 0 | plain email RAG (body-only, email-level rank) | no |
| 1 | + small→big retrieval (thread-level rank — match a unit, return its thread) | no |
| 2 | + isolated per-email summary, embedded | yes |
| 3 | + preceding-thread-context summary (this work) | yes |

Rows 1–2 are existing techniques: small→big is parent-document / auto-merging retrieval
(LlamaIndex, LangChain); embedding an LLM context blurb with each chunk is contextual retrieval
(Anthropic, 2024). Row 3 is the variant we test — the summary sees only the email's *preceding*
messages (causal, append-only) rather than the whole thread. The rest of this section measures what
that change buys, and at what cost.

**Result (n=360 validated queries, thread-level coverage, all summary arms at the same 26b@8bit
quant — see "the quant confound" below for why same-quant matters):**

| arm | R@1 | R@3 | R@5 | R@10 | MRR |
|---|---|---|---|---|---|
| row 1 — body-only | 60 | 71 | 76 | 81 | .675 |
| row 2 — isolated summary @8bit | 70 | 81 | 86 | 89 | .763 |
| **row 3 — preceding-context @8bit (ours)** | 70 | **84** | 87 | 91 | .779 |

The ladder is monotone. The two biggest jumps tell the story: **row 0→1** (R@1 36→60, LLM-free) is
the value of thread expansion alone; **row 2→3** is the increment this work adds.

> **Relation to the headline recall@5 ladder.** This section decomposes the *summary type* (which
> summary helps), all measured **thread-level @8bit without reranking**. The repo's headline
> **technique ladder** — plain dense 46 → +learned sparse 49 → +contextual summary 62 → +rerank 64
> → +thread reconstruction **93** (recall@5) — is a *different cut*: it adds the NVIDIA-reranker step
> and reports thread-recall *with* rerank (89→93). Both are now reproducible on `main`:
> `scripts/eval/bench_avc.py` (technique ladder + NVIDIA C-arms + Enron-QA cross-check) and
> `bench_thread_reconstruction.py` (email-recall vs thread-recall). They agree where they overlap
> (preceding-context hybrid thread-recall@5 ≈ 87–89, → 93 with the reranker).

### ⚠ The quant confound (the key methodology lesson)

An earlier pass compared preceding-context @**8bit** against an isolated control @**6bit** and
reported +6pp R@3 (p=0.0017). Re-running the isolated control at the **same 8bit quant** showed
that was inflated — it bundled two independent steps:

- **6→8bit quantization of the summarizer:** isolated R@3 78 → 81 (+3pp) — nothing to do with
  thread context.
- **Preceding-context (same-quant):** isolated 81 → preceding 84 (+3pp) — the real method effect.

Lesson: a summary-quality experiment must hold the **summarizer quant fixed**, or a quant step
masquerades as a method effect. (Build controls: all arms share the 19,859-email corpus, 21,590
chunks, `--chunk-size 512`, and `--embed-summary` with `embed_max_length` decoupled from chunk_size
(#14) so the summary *adds* headroom rather than *displacing* body tokens.)

### Honest significance (same-quant, n=360)

- **Corpus-wide:** preceding vs isolated, covered@3 = 302/360 (83.9%) vs 290/360 (80.6%), net +12.
  McNemar exact two-sided **p = 0.058 — directional, not significant.**
- **By category (R@3), where the effect actually lives:**

  | category (n) | isolated @8bit | preceding @8bit | Δ | McNemar p |
  |---|---|---|---|---|
  | terse (144) | 75% | 81% | +6pp | **0.035 ✓** |
  | content (144) | 79% | 81% | +2pp | 0.61 |
  | spanning (72) | 94% | 94% | 0 | 1.00 |

- **The defensible claim:** *preceding-thread-context summaries significantly improve **terse-reply**
  retrieval at the top-3 operating point (75%→81%, p=0.035, same-quant).* This is exactly the design
  target — terse request→reply threads where no single email carries the query's vocabulary (§10).
  The corpus-wide effect is real but modest (+3pp) and not yet significant; content/spanning are
  unchanged, and within terse only R@3 (not R@1/5/10) reaches significance. We report the scoped
  result, not a broad win.

### Preceding vs whole-thread: append-only is free

Does *preceding-only* leave accuracy on the table versus conditioning on the **whole** thread (the
bidirectional context Anthropic's contextual retrieval uses)? We built that arm too
(`gen_thread_summaries.py --mode whole`, same 26b@8bit) and measured all three at matched quant:

| arm (thread-level) | R@1 | R@3 | R@5 | R@10 | MRR |
|---|---|---|---|---|---|
| iso @8bit | 70 | 81 | 86 | 89 | .763 |
| preceding @8bit (ours) | 70 | 84 | 87 | 91 | .779 |
| whole @8bit (bidirectional) | 73 | 84 | 89 | 91 | .799 |

**Whole ≈ preceding.** At the top-3 operating point they are statistically indistinguishable —
covered@3 84.4% (whole) vs 83.9% (preceding), McNemar p = 0.82; per-category R@3 is within a point
everywhere (terse 82 vs 81, content 82 vs 81, spanning tied). Both significantly beat iso (whole vs
iso p = 0.02). Whole holds a slight edge at R@1 (73 vs 70) and MRR (.799 vs .779) — seeing later
replies refines the *top* rank a little — but buys nothing at top-3.

That is the result the method was designed around: **bidirectional context adds no coverage over
preceding-only, so the append-only property comes for free.** Preceding-only summarizes each message
once, against what already existed when it arrived, and never re-summarizes or re-embeds a thread's
earlier messages as it grows — where a whole-thread index is unstable, every new reply invalidating
its siblings' vectors. Same accuracy, live-ingest-friendly by construction.

### What the LLM is really buying

It's fair to ask whether a per-email LLM pass earns a +3pp retrieval gain. The framing that answers
it is **one pass, paid twice**. The summary is not a separate call — it is a byproduct of the
noise-classification pass the pipeline already runs on every email (§1–§5): a single call returns
both `is_noise` and the summary. Used this way, that one pass pays twice — it removes the noise a
**regex cannot** (§2: roughly a third of the noise needs the LLM to catch), *and* it yields the
retrieval summary.

The right baseline for the cleanup half is not "no cleanup" but the cleanup you get **free from a
regex**; the LLM's marginal value is the noise the regex *misses*. We tested this directly: a
body-only corpus cleaned by regex rules only (`config/noise_rules.yaml`) leaves ~8.7k LLM-only-noise
emails in as distractors (28,628 emails vs the regex+LLM-clean 19,859); rerun the 360-query coverage
and compare.

The result is smaller than the hypothesis predicted, and worth stating plainly: gold-thread
coverage@3 drops only **71% → 69%** when that residue is left in (+2pp for the LLM's incremental
cleanup; McNemar p = 0.15 — directional, not significant). On this corpus the explanation is the same
one from §12 — newsletter/notification noise is semantically far from the entity-rich queries, so it
rarely out-ranks a specific gold thread. The **recall** cost of leaving it in is small.

So the cleanup's value is not a gold-recall effect — it is a **precision** effect the recall metric
cannot see, and we measured it directly (no answer-LLM, just a top-N retrieval and a
clean-vs-noise membership check on `source_id`). On the regex-only corpus, the vector DB is **not**
good enough to keep noise out: **21% of queries surface at least one noise email in their top-3, and
~11% of all top-3 retrieval slots are noise** (top-5: 30% of queries, 12% of slots; top-10: 43%,
13%). That is the junk a regex misses, sitting in the very context an answer model would receive —
removed *for free* by the same LLM pass that produced the summary.

Net, the "one pass, paid twice" economics hold and now have numbers on both sides: the summary lifts
terse **recall** (+6pp, significant), and the cleanup lifts **precision** (~11% of top-3 slots
de-junked). The recall gains are modest and we say so — but the cleaner earns most of its keep in
precision, which a coverage metric alone would have missed entirely.

### Where this sits in the literature

Prepending an LLM context blurb before embedding is Anthropic's *Contextual Retrieval* (2024), which
conditions on the whole document. The email-RAG tools we looked at take a different cut: RAG-Mail is
thread-aware but embeds raw text; msgvault and Onyx are hybrid/contextual but not per-email
thread-conditioned; the causal-ancestor idea appears in email-thread *summarization* (EmailSum, 2021)
but not as an embedded retrieval signal. The combination we did not find already done is the specific
one here — a per-email summary conditioned on *preceding* thread context, embedded as the retrieval
surrogate. It is a deliberate specialization of contextual retrieval to causal conversation
structure, not a new primitive, and the motivation is practical: preceding-only is **append-only**,
so live ingest summarizes each message once against what already exists, rather than re-summarizing
and re-embedding a whole thread every time it grows.

### Open questions / future work

- **End-to-end answer-quality impact of cleanup.** We have shown the cleanup is a *precision* win at
  retrieval (~11% of top-3 slots de-junked, above) — the open question is how much that junk actually
  degrades *answers*. An end-to-end A/B (regex+LLM vs regex-only corpus, same queries, scored by the
  answer judge) would price it; expected to matter most on terse queries, where noise crowds a thin
  answer signal.

---

## 14. A second corpus — the LLM rubric is *not* portable (#30, 2026-06-05)

[§4](#4-portability--a-shared-starter-blocklist) showed that about a third of the *regex*
blocklist carries across corpora for free. The LLM **rubric** — what the model is told to
treat as noise versus a record worth keeping — does not. I pointed the same pipeline at my
own ~25,000-email personal archive, and the cost of assuming the rubric would carry over
showed up fast.

| corpus | judged | Pass-2 noise rate | kept |
|---|---|---|---|
| work ([§1](#1-the-cleanup-funnel--measured-savings)) | 31,969 | 37.9% | 19,859 |
| personal (rubric calibrated *for this corpus*) | 24,979 | **61.5%** | 9,620 |

The portability test is the part that matters. Re-running Pass-2 over the personal archive
with the *corporate* rubric flagged **87.6%** of it as noise. The rubric calibrated for the
personal corpus flagged **61.5%** (the two runs covered near-identical sets, 24,527 vs
24,979 judged, so compare rates, not raw counts). That 26-point gap isn't borderline churn.
It's genuine personal mail — receipts, bank statements, actual correspondence — that the
corporate rubric's idea of a "record" doesn't recognize. Shipped blind, the wrong rubric
would have deleted about **one in four real personal emails**.

**Why it doesn't port.** What counts as noise depends on what the corpus is for. In a work
mailbox a vendor invoice or a calendar invite is operational record; in a personal mailbox
those same forms mean something else, and the long tail (family logistics, medical, bank
statements) has no corporate equivalent. The rubric encodes those judgments, so it has to be
re-fit per corpus. That's the opposite of the shared regex blocklist.

**What caught it,** before burning the ~6 h run, was the calibrate gate from pipeline
increment 1b:

- a **rubric registry**: shipped generic templates, plus local corpus-specific overrides
  resolved local-first;
- a **calibration pass** over a ~200-email sample that sorts the model's calls into
  *false-noise* (real mail it would drop) and *false-keep* (noise it would keep), and makes
  a human actually read both buckets;
- a **gate**: `mailrag pass2` won't run until the active rubric has been calibrated
  (`--force` overrides).

Calibration takes minutes; the full sweep took about 6 h on a local model (~2,500 emails/hr,
GPU-bound). The gate is there to make you spend the minutes before the hours.

**Verifying the dropped pile.** Calibration only tells you the rubric looks reasonable. It
doesn't prove the 15,359 drops were right, so I spot-checked. A random 20 of the dropped
emails were all genuine noise. A targeted sweep of the drop pile for record-like keywords
turned up about 8% suspects, and every one was a correct drop on inspection (the keyword
matched a promo or a notification, not an actual record). Nothing real surfaced in the sample.

**Where it's still shaky (→ [#30](https://github.com/fmasi/mailrag/issues/30)).** One band
resists a single-pass call: borderline promo-versus-record emails from a recurring sender,
where product pitches sit right next to statement-like messages. The model dropped the
pitches and kept the records, which is defensible, but it wasn't consistent across
near-identical cases. The problem isn't a wrong call, it's an unstable one. That's what #30
is for: a second-opinion verifier on the borderline band, a different model casting an
independent vote.

**Cost.** Pass-2 is cached, so every later decision off this corpus (drop threshold, chunk
size, a possible sender carve-out) rebuilds in about 8 min for $0. Only the one-time ~6 h
sweep is expensive. All of it ran on a local Gemma on Apple Silicon: no cloud spend, and no
mail left the machine.

---

## Open threads / next experiments

- ~~**Thread-aware retrieval**~~ — implemented (§8). Retires C′; dedup subsumed. Sub-research
  remaining: thread-size bounding validation on long threads (LLM thread summary and/or
  parent-id segmentation).
- ~~**Larger labeled eval set**~~ — DONE (§9). Settled it: keep C′, C′+top-1–3 threads, rerank
  off, a small model suffices. **New lead: retrieval coverage (~76%) is the ceiling, not the
  answer model.**
- ~~**Diagnose the coverage ceiling** (#12)~~ — DONE (§10). The ceiling is a **query→document
  matching** problem, not indexing/chunking (0/6 hard misses are chunk defects; the gold email
  is retrievable at rank ~0 from its own text).
- ~~**Eval hygiene** (#18)~~ — DONE (§ eval-refresh). 120 validated queries; clean 82% baseline.
- ~~**Cheap fusion/widen-N wins** (#17)~~ — DONE (§11). **Negative result:** tunable power-mean
  fusion + widen-N are within-noise / counterproductive end-to-end here; kept RRF default, shipped
  `make_rank_fusion` as a corpus-tunable knob. Retrieval-knob tuning is second-order.
- ~~**Query-side retrieval (#16)**~~ — DONE (§12). **Negative result:** HyDE / anchored query
  expansion never beats the raw query on this entity-rich corpus, confirmed across an e4b→Opus
  generator ladder. The query already holds the optimal anchors; a hypothetical can only fabricate
  drift or echo the query. Apparatus kept (`src/query/hyde.py`) for lexical corpora.
- ~~**Doc-side queryable summaries (#11/#13)**~~ — DONE (§13). Preceding-thread-context embedded
  summaries **significantly improve terse-reply retrieval** (75%→81% covered@3, p=0.035, same-quant
  n=360); corpus-wide effect modest (+3pp, n.s.). Key lesson: an earlier +6pp was half a 6→8bit
  **quant confound** — the same-quant control halved it. Whole-thread (bidirectional) control PENDING
  to settle the append-only novelty framing. **#14 (chunk-size) closed via `embed_max_length` decoupling.**
- ~~**Deduplicate results by email** (#2)~~ — subsumed by thread-aware retrieval (§8): grouping
  by `thread_id` and deduplicating by `message_id` inside expansion is the dedup.
- **Finer targeted-LLM** — extend the subject signal to subdivide the dominant work domain
  and re-measure the LLM budget saved.
- **Learn from spam filtering** — decades of prior art (Bayesian filters, shared blocklists,
  greylisting) likely transfers to email-noise cleaning.

> Methodology note: several of these findings overturned the "obvious" assumption once
> measured (the regex-vs-LLM coverage, the contextual-retrieval trade-off). Test the theory
> when you can.
