# Experiments & findings

A running log of what we measured building `mailrag` against a real ~32,000-email
corporate mailbox (all references anonymized). The point is to document the *why* and
the *trade-offs* with real numbers — including the places where the obvious assumption
turned out wrong.

> **Corpus:** 70,016 exported emails → **31,969** selected (work-account folders).
> All identifying names replaced with placeholders (`ACP` = a partner certification
> program; "the work domain" = the employer's own email domain).

---

## 1. The cleanup funnel — measured savings

| stage | what it does | effect |
|------|--------------|--------|
| Scope | keep only work-account folders | 70,016 → **31,969** |
| Pass-1 (regex) | cheap sender/subject rules, *before* any embedding | flags **10.4%** (3,332) standalone |
| Pass-2 (local LLM) | summarize + judge each email | flags **37.9%** (12,123) noise |
| Calendar-collapse + chunk-dedup | 1-line calendar summaries; drop byte-identical chunks | 22,613 → **21,590** chunks |
| **Net** | | 31,969 emails → **19,820 kept** → 21,590 chunks |

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

**Worked example.** Searching for a partner certification program by its acronym (`ACP`)
mixes a *semantic* concept (certification readiness) with a *rare exact token* (`ACP`):
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
  is retrievable at rank ~0 from its own text). Leads, in priority order:
  **#16** (query expansion / HyDE — highest-leverage real fix), **#17** (cheap wins: widen
  thread budget 3→5 + RRF fusion tuning, ~+11% coverage), **#11/#13** (doc-side queryable
  summaries), **#18** (eval hygiene). **#14 (chunk size) de-prioritised** for coverage.
- ~~**Deduplicate results by email** (#2)~~ — subsumed by thread-aware retrieval (§8): grouping
  by `thread_id` and deduplicating by `message_id` inside expansion is the dedup.
- **Finer targeted-LLM** — extend the subject signal to subdivide the dominant work domain
  and re-measure the LLM budget saved.
- **Learn from spam filtering** — decades of prior art (Bayesian filters, shared blocklists,
  greylisting) likely transfers to email-noise cleaning.

> Methodology note: several of these findings overturned the "obvious" assumption once
> measured (the regex-vs-LLM coverage, the contextual-retrieval trade-off). Test the theory
> when you can.
