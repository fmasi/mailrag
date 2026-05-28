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
cast a wide net, then rerank to drop the drift.

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

## Open threads / next experiments

- **Thread-aware retrieval** (parent-document / thread reconstruction) — the elegant
  alternative to two collections + routing: retrieve over one collection (`C`) + reranker, then
  group/expand results by `thread_id` so terse replies are covered as thread context. Could
  retire C′ entirely. Sub-research: thread-size bounding (LLM thread summary and/or parent-id
  segmentation of long threads).
- **Larger labeled eval set** — turn the directional eyeballing of §7 into precision/recall/nDCG
  numbers across A/B/C/C′(+rerank), weighted by the real query mix, to settle the trade-off.
- **Deduplicate results by email** (#2) — multiple chunks of one email currently crowd the
  top-K; group by Message-ID at display time.
- **Finer targeted-LLM** — extend the subject signal to subdivide the dominant work domain
  and re-measure the LLM budget saved.
- **Learn from spam filtering** — decades of prior art (Bayesian filters, shared blocklists,
  greylisting) likely transfers to email-noise cleaning.

> Methodology note: several of these findings overturned the "obvious" assumption once
> measured (the regex-vs-LLM coverage, the contextual-retrieval trade-off). Test the theory
> when you can.
