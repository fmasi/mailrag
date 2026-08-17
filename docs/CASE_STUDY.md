# Case study: what the cleanup and retrieval choices actually bought

The technique-by-technique breakdown behind the headline numbers. What each step cost, what it
bought, and the two places the intuitive choice turned out to be wrong.

This lives here rather than in the README so the front page stays readable. Nothing has been
cut. For which figures are publicly reproducible and when each was last re-run, see
[`CLAIMS.md`](CLAIMS.md). For the full experiment log with the §-numbered ladder, see
[`EXPERIMENTS.md`](EXPERIMENTS.md).

> Most numbers below come from running `mailrag` over a real ~32,000-email corporate mailbox.
> The portability check at the end uses a second, personal archive. They are here so the repo
> doubles as a worked example: why each step exists, what it saves, and what it costs.
>
> **All identifying names are placeholders.** `SPP`, the running example, stands in for a real
> programme name that is not published anywhere in this repo. Company names, senders and
> project names are all replaced the same way.

## Cleanup: measured savings, and an honest cost-benefit

The corpus is filtered in stages before anything gets embedded:

| stage | what it does | effect on this corpus |
|------|--------------|-----------------------|
| **Scope** | keep only the work-account folders | 70,016 exported → **31,969** selected |
| **Pass-1 (regex)** | cheap sender and subject rules drop obvious bulk (newsletters, social, automated senders) before any expensive work | flags **10.4%** (3,332) |
| **Pass-2 (local LLM)** | summarize and judge each email's content | flags **37.9%** (12,123) as noise |
| **Calendar-collapse + chunk-dedup** | one-line calendar summaries, and byte-identical chunks dropped | 22,613 → **21,590** chunks (−1,023) |
| **Net** | | 31,969 emails → **19,859 kept** → 21,590 embedded chunks |

### How much of that actually needed an LLM

Enough to measure rather than guess. Regex rules derived from the corpus, meaning high-noise
sender domains and calendar or out-of-office subject patterns, catch about 65% of the LLM's
noise at high precision. They miss roughly 35%, around 4,200 emails, and the miss is
structural rather than fixable with better rules. The work domain itself is 29% noise: 24,000
emails interleaving real correspondence with compliance reminders, calendar churn, AMAs and
internal newsletters. You cannot write a sender rule for your own domain.

Two qualifiers keep that honest. Rule *discovery* never needed a full pass, because the
dominant noise senders fall straight out of a sender-frequency table and a small sample finds
them without the 32k run. And the 48-hour to under-10-minute embedding win came from the
inference method, FlagEmbedding on Apple-silicon MPS, plus the volume reduction. The LLM had
nothing to do with it.

So the local-LLM pass earns its keep twice over: the ~35% of mixed-domain noise that cheap
rules cannot reach, and the per-email summaries that power the retrieval gains below and make
results readable by a human. Cheap regex for the obvious bulk, the LLM for the interleaved
noise and the summaries only it can write.

### The rubric does not port across corpora

Run the same pipeline over a ~25,000-email *personal* archive and the point makes itself. The
*corporate* noise rubric flagged **87.6%** of personal mail as noise, and it would have deleted
real receipts, bank statements and correspondence. A rubric calibrated for the personal corpus
flagged **61.5%**.

A ~200-email calibration caught the gap before the ~6-hour run, and a spot-check of the
dropped pile confirmed it, all on a local model with no cloud spend. Full write-up in
[EXPERIMENTS §14](EXPERIMENTS.md#14-a-second-corpus--the-llm-rubric-is-not-portable-30-2026-06-05).

## Retrieval: what each technique adds, and what it costs

| technique | what it adds | trade-off (observed) |
|-----------|--------------|----------------------|
| **Dense (semantic) only** | matches meaning and paraphrase | misses rare exact tokens (acronyms, IDs), returns redundant near-duplicate chunks |
| **+ learned sparse + RRF fusion** (bge-m3) | exact-token and acronym precision, fused with semantics | needs a sparse-capable embedder and fusion, more storage |
| **+ LLM noise removal** | precision. Catches the ~⅓ of noise regex cannot, and clears junk out of the top results | one-time LLM cost (see above) |
| **+ contextual retrieval** (each email's summary prepended before embedding, the `C′` / `work-rag-ctx-*` collection) | short and terse emails match by gist. Best ranked arm and the end-to-end winner | one extra embedded collection to build and maintain |
| **+ cross-encoder reranker** | small precision lift on pointed queries (**+2.5 R@5**) | a paid per-query call. Neutral to positive on recall in every category, but it **hurt answer quality under the LLM judge** (§9), so it is off by default |
| **+ thread reconstruction** (pull the full conversation of each top hit) | **message-level recall@5 64.2% → thread-level 93.3%**. Match a small unit, answer from its whole thread | larger context per query, tunable by expanding top-N threads |

### How the eval was run

The eval set is **360 synthetic queries**, split 144 terse, 144 content and 72 spanning, each
generated from a known email so the **recall ladder scores against hard gold labels with no
LLM judge in the loop**.

Generating a query from its answer email risks circularity, since the query could echo the
target's exact tokens. Queries therefore come from body content under a rule that bans
artifact and metadata questions, followed by a validation pass. The load-bearing guard is
external rather than procedural: the ordering holds on public **Enron-QA**, whose questions
were written independently of this generator, and on TREC's real human judgments below.

A separate answer-quality lens does use a local LLM judge, calibrated against a stronger
reference model on 514 pooled pairs (Cohen's κ = **0.52** on the 0–3 scale, Spearman 0.74).
The decisive check was not the correlation. Both pre-registered decisions came out identical
under both judges. Significance tests and confound controls are in
[`EXPERIMENTS.md` §9–§13](EXPERIMENTS.md#9-labeled-eval--retrieval-metrics-coverage-and-end-to-end-answer-quality-2026-05-29).

### What the measurement said

- **Thread reconstruction is the biggest single win, and it needs no LLM.** Matching a small
  unit and returning its whole conversation lifts **recall@5 from 64.2% (message-level) to
  93.3% (thread-level)**, a gain of 29.1. It trades "find the needle" for "find the right
  thread", which the conversation then answers. The target shifts from one message to its
  thread by design: for a conversation, the thread is the right unit of truth.
- **Thread-aware *summaries* help where they were designed to, on terse replies.** *(Note:
  "thread-aware" names two things, the retrieval expansion above and this summary-conditioning
  step. See the [terminology box](EXPERIMENTS.md#terminology-read-this-first).)* Conditioning
  each email's embedded summary on its *preceding* thread context lifts terse-reply retrieval
  from covered@3 75% to 81% (p = 0.035). The corpus-wide effect is real but modest at +3pp,
  and it is reported as such.
- **A confound caught and reported.** An early +6pp headline turned out to be half a
  *quantization* artifact. Re-running the control at matched quant split it into +3pp from
  quantization and +3pp from the method. Holding the summarizer fixed is the difference
  between a result and a mirage.
- **Cleanup pays in precision rather than recall.** Leaving the noise a regex cannot catch
  barely dents gold recall, since the database still finds the answer. But **21% of queries
  then surface noise in their top-3**, about 11% of slots, which the LLM removes for free in
  the pass that also writes the summary.
- **Reranking helps pointed questions, and a claim that it hurt spanning ones did not survive
  re-running.** A cross-encoder reranker adds only **+2.5 recall@5** overall. This document
  used to say it *demoted* the answer on multi-email questions. Re-measuring that arm on
  2026-08-13 over the same 360 queries found the opposite of a demotion: spanning recall is
  identical at @5 (70.8 → 70.8) and better at @1 (+6.9), and no category loses recall. The
  demotion result came from the **§9 LLM-judged answer-quality** eval, a different metric on a
  different run, and it was being restated next to a recall figure as though the two came from
  one measurement. Answer quality is still why the reranker is off by default. Query-side HyDE
  never beat the raw query on this entity-rich corpus. Both stay in-tree for corpora where
  they would pay off. See [`CLAIMS.md`](CLAIMS.md) row R6b.
- **Retrieval is the ceiling, well before the model is.** With the answer in context the answer
  model was right about 88% of the time, and at matched precision a 4 B model essentially tied
  one six times its size. The lost points are queries where retrieval never surfaced the
  thread. Model size came out second-order.

## The compound effect: the canonical recall@5 ladder

Each technique added one at a time, scored on the 360 queries against hard gold labels with no
LLM judge, reproducible via `scripts/eval/bench_avc.py` and `bench_thread_reconstruction.py`:

| step | recall@5 | gain |
|------|----------|------|
| plain dense | 45.6% | — |
| + learned sparse | 48.9% | +3.3 |
| + contextual summary | 61.7% | +12.8 |
| + reranking | 64.2% | +2.5 |
| **+ thread reconstruction** ★ | **93.3%** | **+29.1** |

★ The last step switches from "find the exact email" to "find its *thread*", a legitimately
easier and more useful target, measured as thread-recall. Both of the big levers, thread
reconstruction at +29.1 and contextual summary at +12.8, come from understanding the
conversation rather than from a fancier embedding model. The same ordering holds on public
Enron-QA.

A general-purpose dense-plus-rerank yardstick (NVIDIA's retrieval NIMs) trails on email, at
57% R@5 against the reranked hybrid's 64% in a like-for-like comparison with both arms
reranked. The same yardstick wins on TREC legal e-discovery. That is task fit rather than
brand: their stack is built and tuned for general retrieval and is very good at it, and email
is a different shape of problem. The value here is not any single trick. It is the stack and
the discipline of pricing every layer.

## Worked example: the acronym problem

Searching this corpus for the partner programme by its acronym, `SPP`, mixes a semantic
concept (partnership onboarding) with a rare exact token. Dense-only finds the concept but
ranks the literal acronym low. Sparse-only finds the token but misses paraphrases. Hybrid with
RRF gets both.

Multi-query expansion, which searches several phrasings and fuses them with RRF, further
bridges the acronym to its expansion, at the cost of extra queries per search.

*(`SPP` is a placeholder, as noted at the top. The retrieval behaviour it illustrates is not.)*
