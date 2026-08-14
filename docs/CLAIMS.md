# Claims register

Every number this project publishes, what produced it, and when it was last
verified.

**Why this file exists.** The README and the landing page state a dozen figures.
Most come from one-off scripts run months ago against a private corpus. Without a
register, "is that still true?" is unanswerable, and the honest answer drifts
silently as the code, the models and the dependencies move underneath it. A
published number with no traceable source is a claim, not a result.

It also guards a subtler failure. When `make bench` shipped, its docs listed two
omissions (rerank, thread reconstruction) and missed a third — contextual
summaries were switched off too, so a reader could reasonably have assumed the
+12.8 lever was in play. Writing every claim down next to what actually produces
it is how that class of gap gets caught.

## How to read this

- **Public** — a stranger can reproduce it with no key and no private data.
- **Private** — needs the private mailbox and/or a paid endpoint. Author-reported.
- **Last verified** — when the producing script was last *run*, not when the
  number was written down.

Anything marked ⚠️ is currently unverifiable and should not be restated without
saying so.

## Retrieval claims

| # | Claim | Where published | Produced by | Corpus | Status | Last verified |
|---|---|---|---|---|---|---|
| R1 | dense R@5 **94.4** [91.6, 96.4] vs dense+sparse **97.5** [95.3, 98.7]; McNemar p=0.0034 | README, landing page, `BENCHMARK.md` | `scripts/eval/bench_public.py` | public Enron-QA, 2 000 docs / 360 q | **Public** ✅ | 2026-08-12 |
| R2 | large set: dense **90.0** vs dense+sparse **94.4**; p=0.0001 | README, `BENCHMARK.md` | `scripts/eval/bench_public.py --size large` | public Enron-QA, 10 000 docs | **Public** ✅ | 2026-08-12 |
| R2b | contextual summaries: plain R@5 **60.6** vs with-context **73.7**; McNemar p=0.0044 | README, `make demo` | `scripts/demo.py` | public Enron, 1 200 docs / 99 validated questions | **Public** ✅ | 2026-08-14 |
| R3 | plain-dense baseline **45.6%** R@5 | README, landing page | private eval ladder | private ~32k mailbox | Private | 2026-06 (pre-register) |
| R4 | thread reconstruction **64.2 → 93.3** (+29.1) | README, landing page | `scripts/eval/bench_thread_reconstruction.py` | private, `work-rag-ctx-threadaware` | Private ✅ **re-verified, exact** | 2026-08-13 |
| R5 | contextual summaries **+12.8** | README, landing page | `scripts/eval/build_bodyonly_collections.py` (builds the no-summary control) | private | Private — **corroborated publicly by R2b** (+13.1pp R@5 on a different corpus) | 2026-06 (pre-register) |
| R6a | cross-encoder rerank **+2.5** R@5 overall | README, landing page | `scripts/eval/bench_thread_reconstruction.py` (rerank arm) | private + paid NVIDIA endpoint | Private ✅ **re-verified, exact** (61.7 → 64.2) | 2026-08-13 |
| R6b | rerank *demotes* the answer on **thread-spanning** queries | README, landing page | — | — | ⚠️ **NOT reproduced on recall** — see below | 2026-08-13 |
| R7 | NVIDIA's stack wins on TREC Legal; mailrag wins on email — "opposite winners" | README, landing page | `scripts/eval/bench_trec.py`, `build_trec_collection.py` | TREC Legal + paid NVIDIA endpoint | Private ⚠️ | 2026-06 (pre-register) |
| R8 | an early **+6pp** gain was half a quantization artifact, worth **+3pp** at matched precision | README | private eval, §-numbered in `EXPERIMENTS.md` | private | Private | 2026-06 (pre-register) |

⚠️ **R6b does not hold as written.** Re-running the rerank arm (2026-08-13, 360
queries) shows rerank is **neutral or positive on recall in every category**:

| category | n | E@5 hybrid → +rerank | E@1 |
|---|---|---|---|
| content | 144 | 60.4 → 67.4 (**+7.0**) | +1.4 |
| terse | 144 | 58.3 → 57.6 (−0.7 — *one query*) | +8.3 |
| spanning | 72 | 70.8 → **70.8** (0.0) | +6.9 |

Thread-spanning is *exactly unchanged* at @5 and @10 and better at @1. So the
demotion cannot be a recall result. It comes from the **§9 LLM-judged
answer-quality** eval, where `EXPERIMENTS.md` records that rerank "HURT every
category" — a different metric on a different run. The README states it directly
beside the `+2.5 R@5` figure, which reads as though both come from the same
measurement.

`EXPERIMENTS.md:151` separately describes the demotion as affecting *contentless
**terse*** emails, not spanning ones — which matches this data far better than the
README does. **The README wording needs correcting**; tracked in #128.

⚠️ **R7 remains unverifiable by measurement** until its collections are rebuilt
(`trec-bge` / `trec-e5` are no longer in Qdrant). The key is now available, so it
is unblocked — it just needs the build step first.

## Performance claims

| # | Claim | Where published | Produced by | Status | Last verified |
|---|---|---|---|---|---|
| P1 | `make bench` runs in **1.6 min** (MPS) / **14.7 min** (CPU-only) | README, `BENCHMARK.md`, Makefile | timed `make bench`, M5 Pro 6P+12E / 48 GB, load-gated | **Public** ✅ | 2026-08-12 |
| P2 | `make bench` spends **zero LLM calls** | README, `BENCHMARK.md` | code inspection: `embed_summary=False`, `apply_noise_filter=False`, no profile, retrieval-only scoring | **Public** ✅ | 2026-08-12 |

## Scope claims (things we assert are *not* measured)

| # | Claim | Verified by |
|---|---|---|
| S1 | `make bench` excludes thread reconstruction, summaries, rerank and noise cleanup | `BENCHMARK.md` exclusion table; `bench_public.py` flags |
| S2 | Enron-QA carries no conversation linkage, so thread reconstruction cannot be scored on it *as shipped* | dataset schema: `email / questions / path / user / …`; 400-row sample shows `Message-ID` in 1, `In-Reply-To` in 0 |
| S3 | The **CMU Enron maildir** has no threading headers either — but conversations are derivable | 8 000 real messages: `In-Reply-To` 0.0%, `References` 0.0%, yet `Re:`/`Fw:` subjects 64.3%. Deriving by normalised subject + shared participant puts **50.2%** of 19 530 messages in a multi-message thread (largest 59). Measured 2026-08-14 |
| S4 | Derived threads carry a **measured false-merge rate**, concentrated in generic subjects and long spans | 19 530 Enron messages: 2.2% of same-thread pairs share **no** participant (transitive chaining, 1.0% of threads); 11.1% of threads span >30d and **1.4% span >1 year**; generic subjects ("hey", "lunch", "meeting") account for 55 threads / 2.3% of threaded messages. Worst case observed: "happy hour", 36 messages over 391 days across 16 participants — a recurring invite, not a conversation. Measured 2026-08-14 |

## Re-verification policy

A full private re-run costs GPU hours and LLM calls, so this is **not** per-commit.
Re-verify:

1. **Before any tagged release** — every row, or explicitly mark what was skipped.
2. **After changing** the embedder, chunking defaults, retrieval mode, or fusion.
3. **After a major dependency bump** that touches torch / FlagEmbedding / Qdrant.

Update the "Last verified" column in the same PR as the run. A row whose date is
older than the last release is a row that should not be quoted without a caveat.

## Running the private scripts

They need the private corpus and are not runnable by a stranger; the public path
is `make bench`. Data locations are environment-overridable — see
[`scripts/eval/_paths.py`](../scripts/eval/_paths.py):

```bash
MAILRAG_EVAL_QUERIES=eval/out/queries_360.jsonl \
  python -m scripts.eval.bench_thread_reconstruction      # R4, R6 (needs NVIDIA_API_KEY)

MAILRAG_EVAL_TREC=~/msgvault-eval-proof/trec \
  python -m scripts.eval.bench_trec                       # R7 (needs NVIDIA_API_KEY)
```
