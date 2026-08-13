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
| R3 | plain-dense baseline **45.6%** R@5 | README, landing page | private eval ladder | private ~32k mailbox | Private | 2026-06 (pre-register) |
| R4 | thread reconstruction **64.2 → 93.3** (+29.1) | README, landing page | `scripts/eval/bench_thread_reconstruction.py` | private, `work-rag-ctx-threadaware` | Private | 2026-06 (pre-register) |
| R5 | contextual summaries **+12.8** | README, landing page | `scripts/eval/build_bodyonly_collections.py` (builds the no-summary control) | private | Private | 2026-06 (pre-register) |
| R6 | cross-encoder rerank **+2.5**, and it *demotes* thread-spanning answers | README, landing page | `scripts/eval/bench_thread_reconstruction.py` (rerank arm) | private + paid NVIDIA endpoint | Private ⚠️ | 2026-06 (pre-register) |
| R7 | NVIDIA's stack wins on TREC Legal; mailrag wins on email — "opposite winners" | README, landing page | `scripts/eval/bench_trec.py`, `build_trec_collection.py` | TREC Legal + paid NVIDIA endpoint | Private ⚠️ | 2026-06 (pre-register) |
| R8 | an early **+6pp** gain was half a quantization artifact, worth **+3pp** at matched precision | README | private eval, §-numbered in `EXPERIMENTS.md` | private | Private | 2026-06 (pre-register) |

⚠️ **R6 and R7 are currently unverifiable**: both need `NVIDIA_API_KEY`, which is
not configured. The scripts now fail with an actionable message rather than a bare
assertion, but until a key is supplied these two numbers cannot be re-checked.

## Performance claims

| # | Claim | Where published | Produced by | Status | Last verified |
|---|---|---|---|---|---|
| P1 | `make bench` runs in **1.6 min** (MPS) / **14.7 min** (CPU-only) | README, `BENCHMARK.md`, Makefile | timed `make bench`, M5 Pro 6P+12E / 48 GB, load-gated | **Public** ✅ | 2026-08-12 |
| P2 | `make bench` spends **zero LLM calls** | README, `BENCHMARK.md` | code inspection: `embed_summary=False`, `apply_noise_filter=False`, no profile, retrieval-only scoring | **Public** ✅ | 2026-08-12 |

## Scope claims (things we assert are *not* measured)

| # | Claim | Verified by |
|---|---|---|
| S1 | `make bench` excludes thread reconstruction, summaries, rerank and noise cleanup | `BENCHMARK.md` exclusion table; `bench_public.py` flags |
| S2 | Enron-QA carries no conversation linkage, so thread reconstruction cannot be scored on it | dataset schema: `email / questions / path / user / …` |

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
