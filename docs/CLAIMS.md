# Claims register

Every number this project publishes, what produced it, and when it was last
verified.

**Why this file exists.** The README and the landing page state a dozen figures.
Most come from one-off scripts run months ago against a private corpus. Without a
register, "is that still true?" is unanswerable, and the honest answer drifts
silently as the code, the models and the dependencies move underneath it. A
published number with no traceable source is a claim, not a result.

It also guards a subtler failure. When the public benchmark (now `just bench`) shipped, its
docs listed two omissions (rerank, thread reconstruction) and missed a third — contextual
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
| R1 | dense R@5 **94.4** [91.6, 96.4] vs dense+sparse **97.5** [95.3, 98.7]; McNemar p=0.0034 (sparse fixes 12, breaks 1). R@1 87.5 vs 90.0 and R@10 95.3 vs 98.6, with intervals, in `BENCHMARK.md` | README, landing page, methods page, `BENCHMARK.md` | `scripts/eval/bench_public.py` | public Enron-QA, 2 000 docs / 360 q | **Public** ✅ | 2026-08-12 |
| R2 | large set: dense **90.0** vs dense+sparse **94.4**; p=0.0001 | README, landing page, methods page, `BENCHMARK.md` | `scripts/eval/bench_public.py --size large` | public Enron-QA, 10 000 docs | **Public** ✅ | 2026-08-12 |
| R2b | contextual summaries: plain R@5 **60.6** vs with-context **73.7**; McNemar p=0.0044 (context fixes 16, breaks 3). R@1 37.4 vs 50.5, R@10 74.7 vs 80.8 | README, landing page, methods page, `BENCHMARK.md`, `just demo` | `scripts/demo.py` | public Enron, 1 200 docs / 99 validated questions | **Public** ✅ | 2026-08-14 |
| R2c | thread reconstruction, public: **T@5 97.3%** on spanning questions; message-level retrieval surfaces only **52.6%** of the answering conversation at top-5 | README, landing page, methods page, `BENCHMARK.md`, `just demo` | `scripts/demo.py` | public Enron, 1 200 docs / 73 spanning questions | **Public** ✅ | 2026-08-14 |
| R3 | plain-dense baseline **45.6%** R@5 | README, landing page, methods page, `CASE_STUDY.md`, fmasi.eu blog | `scripts/eval/bench_avc.py` dense arm over the body-only control (built by `build_bodyonly_collections.py`) | private ~32k mailbox, 360 q | Private | 2026-06 (pre-register) |
| R3b | learned-sparse adds **+3.3** R@5 on the body-only control (45.6 → 48.9). On the contextual collection the same comparison read +4.8, partly credited with the summaries' work | landing page, methods page, `CASE_STUDY.md`, fmasi.eu blog | `scripts/eval/bench_avc.py`, dense vs hybrid arm, body-only control vs the default contextual collection | private ~32k mailbox, 360 q | Private | 2026-06 (pre-register) |
| R4 | thread reconstruction **64.2 → 93.3** (+29.1) | README, landing page, methods page, `CASE_STUDY.md`, fmasi.eu blog | `scripts/eval/bench_thread_reconstruction.py` | private, `work-rag-ctx-threadaware` | Private ✅ **re-verified, exact** — and **corroborated publicly by R2c** (T@5 97.3% on public Enron) | 2026-08-13 |
| R5 | contextual summaries **+12.8** (48.9 → 61.7) | README, landing page, methods page, `CASE_STUDY.md`, `BENCHMARK.md`, fmasi.eu blog | `scripts/eval/bench_avc.py` hybrid arm, contextual collection vs the body-only control (built by `build_bodyonly_collections.py`) | private | Private — **corroborated publicly by R2b** (+13.1pp R@5 on a different corpus) | 2026-06 (pre-register) |
| R6a | cross-encoder rerank **+2.5** R@5 overall | README, landing page, methods page, `CASE_STUDY.md`, `BENCHMARK.md` | `scripts/eval/bench_thread_reconstruction.py` (rerank arm) | private + paid NVIDIA endpoint | Private ✅ **re-verified, exact** (61.7 → 64.2) | 2026-08-13 |
| R6b | rerank *demotes* the answer on **thread-spanning** queries | **withdrawn** — was in README, landing page, `CASE_STUDY.md`, `RETRIEVAL_GUIDE.md`; removed from all four 2026-08-17 | — | — | ⚠️ **NOT reproduced on recall** — see below | 2026-08-13 |
| R7 | NVIDIA's stack wins on TREC Legal; mailrag wins on email — "opposite winners" | landing page, methods page, `CASE_STUDY.md`, fmasi.eu blog (no longer in README) | `scripts/eval/bench_trec.py`, `build_trec_collection.py` | TREC Legal + paid NVIDIA endpoint | Private ⚠️ | 2026-06 (pre-register) |
| R8 | an early **+6pp** gain was half a quantization artifact, worth **+3pp** at matched precision | methods page, `CASE_STUDY.md` (no longer in README) | private eval, §-numbered in `EXPERIMENTS.md` | private | Private | 2026-06 (pre-register) |
| R9 | expanding hits to their whole thread roughly **doubles** answer coverage: the answer email is returned for 44% of queries without threads, 82% with them (terse: 33% → 78%) | README | `EXPERIMENTS.md` §9 Lens 2 (`scripts/eval/run_arms.py`, `sweep_thread_n.py`) | private, 45 queries | Private | 2026-05-29 |
| R10 | with the answer in context the 26B model answered correctly about **88%** of the time (67% correct ÷ 76% coverage), and at matched precision a 4B model tied it (e4b@8bit 1.88 vs 26b@6bit 1.91 mean grade) | landing page, `CASE_STUDY.md` | `EXPERIMENTS.md` §9 Lens 3 (`scripts/eval/e2e_context.py`, `e2e_answer.py`, `e2e_judge.py`) | private, 45 queries | Private | 2026-05-29 |
| R11 | the local LLM judge agrees with a stronger reference at Cohen's κ **0.52**, Spearman 0.74, on 514 pooled pairs, and both pre-registered decisions match | methods page, `CASE_STUDY.md` | `scripts/eval/calibrate.py` (`EXPERIMENTS.md` §9) | private, 514 pairs | Private | 2026-05-29 |
| R12 | roughly **40%** of the eval questions depend on a terse message. This is the query set's design (144 terse of 360), not a measured property of mail | README | the 360-query set's category split (`eval/out/queries_360.jsonl`), counted in the R6b table below | private, 360 q | Private | 2026-08-13 |
| R13 | on some content queries whose terms are not in the subject line, relevant-in-top-5 went **0 → 2 → 5** (dense → hybrid → hybrid+rerank). Directional, not a rate | `RETRIEVAL_GUIDE.md` | 3-way mode probe, `EXPERIMENTS.md` §7 | private work corpus, 12 content-rich queries | Private | 2026-05-28 |

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
category" — a different metric on a different run. Every surface that published it
stated it directly beside the `+2.5 R@5` figure, which reads as though both numbers
come from the same measurement.

`EXPERIMENTS.md:151` separately describes the demotion as affecting *contentless
**terse*** emails, not spanning ones — which matches this data far better than the
withdrawn wording did.

**Corrected 2026-08-17** in all four places it was published (README, landing page,
`CASE_STUDY.md`, `RETRIEVAL_GUIDE.md`). The row stays in this register rather than
being deleted: a withdrawn claim is part of the record, and anyone who read the old
wording deserves to find out what replaced it. Tracked in #128.

⚠️ **R7 remains unverifiable by measurement** until its collections are rebuilt
(`trec-bge` / `trec-e5` are no longer in Qdrant). The key is now available, so it
is unblocked — it just needs the build step first.

## Cleanup claims

| # | Claim | Where published | Produced by | Corpus | Status | Last verified |
|---|---|---|---|---|---|---|
| N1 | corpus-derived regex rules catch **65%** of the LLM's noise (7,875) and miss **35%** (4,248), with 120 false drops | landing page, `CASE_STUDY.md` | regex rules compared with the Pass-2 LLM labels, `EXPERIMENTS.md` §2 | private work corpus, 31,969 emails | Private | 2026-05-28 |
| N2 | on a regex-only corpus **21%** of queries surface at least one noise email in their top-3 (~11% of top-3 slots) | README, `CASE_STUDY.md` | top-N retrieval with a clean-vs-noise membership check, `EXPERIMENTS.md` §13 | private work corpus, 360 q | Private | 2026-06-01 |
| N3 | the corporate rubric flagged **87.6%** of a personal archive as noise against **61.5%** for a rubric calibrated on that corpus, caught by a **~200-email** calibration sample | README, landing page, `CASE_STUDY.md`, `VERBS.md` (as ~88%) | Pass-2 under each rubric, `EXPERIMENTS.md` §14. Sample size is the `calibrate` default (`src/pipeline/calibrate.py`) | private personal archive, ~25k emails | Private | 2026-06-05 |

## Performance claims

| # | Claim | Where published | Produced by | Status | Last verified |
|---|---|---|---|---|---|
| P1 | `just bench` runs in **1.6 min** (MPS) / **14.7 min** (CPU-only) | README, `BENCHMARK.md`, justfile | timed `just bench`, M5 Pro 6P+12E / 48 GB, load-gated | **Public** ✅ | 2026-08-12 |
| P2 | `just bench` spends **zero LLM calls** | README, `BENCHMARK.md` | code inspection: `embed_summary=False`, `apply_noise_filter=False`, no profile, retrieval-only scoring | **Public** ✅ | 2026-08-12 |
| P3 | a local answer model needs about **9 GB** of RAM/VRAM (e4b at 8-bit measured 9.0 GB loaded) | README | LM Studio's loaded-model stats, `EXPERIMENTS.md` §9 model table (speed via `scripts/eval/bench_models.py`) | Private | 2026-05-29 |
| P4 | the test suite has about **1,970** test functions | landing page | `git grep -c "def test_" main -- tests` (1,971) | **Public** ✅ | 2026-11-03 |

## Attachment-noise claims (2026-08-19)

All measured on the private work corpus, so **author-reported**: reproducible by anyone
with an equivalent mailbox and the commands below, not by a stranger. Recorded here
because these figures appear in `MCP_SERVER.md`, the methods page and `EXPERIMENTS.md` §16 and §17.

| # | Claim | Produced by | Corpus | Status | Last verified |
|---|---|---|---|---|---|
| A1 | **61%** of attachment rows are decoration (signature blocks, disclaimer images, newsletter headers, spacers) | `store.list_for(include_boilerplate=False)` over the built store | private work corpus, 45 454 rows / 6 761 blobs | Author-reported | 2026-08-19 |
| A2 | Counting distinct *messages* rather than *threads* misclassifies **237 of 659 blobs (36%)** — images quoted down one reply chain | ad-hoc SQL over `attachments`, verified by opening the images | same | Author-reported | 2026-08-19 |
| A3 | Measuring OCR text rescues **43 blobs** the metadata heuristic hid | `classify_blobs` then verdict diff against the heuristic | same | Author-reported | 2026-08-19 |
| A4 | Bulk measurement costs **2 570 blobs in 240s, 0 failures** (~0.09s/blob, tesseract) | `./mailrag attachments build` | same | Author-reported | 2026-08-19 |
| A5 | Tesseract against LLM vision on identical images: **0.41s vs 8.49s** (table), 0.07s vs 1.61s (header) — 16–20× | `scratchpad` benchmark over three labelled blobs | same, 3 images | Author-reported ⚠️ small n | 2026-08-19 |
| A6 | LLM vision inflates a 3-word header from **22 to 159 characters** via its `DESCRIPTION:` preamble, crossing the text-rich threshold | same benchmark | same | Author-reported | 2026-08-19 |
| A7 | Pruning decoration blobs would reclaim **13 MB of a 3.4 GB store (0.4%)** — sha256 dedup already collapses repeats | SQL over distinct `sha256` sizes | same | Author-reported | 2026-08-19 |
| A8 | Bulk-header tagging rates differ by corpus: work **41.2% `is_bulk`** against personal **7.2%** | Qdrant `points/count` with payload filter | `work-rag-ctx-threadaware-v2` (132 269 pts), `personal-rag` (16 743 pts) | Author-reported | 2026-08-19 |
| A9 | Text density separates pictorial from text-bearing documents by ~50x: 10 text PDFs at **99,000-225,000 chars/MB** against **1,869** (diagram PDF) and **66** (21 MB deck). The sparse-text cut of 10,000 chars/MB is a design value (`_SPARSE_TEXT_CHARS_PER_MB` in `src/mcp_server/server.py`) placed in that gap, about 10× below the text-bearing PDFs and 5× above the diagram PDF | ad-hoc extraction over store blobs, tesseract | private work corpus | Author-reported | 2026-08-20 |
| A10 | **100%** of a 4,000-file sample of this corpus begins with an mbox `From ` envelope, which silently voids every header if not stripped | byte inspection over `_discover_eml` sample | private work corpus | Author-reported | 2026-08-20 |
| A11 | Calibration cases for the decoration rule: one **6.5 KB** logo carried by **2,273** messages is a single blob, a pasted "401 Access is denied" screenshot measures **29** characters and is content, a disclaimer image measures **748** characters across **829** threads, signature blocks measure ~195 characters across **61** threads, and the **15–25** thread band is ambiguous (a deadline table at 15, a disclaimer at 18) | store inspection and `classify_blobs` verdicts, `EXPERIMENTS.md` §16 | private work corpus | Author-reported | 2026-08-19 |
| A12 | Before the per-collection split, a single shared attachment store had **four** thread ids present in both a work and a personal corpus, so `list_attachments` on either returned the other's attachments | inspection of the live shared store, recorded in commit `43173b0` | private work + personal stores | Author-reported | 2026-08-20 |

> **A8 is not evidence that work mail is bulkier.** The two collections were built under
> different policies: work v2 tags and keeps, while `personal-rag` was built with
> `--embed-summary`, which drops noise ≥ 0.7 at build time — so its low residual rate means
> the noise was already removed. Both are chunk-weighted, not per-email. Recorded because
> the raw pair is easy to misread.

> **A5 rests on three images.** Enough to choose the cheap engine by an order of magnitude,
> not enough to publish a per-image latency figure. What A6 establishes — that the two
> engines' character counts are not comparable without normalisation — does not depend on n.

## MCP server claims

Measured on the author's own mailbox while fixing the MCP tools, so **author-reported**.

| # | Claim | Where published | Produced by | Corpus | Status | Last verified |
|---|---|---|---|---|---|---|
| M1 | before output was bounded, one `search_email` call (`top_k=8`) emitted **130,196** characters (~130 K) | `MCP_SERVER.md`, `RELEASE_NOTES.md` | a live call against the pre-fix server, issue #84 | private mailbox | Author-reported | 2026-07-13 |
| M2 | the old `get_thread`, which re-ran vector search with the id as the query, resolved **3 of 12** ids that `search_email` had returned (~25%) | `MCP_SERVER.md`, `RELEASE_NOTES.md`, `EXPERIMENTS.md` §15.3 | 12 ids from 4 `search_email` calls fed back to `get_thread`, issue #109 | `personal-rag` (16,586 points) | Author-reported ⚠️ small n | 2026-08-08 |
| M3 | `grep_email` decodes about **2 ms** per message (regex **1.6** vs literal **1.8** ms/file), over a corpus of **73,219** messages / **11 GB**. A rare regex once ran an agent's call into the client's **30-minute** timeout. That is the client's limit, not a mailrag setting; mailrag's own bound is `max_seconds` (default 60, hard cap 900) | `MCP_SERVER.md` | timing on the live `.eml` root and the incident behind commit `8ff8e66` | private `.eml` root | Author-reported | 2026-08-19 |

## Scope claims (things we assert are *not* measured)

| # | Claim | Verified by |
|---|---|---|
| S1 | `just bench` excludes thread reconstruction, summaries, rerank and noise cleanup | `BENCHMARK.md` exclusion table; `bench_public.py` flags |
| S2 | Enron-QA carries no conversation linkage, so thread reconstruction cannot be scored on it *as shipped* | dataset schema: `email / questions / path / user / …`; 400-row sample shows `Message-ID` in 1, `In-Reply-To` in 0 |
| S3 | The **CMU Enron maildir** has no threading headers either — but conversations are derivable | 8 000 real messages: `In-Reply-To` 0.0%, `References` 0.0%, yet `Re:`/`Fw:` subjects 64.3%. Deriving by normalised subject + shared participant puts **50.2%** of 19 530 messages in a multi-message thread (largest 59). Measured 2026-08-14 |
| S4 | Derived threads carry a **measured false-merge rate**, concentrated in generic subjects and long spans | 19 530 Enron messages: 2.2% of same-thread pairs share **no** participant (transitive chaining, 1.0% of threads); 11.1% of threads span >30d and **1.4% span >1 year**; generic subjects ("hey", "lunch", "meeting") account for 55 threads / 2.3% of threaded messages. Worst case observed: "happy hour", 36 messages over 391 days across 16 participants — a recurring invite, not a conversation. Measured 2026-08-14 |

## What is still author-reported, and what would close it

Both of the ladder's biggest levers are now publicly demonstrated. Contextual embedding
(R2b) and thread reconstruction (R2c) both run inside `just demo`, on 1,200 public Enron
emails anyone can download, with a paired significance test on each.

What stays author-reported is the **magnitude on a real mailbox**. A 32,000-email archive
is a harder target than 1,200 emails, and the private ladder's 45.6 → 93.3 cannot be
re-run by a stranger. R3, R3b, R4, R5 and R8 all sit there.

Two levers cannot be made public at all as things stand. Reranking (R6a) needs a paid
endpoint, and a benchmark whose headline requires the reader to hold an API key is not a
public benchmark. The TREC contrast (R7) needs a corpus that cannot be redistributed.

The gap that could realistically close next is R7, which is unblocked but needs its
collections rebuilt before it can be re-measured.

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
is `just bench`. Data locations are environment-overridable — see
[`scripts/eval/_paths.py`](../scripts/eval/_paths.py):

```bash
MAILRAG_EVAL_QUERIES=eval/out/queries_360.jsonl \
  python -m scripts.eval.bench_thread_reconstruction      # R4, R6 (needs NVIDIA_API_KEY)

MAILRAG_EVAL_TREC=~/msgvault-eval-proof/trec \
  python -m scripts.eval.bench_trec                       # R7 (needs NVIDIA_API_KEY)
```
