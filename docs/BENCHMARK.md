# `make bench`: the number you can check yourself

Most of this project's retrieval figures are measured on a **private** mailbox. That
is the corpus the system is actually for, and it is not something a reader can be
handed. So those numbers are, unavoidably, author-reported.

`make bench` exists to make the core claim checkable anyway: it scores retrieval on
the **public** [Enron-QA](https://huggingface.co/datasets/MichaelR207/enron_qa_0922)
dataset, needs no API key and no private data, and prints a real recall table in a
few minutes.

### Why the corpus is from 2001

Why benchmark on Enron rather than something modern? Because, as far as I have been
able to establish, there is nothing else. Every public email corpus exists because
someone lost control of one, which [`WHY_LOCAL.md`](WHY_LOCAL.md) sets out in full.

There is a second reason specific to *this* benchmark, and it rules out the one
consented alternative. The corpora anyone published on purpose are **mailing lists**:
Apache, LKML, the W3C collection behind TREC Enterprise. They are the wrong shape for
this problem. List traffic is well written, self-contained and carries its own context,
whereas the hard case in a real mailbox is the terse reply, *"sounds good, go ahead"*,
which means nothing without the messages above it. Benchmarking thread reconstruction
on a mailing list would test the mechanism on precisely the case that needs it least.

So a dataset from 2001 remains the field standard twenty-five years later, and the
nearest alternative is another defunct company's mail behind a licence.

*If you know of a corpus that fits better, meaning real mailbox correspondence under an
open licence, please [open an issue](https://github.com/fmasi/mailrag/issues).*

```bash
docker compose up -d          # Qdrant
make bench                    # 2 000 docs / 360 queries
make bench SIZE=large         # 10 000 docs / 360 queries (harder)
```

First run downloads bge-m3 (~2 GB) and the Enron-QA test split.

### How long it takes

Measured rather than estimated, on `make bench` at the default size, covering build
**and** scoring:

| device | wall clock |
|---|---|
| Apple Silicon GPU (MPS) | **1.6 min** |
| CPU only | **14.7 min** |

Hardware: Apple M5 Pro (6 performance + 12 efficiency cores, 48 GB), macOS 26.5.2,
torch 2.13, Python 3.13. Measured on a working laptop with an editor and Docker
running, rather than a dedicated benchmark host, with runs gated on a 1-minute load
average below 2.5 and no other GPU work in flight. Two MPS repetitions agreed to
within 2.3%. Treat these as indicative of the shape rather than a spec sheet.

**The 9× gap is the number to plan around.** PyTorch defaults to 6 threads here, using
the performance cores and leaving the efficiency cores idle, and the scoring half is
latency-bound regardless: 720 sequential single-query searches, which no amount of
parallelism shortens. Set `RAG_EMBED_DEVICE=cpu|mps|cuda` to force a device;
otherwise it picks cuda > mps > cpu.

### It costs no LLM calls

The benchmark spends **zero** LLM tokens. It builds with `embed_summary=False` and
`apply_noise_filter=False`, passes no corpus profile (so the Pass-2 cache path never
runs), and scores with retrieval only, so no answer generation and no cross-encoder. The
times above are embedding plus Qdrant and nothing else.

That is also the scope boundary. Recall@k measures the **retrieval layer** rather than
the product. `make demo` spends no LLM calls either, because its summaries are committed
as fixtures. `./mailrag ask` is the one that does, since it generates an answer. Needing
no key is exactly why anyone can reproduce these two commands.

## What it measures

Two arms, both fully local:

| arm | what it is |
|---|---|
| `dense` | bge-m3 dense vectors only, the plain-RAG baseline |
| `dense+sparse` | bge-m3 dense **plus learned-sparse**, fused with RRF, the default path |

Recall@k is **document** recall: how often the email the question was written from
appears in the top *k* distinct emails returned. Chunks are collapsed to their message
first, so an email contributing three chunks counts once. Without that collapse every
arm's recall would be inflated by chunk duplication.

## Results

Standard set — 2 000 documents, 360 queries:

| arm | R@1 | R@5 | R@10 |
|---|---|---|---|
| dense | 87.5 [83.7, 90.5] | 94.4 [91.6, 96.4] | 95.3 [92.6, 97.0] |
| **dense + learned-sparse** | **90.0 [86.5, 92.7]** | **97.5 [95.3, 98.7]** | **98.6 [96.8, 99.4]** |

Large set — same 360 queries against a 5× bigger distractor pool (10 000 documents):

| arm | R@1 | R@5 | R@10 |
|---|---|---|---|
| dense | 76.1 [71.4, 80.2] | 90.0 [86.5, 92.7] | 92.5 [89.3, 94.8] |
| **dense + learned-sparse** | **80.6 [76.2, 84.3]** | **94.4 [91.6, 96.4]** | **96.7 [94.3, 98.1]** |

Brackets are **95% Wilson score intervals**, not `mean ± z·SE`. The textbook Wald
interval misbehaves exactly where this benchmark lives. At p̂ = 0.975, n = 360 the true
interval is [95.3, 98.7], asymmetric by a full percentage point, and at p̂ = 1.0 Wald
collapses to ±0.00 and claims perfect certainty from 360 samples. Bounds are printed
rather than a single ± for the same reason. Near the ceiling there is no honest
symmetric summary.

### Significance

The marginal intervals above **overlap**, which invites the conclusion that the
difference is not real. That conclusion would be wrong, and the reason is worth
stating: both arms answer the *same* 360 queries, so the comparison is **paired**.
Overlapping marginal intervals discard the pairing and are badly conservative for
paired data. `make bench` therefore also reports an exact **McNemar** test over the
queries where the arms disagree:

| set | k | sparse fixes | sparse breaks | McNemar exact p |
|---|---|---|---|---|
| standard | @1 | 11 | 2 | 0.0225 |
| standard | @5 | 12 | 1 | **0.0034** |
| standard | @10 | 12 | 0 | 0.0005 |
| large | @1 | 19 | 3 | 0.0009 |
| large | @5 | 17 | 1 | **0.0001** |
| large | @10 | 15 | 0 | 0.0001 |

The effect is one-sided at every cut: learned-sparse rescues 11–19 queries and
breaks at most 3.

**The direction is the result.** The sparse advantage at R@5 is +3.1pp at 2 000
documents and **+4.4pp at 10 000**. It grows as the task gets harder, and the paired
p-value falls by an order of magnitude with it. A retrieval trick that only helps on an
easy corpus is not worth shipping. This one earns more as the distractor pool widens.

## Why the corpus is not smaller

An earlier draft used 500 documents so the benchmark would finish in about a minute.
At that size both arms sat at the ceiling, 96.7% against 95.3% at R@1 with dense
*ahead*, which is noise rather than a finding. Retrieval difficulty is set by the size
of the distractor pool, so a corpus small enough to be trivially fast is also small
enough to measure nothing.

Query count is nearly free by comparison. Scoring 360 queries across both arms costs
about 75 seconds, against roughly 100 seconds to build the 2 000-document index. So
queries are spent generously to tighten the interval rather than rationed for speed.

## What this proves — and what it does not

**Read this before quoting the number.**

`make bench` validates **one layer** of mailrag: hybrid retrieval. It is not a
reproduction of the headline result, and it is not a demonstration of the system as
a whole. Every other lever is switched off:

| lever | private-eval value | in `make bench`? | why not |
|---|---|---|---|
| Thread reconstruction | **+29.1** recall@5, the flagship | ❌ | Enron-QA rows carry no conversation linkage. The schema is `email / questions / path / user / …`, where `path` is a per-user mailbox path. **There are no threads to reconstruct.** Measured by `make demo` instead. |
| Contextual summaries | **+12.8** recall@5 | ❌ | Built with `embed_summary=False`. Generating summaries live would need an LLM and forfeit the no-key property. Measured by `make demo` instead, from committed fixtures. |
| Cross-encoder rerank | +2.5 recall@5 | ❌ | Needs a paid NVIDIA endpoint. A benchmark whose headline requires the reader to hold an API key is not a public benchmark. |
| Noise cleanup | precision rather than recall | ❌ | Built with `apply_noise_filter=False`. The corpus *is* the benchmark, so dropping documents would change what is being scored. |
| **Hybrid dense + learned-sparse** | — | ✅ | **+3.1 / +4.4 pp, the number above** |

So the honest summary is:

- **What it proves.** The retrieval floor is soundly built. The learned-sparse leg is
  worth having, the advantage is statistically real under a paired test, and it *grows*
  as the corpus gets harder. It also shows the project's numbers are stated in a
  falsifiable form: committed fixtures, interval estimates, a paired significance test,
  declared omissions, and measured timings on named hardware.
- **What it does not prove.** That mailrag beats a generic RAG on email. The headline
  45.6% → 93.3% ladder rests on the **private** corpus and cannot be reproduced here, so
  a reader should treat it as author-reported.

The excluded levers are measured in the private harness and reported in
[`EXPERIMENTS.md`](EXPERIMENTS.md). Two of them are no longer excluded everywhere:
`make demo` now measures both contextual summaries and thread reconstruction on public
data, which is the subject of the next section.

## The other public number: `make demo`

`make bench` scores the retrieval layer. `make demo` answers a different question, on a
different corpus, and the two are easy to confuse.

The demo builds **two** indexes over the same 1,200 public Enron emails. One embeds each
message as it stands. The other embeds it together with a short summary of what came
before it in its conversation. Same embedder, same questions, no API key, and everything
it needs is committed under [`eval/demo/`](../eval/demo/).

Then it asks two question sets, because findability and completeness are not the same
property.

**99 single-message questions**, each generated to be answerable from one specific
message and vetted by a separate validator:

| index | R@1 | R@5 | R@10 |
|---|---|---|---|
| plain | 37.4% | 60.6% | 74.7% |
| with thread context | **50.5%** | **73.7%** | **80.8%** |

Both arms answer identical queries, so the paired test is the one that matters. At R@5
context fixes 16 questions and breaks 3, McNemar exact **p = 0.0044**.

**73 spanning questions**, whose answers genuinely need several messages. This is the
case a single-message benchmark cannot see at all:

| measurement | result |
|---|---|
| the right conversation is found | **T@1 91.8%** · **T@5 97.3%** |
| how much of it the top-5 *messages* give you | **52.6%** |
| how much thread expansion gives you | **100%**, whenever the thread is found |

That 52.6% is the number worth sitting with. A generic RAG hands you half the
conversation, and it tends to be the half the answer is missing from.

**Caveats, since they matter here.** Enron carries no `In-Reply-To` headers at all, so
conversations are derived from normalised subject plus shared participants, and the
measured false-merge rate for that derivation is row S4 in [`CLAIMS.md`](CLAIMS.md).
Reranking and noise cleanup are not measured by the demo either. Absolute figures move
about 3pp between runs because Qdrant rebuilds its HNSW graph each time, so read the
direction and the significance rather than the third digit.

Runtime is under two minutes on an Apple-silicon GPU. Expect roughly 15 on CPU, at the
same 9x penalty measured for `make bench` below.

**What remains private.** Thread reconstruction and contextual summaries are now both
publicly demonstrated, on a corpus anyone can download. What is still author-reported is
their **magnitude on a real mailbox**, where 32,000 emails is a harder target than 1,200,
plus two levers that cannot be made public: reranking needs a paid endpoint, and the TREC
comparison needs data that cannot be redistributed.

## How reproducibility is pinned

The corpus and query set are **committed**, not re-derived at run time:

```
eval/public/enron_qa_standard_corpus.txt     one Enron-QA path per line
eval/public/enron_qa_standard_queries.jsonl  {query, answer_path, category}
eval/public/enron_qa_large_corpus.txt
eval/public/enron_qa_large_queries.jsonl
```

The benchmark selects corpus rows by **explicit path**, not by replaying a seeded
shuffle. A seeded shuffle only reproduces if the upstream dataset's row *order* never
changes, which nothing guarantees; a path manifest is stable under any reordering,
and it makes "is the gold document actually in the corpus?" something you can verify
by reading the files rather than trusting a PRNG. A unit test asserts that every
query's gold path is present in its corpus manifest.

To regenerate the fixtures (only needed if the upstream dataset changes):

```bash
python -m scripts.eval.gen_public_benchset
```

## Files

| path | role |
|---|---|
| [`scripts/eval/bench_public.py`](../scripts/eval/bench_public.py) | the benchmark |
| [`scripts/eval/gen_public_benchset.py`](../scripts/eval/gen_public_benchset.py) | one-time fixture generator |
| [`tests/test_bench_public.py`](../tests/test_bench_public.py) | scoring-logic unit tests |
| `eval/public/` | the committed corpus manifests and query sets |
