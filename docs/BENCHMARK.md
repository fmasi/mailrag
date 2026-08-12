# `make bench` — the number you can check yourself

Most of this project's retrieval figures are measured on a **private** mailbox. That
is the corpus the system is actually for, and it is not something a reader can be
handed. So those numbers are, unavoidably, author-reported.

`make bench` exists to make the core claim checkable anyway: it scores retrieval on
the **public** [Enron-QA](https://huggingface.co/datasets/MichaelR207/enron_qa_0922)
dataset, needs no API key and no private data, and prints a real recall table in a
few minutes.

```bash
docker compose up -d          # Qdrant
make bench                    # 2 000 docs / 360 queries  (~3 min)
make bench SIZE=large         # 10 000 docs / 360 queries (~6 min, harder)
```

First run downloads bge-m3 (~2 GB) and the Enron-QA test split.

## What it measures

Two arms, both fully local:

| arm | what it is |
|---|---|
| `dense` | bge-m3 dense vectors only — the plain-RAG baseline |
| `dense+sparse` | bge-m3 dense **+ learned-sparse**, fused with RRF — the default path |

Recall@k is **document** recall: how often the email the question was written from
appears in the top *k* distinct emails returned. Chunks are collapsed to their
message first, so an email that contributes three chunks counts once — otherwise
every arm's recall would be inflated by chunk duplication.

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
interval misbehaves exactly where this benchmark lives: at p̂ = 0.975, n = 360 the
true interval is [95.3, 98.7] — a full percentage point asymmetric — and at p̂ = 1.0
Wald collapses to ±0.00, claiming perfect certainty from 360 samples. Bounds are
printed rather than a single ± for the same reason: near the ceiling there is no
honest symmetric summary.

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
documents and **+4.4pp at 10 000** — it grows as the task gets harder, and the
paired p-value falls by an order of magnitude with it. A retrieval trick that only
helps on an easy corpus is not worth shipping; this one earns more as the distractor
pool widens.

## Why the corpus is not smaller

An earlier draft used 500 documents so the benchmark would finish in about a minute.
At that size both arms sat at the ceiling — 96.7% vs 95.3% at R@1, with dense
*ahead*, which is noise rather than a finding. Retrieval difficulty is set by the
size of the distractor pool, so a corpus small enough to be trivially fast is also
small enough to measure nothing.

Query count, by contrast, is nearly free: scoring 360 queries across both arms costs
about 75 seconds against roughly 100 seconds to build the 2 000-document index. So
queries are spent generously to tighten the interval rather than rationed for speed.

## What it deliberately does not measure

Two real parts of the system are **absent on purpose**, because including them would
produce a number a reader cannot reproduce:

- **Thread reconstruction** — the single biggest lever in the private eval (+29.1
  recall@5). Enron-QA rows carry no conversation linkage at all; the schema is
  `email / questions / path / user / …` where `path` is a per-user mailbox path.
  There are no threads to reconstruct, so the benchmark cannot score it.
- **Cross-encoder rerank** — needs a paid NVIDIA endpoint. A benchmark whose headline
  requires the reader to hold an API key is not a public benchmark.

Both are measured in the private harness and reported in
[`EXPERIMENTS.md`](EXPERIMENTS.md). This file only claims what a stranger can
regenerate.

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
