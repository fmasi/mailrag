# RAG Study Guide

*[← docs index](INDEX.md) · [README](../README.md) · the measured results behind this guide live in [`EXPERIMENTS.md`](EXPERIMENTS.md)*

A working reference for the retrieval techniques behind this project — written to be
**interview-ready**: crisp definitions, the *why*, the trade-offs, and concrete examples
you can talk through out loud. Grown incrementally as we build the system.

> Running example used throughout: searching this corpus for **"SPP"** (the
> *Salon Partner Programme*). It's a great teaching case because it
> mixes a semantic concept ("partnership / onboarding programme") with rare exact
> tokens ("SPP", order IDs like `00451823`) — which is exactly where the
> different retrieval methods diverge.

---

## Contents

0. [Start here: the big picture (problem + plain glossary + drift)](#start-here-the-big-picture-plain-language-primer)
1. [Retrieval fundamentals: dense vs sparse](#1-retrieval-fundamentals)
2. [BM25 — classic lexical search](#2-bm25--classic-lexical-search)
3. [Learned sparse retrieval (SPLADE, bge-m3 sparse)](#3-learned-sparse-retrieval)
4. [bge-m3 — the M3 capabilities](#4-bge-m3--the-m3-capabilities)
5. [Hybrid search & Reciprocal Rank Fusion (RRF)](#5-hybrid-search--rrf)
6. [ColBERT & late interaction](#6-colbert--late-interaction)
7. [Serving embeddings on Apple Silicon (UMA, MPS vs MLX, M5)](#7-serving-embeddings-on-apple-silicon)
8. Hybrid retrieval, fusion & reranking — how this project queries (measured; see also `EXPERIMENTS.md` §6–8)
9. [Thread-aware retrieval (small→big expansion)](#thread-aware-retrieval-smallbig-expansion) — what/why/how + token bounding
10. [Roadmap / coming next](#roadmap--coming-next) — evaluation metrics, chunking, RAPTOR, late chunking

---

## Start here: the big picture (plain-language primer)

New to the query side? Read this first — the rest of the guide goes deep, but this section
explains *what we're solving* and gives a plain-language glossary you can hold in your head.

### The problem we're actually solving
There are tens of thousands of emails in a vector database. The goal: **you type a question
or topic, and the system returns the *most relevant* emails** (an LLM can then answer using
them). The entire difficulty is in that one word — *relevant*. There are several ways to
measure relevance, each good at some things and bad at others, so the work is finding the
combination that puts the *right* emails at the top.

### How a computer finds "relevant" at all
A computer can't compare meaning directly, so we turn every email — and your query — into a
list of numbers (a **vector**) that represents it. "Relevant" then means "whose numbers are
closest to the query's numbers." There are **two different ways to make those numbers**, good
at opposite things:

- **Dense vector — the "meaning" method.** Captures the overall gist. Great at synonyms and
  paraphrase (it knows *reschedule* ≈ *new time proposed*). Weak at exact rare terms — it may
  not lock firmly onto a product code or version number.
- **Sparse vector — the "keyword" method.** Like a smart keyword match: it emphasises the
  important *words*. Great at exact terms, acronyms, and codes. Blind to meaning — if you don't
  use the same word, it misses.

Think of two assistants: one *understands what you mean* but is vague on specifics; the other
is a *literal keyword bloodhound* but doesn't get nuance.

### Hybrid + fusion — use both assistants
Each method returns its own ranked list, and they disagree. **Fusion** is the recipe for
merging two ranked lists into one. We use **Reciprocal Rank Fusion (RRF)**: an email that
*both* methods rank highly gets boosted. So **hybrid search = the best of meaning-match and
keyword-match.** (RRF fuses *ranks*, not scores, so it needs no score calibration — see §5.)

### Reranking — a slow, smart second opinion
Everything above is **fast but approximate**: each email's vector and the query's vector were
computed *separately*, so the model never saw them side by side. A **reranker (cross-encoder)**
is a slower, smarter model that reads the query and **one email together** and scores how
relevant it really is. Far more accurate, but too slow for the whole corpus — so the pattern is
**"retrieve wide, rerank narrow":** the fast methods fetch ~20 candidates, then the reranker
re-reads and reorders just those. (See §6 for the bi-encoder-vs-cross-encoder spectrum.)

### Drift — and why "contextual retrieval" can cause it
A retrieval method **drifts** when it ranks loosely-related results *above* the ones that
precisely answer the query — the top results slide off-target toward generic topical neighbours.

> **Drift, in one line:** loosely-on-topic results creeping above the precise ones.

One experiment in this project, *contextual retrieval*, glues a one-line summary onto each email
*before* embedding it, hoping the extra context helps. The side effect: the summary makes the
email's vector look like a blend of *generic topics*, so a precise query can match emails that
are only vaguely related — i.e. it **increases drift**. A reranker is the standard cure: because
it re-reads the *actual* query against the *actual* email, it pushes drifted-in results back down.

### What we measured (directional, small sample)

> **⚠️ Superseded by the labeled eval — see [`EXPERIMENTS.md` §9](EXPERIMENTS.md).** The reads
> below were early, eyeballed, small-sample. A 45-query labeled eval (3 lenses + LLM-as-judge)
> later **revised two of them**:
>
> *Notation:* `C′` is the **summary-embedded contextual collection** (live: `work-rag-ctx-*`);
> `C` is the cleaned **body-only** collection (live: `work-rag-bodyonly`). See the
> [terminology box in `EXPERIMENTS.md`](EXPERIMENTS.md#terminology-read-this-first), which also
> disambiguates the two senses of "thread-aware" (retrieval expansion vs. summary conditioning).
> - **Reranking helps pointed queries (+2.5 recall@5) but demotes the answer on thread-spanning
>   ones** (and hurt outright under the earlier LLM-judged eval). **Off by default.**
> - **Contextual retrieval (`C′`) was the *best* arm** (ranked and end-to-end) — the "drift"
>   penalty did not reproduce; **`C′` is kept, not retired.**
> - **Thread reconstruction is the headline win — recall@5 62% → 93%** (thread-recall): match a
>   small unit, answer from its whole thread; recommended stack is **`C′` + expand top ~1–3 threads**.
> - **Retrieval coverage (~76%), not the answer model, is the ceiling.**

The original directional reads (kept for the record), comparing **dense → hybrid → hybrid+rerank**
on a cleaned collection (no summaries) vs a summary-embedded one:

- **Reranking is a clear, consistent win — biggest where the fast methods are weakest.** When
  the query's words are *not* in the subject line, dense/hybrid struggle and rerank rescues it
  (on some queries, relevant-in-top-5 went from **0 → 2 → 5** across the three stages).
- **On queries naming a specific entity, all methods already find on-topic emails** (the corpus
  has exact-match threads) **but rerank fixes the *order*,** surfacing the precisely-relevant
  thread above generically-related mail.
- **The summary-embedded collection (C′) is a *bi-directional* trade-off, not a clear win:** it
  clearly *helps* terse/contentless emails (ranks them far higher — its designed purpose) but
  clearly *hurts* literal/precise queries (topic drift). The reranker is the bigger lever for
  content queries. This trade-off is what motivates **thread-aware retrieval over one collection**
  as the cleaner resolution (see the Roadmap below and `EXPERIMENTS.md` §7); a larger *labelled*
  eval would quantify it.
- **Caveat — duplicate results:** the same email/thread can appear several times in the top-K
  (multiple chunks). Grouping by thread (thread-aware retrieval) is the fix — and *is* the dedup.

---

## 1. Retrieval fundamentals

There are two fundamentally different ways to match a query to documents. Understanding
*why each fails where the other succeeds* is the whole game.

### Dense retrieval (semantic / vector search)

An **embedding model** converts text into a single fixed-length vector of numbers
(bge-m3 → 1024 dimensions) that captures **meaning**. Texts with similar meaning land
close together in vector space; you retrieve by **nearest-neighbour** search, usually by
**cosine similarity**.

- ✅ **Understands synonyms & paraphrase.** A query *"time-off rules"* matches an email
  about *"vacation policy"* even with **zero shared words**.
- ✅ Language-agnostic with a multilingual model — a Korean reply can match an English query.
- ❌ **Weak on rare exact tokens.** Product codes, order IDs (`00451823`), acronyms
  (`SPP`), surnames — the model captures the *gist*, not the literal string, so a precise
  identifier can rank surprisingly low.

> **Why "dense"?** The vector is *dense*: (almost) every one of the 1024 dimensions has a
> non-zero value. The meaning is smeared across all of them.

### Sparse retrieval (lexical / keyword search)

Represents text as a **very high-dimensional, mostly-zero** vector — conceptually one
dimension per vocabulary term, non-zero only for the terms actually present. Matching is
about **shared exact terms**, scored by a formula like BM25 (next section).

- ✅ **Nails exact tokens** — names, IDs, error codes, acronyms. Precisely dense's blind spot.
- ❌ **No notion of meaning** — *"time-off rules"* will **not** match *"vacation policy"*:
  no shared words, no score. This is the classic **vocabulary-mismatch problem**.

> **Why "sparse"?** The vocabulary is huge (tens of thousands of terms) but any one
> document uses only a few hundred — so the vector is mostly zeros, i.e. *sparse*.

**The one-sentence summary for an interview:**
> *Dense search matches meaning; sparse search matches exact words. Each fails where the
> other shines — which is why we combine them (hybrid).*

---

## 2. BM25 — classic lexical search

**BM25** ("Best Match 25") is the decades-old, still-excellent scoring function behind
sparse/keyword search (Elasticsearch, Lucene, etc.). It scores a document for a query by
summing, over each query term, three intuitions:

1. **Term Frequency (TF)** — a document mentioning the term more is more relevant…
   - …but with **saturation**: the 10th occurrence adds far less than the 2nd (diminishing returns), unlike naive TF-IDF.
2. **Inverse Document Frequency (IDF)** — rare terms across the corpus matter more.
   *"the"* is in every email → near-zero weight; *"SPP"* is rare → high weight.
3. **Document-length normalisation** — long documents naturally contain more words, so
   BM25 discounts them to avoid a length bias (tunable via a parameter `b`).

It has two knobs: `k1` (how fast TF saturates, ~1.2–2.0) and `b` (how strongly length is
normalised, ~0.75). No training, no embeddings, extremely fast, and a famously strong
baseline — modern neural systems are often measured *against* BM25.

- **Strength:** exact-match precision; transparent and explainable ("matched these terms").
- **Weakness:** the vocabulary-mismatch problem — it cannot bridge synonyms or paraphrase.

> **Interview soundbite:** *"BM25 is smart keyword matching: term frequency with
> saturation, weighted by how rare the term is, normalised for document length. It's the
> baseline every neural retriever has to beat — and it still wins on exact identifiers."*

---

## 3. Learned sparse retrieval

The modern evolution that keeps BM25's exact-match strength **while softening** the
vocabulary-mismatch weakness. A neural model produces a **sparse vector of term weights**,
but it can also do **term expansion** — assigning weight to *related* terms that don't
literally appear in the text.

- **SPLADE** is the canonical example: a transformer predicts, for each input, a sparse
  set of weighted vocabulary terms (including expansions), stored and searched with the
  same inverted-index machinery as BM25 — so it's fast *and* a bit semantic.
- **Why it matters:** you get keyword-grade precision plus a little synonym awareness, in a
  format that's cheap to index and inspect.

> **Interview soundbite:** *"Learned sparse models like SPLADE are 'neural BM25' — a model
> assigns term weights and even adds related terms, so you keep exact-match precision but
> partly fix vocabulary mismatch, all in an inverted index."*

### bge-m3's sparse output

Crucially for us, **bge-m3 emits a learned-sparse (lexical-weight) vector in the same
forward pass as its dense vector** — so we get a "free" sparse representation tuned to our
corpus, ready for hybrid search, without running a second model.

---

## 4. bge-m3 — the M3 capabilities

`BAAI/bge-m3` is the embedding model this project uses. The **"M3"** name = three
properties, all of which we exploit:

| "M" | Meaning | Why it matters here |
|---|---|---|
| **Multi-Linguality** | 100+ languages in one model | Our corpus has English + Korean + Japanese + Chinese subjects — one model handles all, and cross-lingual matching works (English query → Korean email). |
| **Multi-Functionality** | Produces **dense**, **sparse/lexical**, AND **multi-vector (ColBERT)** representations | One model gives us *both* sides of hybrid search (dense + sparse) in a single pass — no separate sparse model needed. ColBERT-style multi-vector is available for fine-grained reranking later. |
| **Multi-Granularity** | Handles inputs up to **8192 tokens** | Long context enables whole-email / whole-thread embedding and unlocks **late chunking** (embed the long doc once, then derive chunk vectors). |

Other specs worth quoting: **1024-dim** dense vectors, **cosine** similarity, strong on the
MIRACL/MTEB multilingual benchmarks.

- **Dense** → semantic recall.
- **Sparse (lexical)** → exact-match precision (the SPP/order-ID case).
- **ColBERT (multi-vector)** → token-level late-interaction scoring; heavier, typically used
  to rerank a shortlist rather than to search the whole corpus.

> **Interview soundbite:** *"bge-m3 is 'M3' — multilingual (100+ languages), multi-functional
> (dense + learned-sparse + ColBERT from one pass), and multi-granular (8k context). That
> means a single local model gives me both halves of a hybrid index and long-context
> embedding — ideal for a private, offline email RAG."*

---

## 5. Hybrid search & RRF

**Hybrid search** runs **both** dense and sparse retrieval and **merges** their ranked
lists, so the final results inherit dense's semantic recall *and* sparse's exact-match
precision.

The challenge: dense scores (cosine, ~0–1) and BM25 scores (unbounded) live on **different
scales**, so you can't just add them. Two common fixes:

1. **Score normalisation** then weighted sum — fiddly, scale-sensitive, needs tuning.
2. **Reciprocal Rank Fusion (RRF)** — ignore the raw scores entirely; fuse on **rank
   position**. Robust, parameter-light, and the usual default.

### Reciprocal Rank Fusion (RRF)

For each document *d*, sum across the result lists it appears in:

```
RRF_score(d) = Σ_lists  1 / (k + rank_of_d_in_that_list)
```

- `rank` starts at 1 for the top hit.
- `k` is a smoothing constant, **conventionally 60**. A larger `k` flattens the
  contribution of top ranks (less weight on being #1 vs #3); a smaller `k` sharpens it.
- A document gets **two contributions if it appears in both lists** → appearing in *both*
  methods is rewarded, which is exactly what we want.

#### Worked example (k = 60)

Query: *"SPP training materials"*. Suppose:

- **Dense** ranks: 1️⃣ EmailA · 2️⃣ EmailB · 3️⃣ EmailC
- **Sparse** ranks: 1️⃣ EmailC · 2️⃣ EmailD · 3️⃣ EmailA

| Doc | In dense | In sparse | RRF score | Result |
|---|---|---|---|---|
| **A** | rank 1 → 1/61 | rank 3 → 1/63 | 0.0164 + 0.0159 = **0.0323** | top |
| **C** | rank 3 → 1/63 | rank 1 → 1/61 | 0.0159 + 0.0164 = **0.0323** | top |
| **B** | rank 2 → 1/62 | — | **0.0161** | lower |
| **D** | — | rank 2 → 1/62 | **0.0161** | lower |

A and C rise to the top because **each was found by both methods** (semantic *and* lexical
agreement), while B and D — found by only one — sit below. That's RRF doing its job: it
trusts cross-method consensus over any single ranker's confidence.

> **Interview soundbite:** *"Hybrid search fuses dense and sparse results. The scales differ,
> so instead of adding raw scores I use Reciprocal Rank Fusion — score each doc as the sum
> of 1/(k+rank) across both lists, k≈60. It's scale-free and rewards documents that both
> methods agree on. In Qdrant this is a native Query-API primitive."*

### In this project

bge-m3 gives us dense + sparse from one pass; Qdrant stores both as named vectors and runs
RRF fusion server-side. The cross-encoder **reranker** (a later section) then re-orders the
fused shortlist for final precision. The pipeline becomes:

```
query → dense + sparse retrieve → RRF fuse (top ~50) → cross-encoder rerank (top ~10) → LLM
```

---

## 6. ColBERT & late interaction

**ColBERT** = *Contextualized **Late Interaction** over BERT*. It's a third retrieval style
that sits **between** single-vector dense search and a full cross-encoder reranker — and
bge-m3 produces ColBERT-style vectors as its third output, so it's directly relevant to us.

### The core idea: keep one vector *per token*

- A **bi-encoder** (ordinary dense retrieval) squashes a whole document into **one** vector.
  Fast and scalable, but lossy — all the nuance is averaged into 1024 numbers.
- **ColBERT instead keeps a contextual embedding for *every token*.** A document becomes a
  **bag of token vectors**; the query is likewise a bag of token vectors. Each token vector
  is *contextualized* (it "knows" its surrounding sentence), unlike BM25's bare string match.

### Scoring: MaxSim (the "late interaction")

For each **query** token, find its best-matching **document** token and take that similarity;
then sum across query tokens:

```
score(q, d) = Σ_{i ∈ query tokens}  max_{j ∈ doc tokens} ( q_i · d_j )
```

Intuition: *"every word in my query gets to find the single best word in the document to
latch onto."* The token "SPP" in the query can lock onto the contextual "SPP" token in a
long email, even if the rest of the email is unrelated — combining exact-token precision
with semantic, context-aware matching.

**Why "late"?** Query and document are encoded **independently** (so document vectors can be
**precomputed and indexed** — scalable), and the query↔document *interaction* happens only at
the end, in the cheap MaxSim step. Contrast with a **cross-encoder**, where query and
document are fed through the model **together** ("early interaction") — most accurate, but
nothing can be precomputed, so it only works for reranking a small shortlist.

### The interaction spectrum (interview gold)

| | Vectors per doc | Interaction | Speed / scale | Accuracy | Typical use |
|---|---|---|---|---|---|
| **Bi-encoder (dense)** | 1 | none (independent) | ⚡ fast, indexable | good | first-stage retrieval |
| **ColBERT (late)** | N (one per token) | late, MaxSim | medium; indexable but heavy | better | retrieval *or* reranking |
| **Cross-encoder** | — (joint encode) | early (full attention) | 🐢 slow, not indexable | best | rerank top-k only |

### The catch: storage

Storing **N token vectors per document** instead of 1 makes the index much larger.
**ColBERTv2** addresses this with **residual compression** (store centroids + tiny residuals)
and **PLAID** for fast candidate generation — bringing footprint and latency back to
practical levels. Still, ColBERT indexes are heavier than single-vector dense.

### How it fits *this* project

bge-m3 emits ColBERT multi-vectors in the same forward pass as dense + sparse. Because of the
storage cost, the pragmatic use here is **reranking, not first-stage search**: let dense +
sparse + RRF produce a shortlist, then apply ColBERT MaxSim (or a cross-encoder) to reorder
the top candidates. So our pipeline becomes:

```
query → dense + sparse retrieve → RRF fuse (top ~50) → ColBERT / cross-encoder rerank (top ~10) → LLM
```

ColBERT is the "free" reranker we already get from bge-m3; a dedicated cross-encoder
(e.g. bge-reranker-v2-m3) is the heavier, often higher-accuracy alternative — we'll compare
both in the reranking section.

> **Interview soundbite:** *"ColBERT keeps one contextual vector per token instead of one per
> document, and scores with MaxSim — each query token grabs its best-matching document token.
> It's 'late interaction': documents are encoded independently and indexed (scalable), and the
> query-document interaction is deferred to a cheap MaxSim step. It sits between a bi-encoder
> (one vector, fast, coarse) and a cross-encoder (joint encoding, accurate, unscalable). The
> cost is storage — N vectors per doc — which ColBERTv2 compresses with residual encoding and
> PLAID. bge-m3 gives me ColBERT vectors for free, so I use them to rerank a hybrid shortlist."*

---

## 7. Serving embeddings on Apple Silicon

The embedding model has to run *somewhere*. On a Mac you pick a **compute backend**, and
the first thing to internalise is:

> **Backend choice is a speed/memory decision, NOT a quality decision.** The same model
> weights doing the same math produce the same vectors within float-rounding tolerance.
> Embedding quality depends on the **model** and the **precision** (fp32 vs fp16), never on
> whether MPS or MLX drives the GPU. So you can chase performance freely without risking recall.

### Unified Memory Architecture (UMA)

Apple Silicon has **one physical memory pool shared by CPU and GPU** — no separate VRAM, no
PCIe bus to copy tensors across. **Both** PyTorch-MPS and MLX run on UMA; they just exploit
it differently:

- **MLX** is *designed around* UMA: arrays "live in shared memory," you choose the device
  **per operation** (not per tensor), and CPU/GPU ops can touch the same array with **zero
  copies**.
- **PyTorch MPS** keeps the CUDA-style **device-copy** model: `.to("mps")` is a logical
  transfer. Thanks to UMA the bytes don't cross a real bus so it's cheap — but it isn't the
  zero-ceremony model MLX has, and it pays small allocator/sync overheads.

### MPS vs MLX — the optimisation difference

| | **PyTorch MPS** | **MLX** |
|---|---|---|
| What it is | Metal backend bolted onto PyTorch's **eager** model | Apple's array framework, **built for Apple Silicon** |
| Execution | Eager op-by-op | **Lazy** — records & optimises the compute graph, materialises on `eval()` |
| Big weakness | **Operator-coverage gaps** → unsupported ops silently **fall back to CPU** (GPU→CPU→GPU hop stalls the pipeline) | Smaller ecosystem; fewer prebuilt models/servers |
| Ecosystem | Huge (all of PyTorch, HF, Infinity, TEI…) | Growing; great for LLM/embedding inference |

The MPS gotcha to remember: run with `PYTORCH_ENABLE_MPS_FALLBACK=1` so jobs don't crash on
a missing op — **but watch the logs**, because a hot op falling back to CPU is usually the
real bottleneck.

### The M5 twist: Neural Accelerators & Metal 4 TensorOps

The M5 family put a **Neural Accelerator (a dedicated matmul/tensor unit) in every GPU
core** — the headline AI feature (>4× peak GPU AI compute vs M4). The catch is *which
framework actually uses them*:

- **MLX: yes** — it targets the M5 Neural Accelerators via **Metal 4 TensorOps**, with
  ~3.3–4.1× prefill speedups M5-vs-M4. A transformer **encoder forward pass — i.e. embedding —
  is exactly that compute-bound, matmul-heavy regime.**
- **PyTorch MPS: not confirmed** to drive the tensor units (as of early 2026). You get the
  general GPU + memory bandwidth, but likely not the dedicated matmul units.

So on an M5, **MLX is the only path that exploits the marquee hardware** — but for **bge-m3
specifically, MLX is dense-only** (no learned sparse), which is why a hybrid build can't be
"all MLX."

### Practical recipe for this project (bge-m3 dense + sparse, M-series host)

- **Must run on the host, not a devcontainer** — Docker on macOS has **no Metal/MPS
  passthrough**, so GPU acceleration is host-only. (A *host* conda env keeps package isolation.)
- **Default: [Infinity](https://github.com/michaelfeil/infinity) on `--device mps` with
  fp16.** One server returns dense + learned-sparse + ColBERT; fp16 boosts throughput with
  negligible retrieval impact; push batch size up (UMA has headroom); try
  `PYTORCH_MPS_PREFER_METAL=1`.
- **Max-performance option:** split — **MLX-fp16 for dense** (hits the M5 tensor units) +
  Infinity/CPU for sparse. More plumbing; only worth it if a quick benchmark shows MLX is
  dramatically faster. Quality is identical either way, so benchmark and decide empirically.

> **Interview soundbite:** *"On Apple Silicon, embedding-backend choice is about speed, not
> quality — same weights, same vectors. Both PyTorch-MPS and MLX use the unified memory
> architecture, but MLX is built for it (lazy graph, per-op device, zero-copy) and is the
> only stack that drives the M5's per-core Neural Accelerators via Metal 4 TensorOps. The
> trade-off is that for bge-m3, MLX only does dense, so a hybrid index that needs learned
> sparse runs on a PyTorch-MPS server like Infinity instead."*

---

## Hybrid retrieval, fusion, and reranking (how this project queries)

### Dense vs learned-sparse vectors
A **dense** vector (bge-m3, 1024-d, cosine) captures *meaning* — it matches paraphrases
and acronym/expansion synonymy, but can miss exact rare tokens. A **learned-sparse**
vector is a mostly-zero vector over the model's vocabulary: only the tokens the model
judges important get non-zero weight. It is like BM25, but the weights — and *which*
terms light up, including some not literally present ("term expansion") — are learned.

**Why query and stored sparse must share a vocabulary.** Sparse weights are keyed by a
specific tokenizer's token-IDs. bge-m3 and SPLADE (LlamaIndex's default `fastembed`
sparse) are different models with different vocabularies, so their sparse vectors are not
comparable — dot-producting them is meaningless. Our collections store **bge-m3** sparse
weights, so the query's sparse side must also be bge-m3. We supply a custom
`sparse_query_fn` (see `src/query/bge_m3_embedding.py`) that encodes the query with bge-m3
and reshapes its `lexical_weights` (`{token_id: weight}`) into Qdrant's `(indices, values)`
form — bypassing the default SPLADE encoder so query and stored vectors share one vocabulary.

### Hybrid + Reciprocal Rank Fusion (RRF)
We retrieve a dense leg and a sparse leg, then fuse with **RRF**: a document's score is
`sum 1/(k + rank)` over the lists it appears in (k=60). RRF fuses *ranks*, not scores, so
it needs no score normalization and is robust across modalities. Vector-DB research showed
server-side RRF is not universal (Qdrant/Milvus/ES yes; Pinecone uses alpha; Weaviate its
own fusion), so client-side RRF is the portable default — which is what LlamaIndex does.

### Bi-encoder vs cross-encoder (why rerank)
bge-m3 is a **bi-encoder**: query and document are embedded *separately*, then compared —
fast and precomputable, but lossy (the model never sees them together). A **cross-encoder**
reranker (bge-reranker-v2-m3) feeds *(query, document) together* through a transformer and
emits one relevance score — far more accurate, but not precomputable, so it runs only on the
top-K candidates: **retrieve wide, rerank narrow**. It is the standard fix for the topic
drift contextual-embedding (C′) introduces, because it re-reads the real query against each
candidate and demotes off-topic results the diluted embedding floated up.

### Framework vs. our manual approach (and limitations)
We use LlamaIndex's native components rather than rolling our own: `QdrantVectorStore`
(hybrid mode) is the backend + dense/sparse retrieval, `hybrid_fusion_fn` is the fusion
swap point, and a node postprocessor (`FlagEmbeddingReranker`) is the rerank stage. This
replaces the earlier hand-built path (qdrant-client direct upsert + manual RRF in an
ephemeral script). Where the framework is strictly better: maintained, far less code, clean
swap points. Where it constrains us / limitations: (1) in hybrid mode the dense leg must go
through the index's `embed_model` (we wrap our FlagEmbedding bge-m3 in a `BaseEmbedding` for
parity — `HuggingFaceEmbedding("BAAI/bge-m3")` can drift); (2) fusion is client-side (a
server-side Qdrant-native RRF fast-path is a tracked enhancement); (3) bge-m3 sparse needs a
custom `sparse_query_fn` because the default sparse encoder is SPLADE; (4) the shipped Qdrant
fusion is relative-score only, so RRF is supplied as a small callback.

---

## Thread-aware retrieval (small→big expansion)

> This is the **retrieval** sense of "thread-aware" — match a unit, return its whole thread.
> Do not confuse it with the **summary** sense (per-email summaries written with preceding-thread
> context), which is a build-time step covered in [`EXPERIMENTS.md` §13](EXPERIMENTS.md). See the
> [terminology box](EXPERIMENTS.md#terminology-read-this-first).

### What it does

After the normal hybrid retrieve + optional rerank step, the result set is **expanded into
full email threads**. Each retrieved node carries a `thread_id` in its metadata; the expander
fetches *all* emails in that thread from Qdrant (by scrolling the collection filtered on
`thread_id`), reconstructs multi-chunk emails by rejoining their body chunks (best-effort
order — there is no `chunk_index` field yet), sorts the emails chronologically, and renders
each one as an attributed block — a single header line followed by the body:

```
[Thread: <subject>]

[2015-01-08 16:05] From: <sender>  To: <recipient(s)>  Cc: <cc or —>
  <body>
```

All emails from the same thread are concatenated into a single `ThreadContext` object. The
final result is a list of `ThreadContext` values — one per unique thread — rather than a flat
list of retrieved chunks.

### Why it matters

Two concrete problems motivate this:

1. **Terse replies embed poorly in isolation.** A one-line reply ("sounds good") has almost
   no semantic signal on its own. Embedding it independently gives the dense retriever nothing
   to latch onto, and the sparse side only sees a handful of common tokens. The surrounding
   thread — with the substantive question it was answering — carries the real meaning. By
   matching on *any* email in the thread (typically the substantive ones) and then pulling
   the *whole* thread, terse replies become reachable without embedding tricks.

2. **Explicit attribution for the LLM.** Threads rendered with a per-email From/To/Cc/Date
   header let the language model see *who said what and when*. Without this, summarisation
   tasks can conflate or mis-attribute statements across participants — the model sees a wall
   of text with no speaker boundaries.

### How to call it

`HybridSearcher` exposes two methods:

```python
# Plain retrieval — returns a list of LlamaIndex NodeWithScore objects (unchanged)
nodes = searcher.search(query)

# Thread-aware retrieval — returns a list of ThreadContext objects
contexts = searcher.search_threads(query)
```

Each `ThreadContext` has:
- `.thread_id` — the thread identifier (matches `thread_id` in Qdrant payload)
- `.subject` — the thread subject (from the first email's payload)
- `.emails` — list of `ThreadEmail` objects (one per reconstructed email)
- `.text` — the fully rendered, attributed block ready to pass to an LLM

`build_hybrid_searcher` accepts `mode="hybrid"` (default) or `"dense"`, and `rerank=True`
to attach the cross-encoder before thread expansion:

```python
searcher = build_hybrid_searcher("my-collection", mode="hybrid", rerank=True)
contexts = searcher.search_threads("when did we agree on the deadline")
```

See `scripts/probe_threads.py` for a ready-to-run manual probe against a live collection.

### Token bounding (off by default)

Very long threads can overflow the context window of a small LLM. The `bound_thread`
function (in `src/query/thread_expand.py`) accepts a `max_tokens` limit and a pluggable
`summarizer` callable. When the rendered thread exceeds the limit, it keeps the thread's
root (first) and most-recent (last) email verbatim and replaces the middle — either by
running the summarizer over it, or, if no summarizer is supplied, by eliding it with a
`[N earlier emails omitted]` marker. This is off by default — `assemble_threads` returns
unbounded threads — and is intended for environments with tight context budgets or
unusually large threads.

---

## Roadmap / coming next

Covered above and **measured** (see `EXPERIMENTS.md` §6–9): hybrid dense+sparse + RRF,
learned-sparse (bge-m3), the cross-encoder reranker, contextual retrieval (C′), and
thread-aware retrieval.

- ✅ **Evaluation — DONE (`EXPERIMENTS.md` §9).** A 45-query labeled eval across 3 lenses
  (ranked metrics, answer-coverage, end-to-end answer quality) and 2 answer models, with an
  LLM judge calibrated against a stronger reference. Verdict: **keep `C′`; recommended stack
  `C′` + expand top ~1–3 threads, reranker off; a small model suffices; retrieval coverage is
  the ceiling.**
- **▶ Next lead — lift the retrieval ceiling (~76%).** The eval showed the answer model isn't
  the bottleneck; retrieval is. Open issues: per-thread summaries (#11), coverage diagnosis
  (#12), summary-prompt quality (#13), chunk size (#14).
- **Chunking strategy, RAPTOR, late chunking** — deeper theory sections still to write.
