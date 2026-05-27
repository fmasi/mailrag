# RAG Study Guide

A working reference for the retrieval techniques behind this project — written to be
**interview-ready**: crisp definitions, the *why*, the trade-offs, and concrete examples
you can talk through out loud. Grown incrementally as we build the system.

> Running example used throughout: searching this corpus for **"ACP"** (the
> *Acme Certified Partner* program). It's a great teaching case because it
> mixes a semantic concept ("readiness / certification program") with rare exact
> tokens ("ACP", ticket IDs like `00451823`) — which is exactly where the
> different retrieval methods diverge.

---

## Contents

1. [Retrieval fundamentals: dense vs sparse](#1-retrieval-fundamentals)
2. [BM25 — classic lexical search](#2-bm25--classic-lexical-search)
3. [Learned sparse retrieval (SPLADE, bge-m3 sparse)](#3-learned-sparse-retrieval)
4. [bge-m3 — the M3 capabilities](#4-bge-m3--the-m3-capabilities)
5. [Hybrid search & Reciprocal Rank Fusion (RRF)](#5-hybrid-search--rrf)
6. [ColBERT & late interaction](#6-colbert--late-interaction)
7. [Serving embeddings on Apple Silicon (UMA, MPS vs MLX, M5)](#7-serving-embeddings-on-apple-silicon)
8. _(coming next)_ Chunking, parent-document retrieval, reranking, RAPTOR, contextual retrieval, late chunking, thread reconstruction, evaluation

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
- ❌ **Weak on rare exact tokens.** Product codes, ticket IDs (`00451823`), acronyms
  (`ACP`), surnames — the model captures the *gist*, not the literal string, so a precise
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
   *"the"* is in every email → near-zero weight; *"ACP"* is rare → high weight.
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
- **Sparse (lexical)** → exact-match precision (the ACP/ticket-ID case).
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

Query: *"ACP test cases"*. Suppose:

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
latch onto."* The token "ACP" in the query can lock onto the contextual "ACP" token in a
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

_Next sections to add: chunking strategy, parent-document / thread-as-parent retrieval,
cross-encoder reranking, RAPTOR, contextual retrieval vs late chunking, email thread
reconstruction (Message-ID/In-Reply-To/References, JWZ), and evaluation metrics
(recall@k, nDCG, MRR)._
