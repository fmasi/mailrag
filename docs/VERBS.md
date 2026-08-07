# Verbs & Personas

`mailrag` cleans and indexes an email corpus through a small set of **verbs**. They
sit on **two axes**, and — this is the key thing — **only `index` ever deletes
anything.** Everything before it tags, measures, judges, or looks; the raw `.eml`
files on disk make every drop reversible.

- **Scope** — *which* mail is even in play (`scope`).
- **Noise & context** — of the in-scope mail, what's junk and what context to attach.
  This is a **cost-ordered funnel**.

```
 SCOPE                  NOISE & CONTEXT  (cost-ordered; only `index` drops)
 ─────                  ───────────────────────────────────────────────────
 scope → measure → tag  →  scan  →  judge  → calibrate → summarize → prune → index
  pick    chunk   regex    no-LLM   cheap     sample      LLM         apply   embed
  folders  size   flags    clusters LLM       gate        context     drops   + store
  ─────────────   ─────    ──────   ─────     ──────      ───────     ─────   ─────
   free    free   free     cheap    small     small       BIG         free    GPU
```

The one cost that dominates is the **LLM pass over each email body**. So the golden
rule is: **never send the same email to the LLM twice.** Each email gets *at most one*
LLM call — a cheap `judge` (for mail you'll drop) **or** a `summarize` (for mail you
keep), never both.

## The verbs

| Verb | What it does | Cost | Drops? |
|------|--------------|------|--------|
| `scope` | Choose which folders/accounts are in play. | free | no |
| `measure` | Measure the corpus, suggest a chunk size. | free | no |
| `tag` | Regex/header rules flag obvious bulk (newsletters, `no-reply@`). | free | no (tags) |
| `scan` | Cluster the embeddings (no LLM) and rank "noise pockets". Also **recommends a persona** from your corpus's size and obvious-noise share. | cheap (GPU) | no |
| `judge` | Cheap LLM **verdict only** (noise/keep). Run on mail you intend to *drop*, so you never pay to summarize it. | small LLM | no (verdict) |
| `calibrate` | Judge a small **sample** with the rubric and bucket the suspected mistakes — the gate before any full LLM run. | small LLM | no |
| `summarize` | LLM **summary/context** for the emails you keep (the noise verdict comes free in the same call). The expensive step. | big LLM | no (verdict) |
| `prune` | Apply the `tag`/`scan`/`judge` drops **before** the LLM pass, so the LLM only ever runs on the keep set. | free | marks drops |
| `index` | Embed (BGE-M3 dense + sparse) and store in Qdrant. **The only step that actually removes the dropped mail.** Incremental by default; `--recreate` rebuilds from scratch. | GPU | yes |
| `ask` | Answer a question against an indexed collection (hybrid retrieval → thread expansion → grounded answer). | LLM at query time | — |
| `onboard` | One-shot, zero-config build from an `.eml` directory. | — | — |
| `run` | Execute a persona recipe end-to-end (headless). Needs `--persona`, and `--model` for any LLM step. | per recipe | via `index` |
| `wizard` | Full-screen guided pipeline (persona → scope → review → live run). `--classic` for the old prompt flow. | per recipe | via `index` |
| `sync` | Fetch new mail from a live account and index the delta, so a collection stays fresh instead of frozen at its export date. See [`SYNC.md`](SYNC.md). | small LLM (delta only) | no |
| `mcp` | Run the stdio MCP server over an indexed collection, so any agent can query it. See [`MCP_SERVER.md`](MCP_SERVER.md). | — | no |
| `attachments` | `build` / `list` / `get` an email's attachments (extract + OCR text). See [`SETUP.md § 9`](SETUP.md#9-attachment-extraction). | free · OCR LLM | no |

Every verb is invoked through the repo-root shim — `./mailrag <verb>` (there is no
installed console command; Poetry stays `package-mode = false`).

### `index` is incremental

Re-running `index` over a corpus that has grown does **not** duplicate what is
already there, and does not require rebuilding the collection:

```bash
./mailrag index --profile personal.json --embed-summary   # incremental (default)
./mailrag index --profile personal.json --recreate        # full rebuild
```

Each email carries a stable `message_key` — its normalized `Message-ID`, or a
content hash when it has none — and every chunk's Qdrant point id is derived from
that key plus the chunk's position *within its own email*. So the same email always
lands on the same point ids no matter what else is in the run, and an incremental
pass deletes an email's existing points before upserting its new ones. Re-indexing
unchanged mail is a no-op; re-indexing changed mail leaves no stale chunks behind;
an email's attachment chunks share its key and are replaced with it.

Combined with the content-hash Pass-2 cache, that makes "index the 40 emails that
arrived since Tuesday" cheap: no LLM calls for mail already judged, and no full
re-embed. This is the foundation continuous sync builds on
([#101](https://github.com/fmasi/mailrag/issues/101)).

> **Policy guard.** Every point also records the rules it was produced under —
> preprocessing version, chunk policy, chunk size/overlap, `--embed-summary`,
> embedder. An incremental run refuses if the collection was built under
> different ones, rather than silently mixing two incomparable populations of
> vectors in the same collection. See
> [`EMAIL_PREPROCESSING.md § Re-indexing after changes`](EMAIL_PREPROCESSING.md#re-indexing-after-changes).

> **One-time rebuild.** Collections built before deterministic ids carry random
> point ids and no `message_key`, so appending to them would *duplicate* every
> chunk rather than replace it. `index` detects this and refuses, pointing you at
> a single `--recreate` run — which costs no LLM calls, since every judgment is
> already cached. `--allow-legacy-append` overrides the check if you really want it.

> Older verb names still work for one release as hidden aliases:
> `select→scope`, `profile→measure`, `pass1→tag`, `explore→scan`, `pass2→summarize`,
> `build→index`, `query→ask`.

## Personas

You rarely run verbs one by one. A **persona** is a named recipe — *which verbs, in
what order* — for a given budget/quality tradeoff. Run `scan` first and it will
recommend one.

| Persona | What you get | Recipe |
|---------|--------------|--------|
| **`llm-none`** | Fast, **no LLM at all**. Body-only embeddings; `prune` drops the regex-tagged bulk. Cheapest, noisiest. | `scope → measure → [scan] → tag → prune(tag) → index` |
| **`llm-verify`** | **Safe budget.** A cheap `judge` confirms drops only on the `scan`-flagged suspects; `prune` blacklists them *before* the summary pass; summaries on what's kept. Won't drop real mail blind. | `scope → measure → scan → calibrate → judge(suspects) → prune(judge) → summarize(rest) → index` |
| **`llm-all`** | **Max quality.** One combined LLM call per email (summary + verdict); `prune` then drops the confident noise. No email is dropped unseen. | `scope → measure → calibrate → summarize(all) → prune(summarize) → index` |

Run a persona end-to-end with `./mailrag run --persona <name> --profile <p> [--model <m>]`,
or interactively with `./mailrag wizard`.

> **An honest note on the budget.** Each LLM call is dominated by reading the email
> *body*, so `judge` is not dramatically cheaper *per call* than `summarize`.
> `llm-verify`'s saving over `llm-all` is mainly the **summary output it avoids on the
> bulk it drops** (and not summarizing obvious junk), plus the safety of LLM-confirmed
> drops — not a big reduction in body processing. If your corpus has little obvious
> noise, `llm-all` is simpler and barely more expensive; `scan` tells you which case
> you're in.

> **Why does `llm-all` still `calibrate` if it summarizes every email anyway?**
> `calibrate` doesn't change what `summarize` does — it's a cheap (~200-email) dry-run
> that checks the *rubric* before the full pass. In `llm-all` the noise verdict that
> drives `prune` is a **by-product of each `summarize` call**, so the rubric decides
> what gets dropped across the whole corpus. A mis-tuned rubric mis-drops everywhere,
> and you'd only discover it *after* paying for ~all the calls (the corpus-portability
> case: a wrong rubric flagged ~88% as noise). Calibrate lets a human read the
> false-noise / false-keep buckets and trust the verdicts first. It de-risks the *drop
> decision*, not the summary work — ~200 calls of insurance before tens of thousands of
> spend. `llm-none` skips it because it never runs the LLM (no rubric to trust).

**How to choose:**

- Want speed and don't need summary context? → **`llm-none`**.
- Want great quality but your corpus has a lot of *obvious* noise to skip? →
  **`llm-verify`** (let `scan` find the pockets, `judge` confirm them).
- Want the cleanest possible result and the LLM to look at everything? →
  **`llm-all`**.

`scan` quantifies the choice: it reports how much of your corpus is *obviously*
droppable, which is exactly what decides whether the budget personas are worth it.

## Custom personas

Personas are just data (a recipe in the persona registry, `personas.yaml`), so you can
add your own — a different order, a subset of verbs, or pinned settings — without
touching code. The TUI and `./mailrag run --persona <name>` both read the same registry.
