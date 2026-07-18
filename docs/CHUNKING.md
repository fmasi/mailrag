# How mailrag chunks & embeds emails (and attachments)

When mailrag indexes one email, it produces **two different kinds of chunk**, embedded
differently, but stitched back together at query time by a shared `thread_id`.

## The short version

- **Body chunk** = email text **with its LLM contextual summary glued on the front**, then
  embedded. The summary is *baked into the vector* — so a terse reply ("Approved") still
  matches its topic.
- **Attachment chunk** = a spreadsheet/PDF/deck split by its **own structure** (spreadsheet
  row-groups with the header repeated, PDF pages, slides), embedded as **just the raw
  content + exact-number tokens**. **No summary is glued on** — the vector is "pure"
  attachment content, which is *better* for finding a fact buried inside it (a summary would
  only add noise).

> **Does an attachment chunk carry the thread/email summary?** **No** — not in its embedding.
> Only body chunks do. But every attachment chunk is stamped with the `thread_id`, so a hit
> on it still pulls back the whole thread (body + summary + attachment) via thread expansion.
> - *What it's matched on (the vector):* attachment text only.
> - *What you get back:* the whole thread.

## Diagram

```
                     ┌──────────────  ONE EMAIL  (part of a thread)  ──────────────┐
                     │                                                             │
    ┌────────────────┴─────────────────┐              ┌────────────────────────────┴───────────┐
    │            BODY                   │              │       ATTACHMENT  (Q3 targets.xlsx)      │
    │  "Team, here are the targets…"    │              │   pandas reads it → row-groups           │
    │                                   │              │   (header row repeated in each chunk)    │
    │  ➊ PREPEND the LLM summary  ◄──────┼── cached     │   ➊ NO summary added                     │
    │     summary + body text           │   summaries  │   ➋ + numeric tokens ($210,000,000       │
    │  ➋ split into ~512-tok chunks     │              │        → 210000000)                       │
    │                                   │              │   ➌ split by structure (already ≤budget) │
    └───────────────┬───────────────────┘              └───────────────────┬──────────────────────┘
                    │  bge-m3 embed                                          │  bge-m3 embed
                    ▼  (summary IS in the vector)                            ▼  (summary NOT in the vector)
    ┌───────────────────────────────────┐              ┌───────────────────────────────────────────┐
    │  Qdrant points  content_kind=body  │              │  Qdrant points  content_kind=attachment     │
    │  payload: {thread_id, summary,     │              │  payload: {thread_id, attachment_name,      │
    │            subject, text…}         │              │            parent_message_id, chunk_index…} │
    └───────────────┬───────────────────┘              └───────────────────┬──────────────────────┘
                    └──────────────┬───────────────────────────────────────┘
                                   │   both tagged with the SAME thread_id
                                   ▼
      ═══════════════════════  AT QUERY TIME  ═══════════════════════
      query → bge-m3 embed (+ numeric tokens) → hybrid (dense+sparse, RRF)
            → matches a chunk (body OR attachment)
            → thread_id links it → RETURN THE WHOLE THREAD
              (body + summary + attachment text, stitched together)
```

## Step by step

1. **Load & (optionally) drop noise.** Emails are loaded; with `--embed-summary`, the
   calibrated Pass-2 verdicts drop confident noise and attach each kept email's cached
   contextual **summary** to its metadata.
2. **Body → Documents.** Each email becomes a LlamaIndex `Document` carrying
   `metadata["summary"]`.
3. **Attachments → Documents.** `build_attachment_documents` parses each attachment by MIME
   type with a **format-aware splitter** (`src/indexing/attachment_chunking.py`): pandas
   row-groups for `.xlsx`/`.csv` (header repeated per chunk), per-page for PDF, per-slide for
   PPTX, per-section for DOCX — each already sized under the token budget. These Documents
   carry `content_kind="attachment"`, `attachment_name`, `parent_message_id`, `thread_id`,
   `chunk_index` — but **no `summary`**.
4. **Split.** All Documents go through the `SentenceSplitter` (`chunk_size≈512`,
   `overlap 64`). Body Documents split into ~512-token chunks; attachment Documents are
   already ≤budget so they pass through intact (never mixing a 4-line body with a 500-row
   sheet).
5. **Dedup** exact-duplicate chunk contents.
6. **Embed.** For each chunk, just before embedding
   (`src/indexing/contextual_index.py`):
   - `prepend_summary(text, metadata.get("summary"))` — prepends the summary **only if the
     chunk has one** (body chunks do; attachment chunks don't, so they're unchanged).
   - `augment_numeric(text)` — appends canonical integer tokens (`$210,000,000 → 210000000`)
     so exact-figure queries can hit. This runs at **both** index and query time so both
     sides share the token vocabulary.
   - `bge-m3` produces a **dense** vector + a **learned-sparse** weight map.
7. **Upsert** to Qdrant as a hybrid point: dense vector + sparse vector + a `payload` that
   keeps the **untouched surface `text`** (the summary/numeric tokens are in the *vector*,
   not shown back to you) plus all the metadata.

## Why it's designed this way

- The attachment vector staying **summary-free** is deliberate: it's matched purely on what's
  *inside* the attachment, so a number/fact isn't diluted by a summary about the covering
  email.
- The `thread_id` is the glue: you can hit the corpus via a body chunk **or** an attachment
  chunk, and either way `search_threads` expands the hit to the **whole thread**, so the
  summary/context rides along at *retrieval* time (not *embedding* time).
- Consequence (relevant to retrieval tuning): a **vague** query leans on the summary-rich
  *body* chunks to find the thread (then the attachment content is right there in the returned
  thread); a **pointed/numeric** query can hit the *attachment* chunk directly. This is why
  the retrieval eval segments pointed vs vague queries.

See also: `docs/MCP_SERVER.md` (how to query), and the retrieval-improvement plan in the
issue tracker.
