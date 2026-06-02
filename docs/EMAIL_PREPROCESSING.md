# Email Preprocessing: Reply Chain Stripping

*[← docs index](INDEX.md) · [README](../README.md) · the measured cleanup economics are in [`EXPERIMENTS.md` §1–§5](EXPERIMENTS.md)*

## Why this matters

Work emails stored in reply threads contain the full conversation history
re-quoted in every message. Without stripping, a single `.eml` file for a
20-reply thread might be 10,000+ tokens — almost all of it repeated content
from earlier messages.

This has two concrete effects on RAG quality:

1. **Wasted vector store space.** Nearly identical chunks are indexed for
   every reply in the thread.
2. **Poor retrieval.** A chunk containing one relevant sentence buried in
   5,000 tokens of old quoted text will score lower than a focused chunk.

On the sample corporate inbox used during development (31,981 emails),
stripping achieved a **68% total token reduction** — median body length
dropped from 501 → 290 tokens and 67% of emails now fit inside a single
512-token chunk without being split.

---

## Where stripping happens

`src/data/loaders/mail_archive_x.py` — `MailArchiveXLoader._strip_reply_chain()`

Called automatically during `_parse_eml_file()`, after the body is extracted.
HTML-only emails (no plain-text part) are first converted to plain text by
`_HTMLTextExtractor` so that separator patterns become visible.

Separator patterns detected:

| Pattern | Example | Source |
|---|---|---|
| `>` quoted lines | `> Original text` | RFC standard |
| `-----Original Message-----` | `-----Original Message-----` | Outlook |
| `-----Forwarded message-----` | `-----Forwarded message` | Outlook |
| `________________________` | 10+ underscores | Outlook |
| 10+ dash/star/underscore line | `------------------------------` | Various |
| 3+ dashes only line | `---------` | Globex |
| Line containing "Original Message" | `Original Message ---------` | Globex |
| `On <date> ... wrote:` | `On Mon, Jan 1, 2024, Alice wrote:` | Gmail / Apple Mail |
| Outlook inline headers | `From: Alice\nDate: ...\nTo: ...\nSubject: ...` | Outlook |
| Korean Outlook headers | `보낸 사람: ...\n날짜: ...\n받는 사람: ...\n주제: ...` | Korean Outlook |
| Japanese Outlook headers | `差出人: ...\n日付: ...\n宛先: ...\n件名: ...` | Japanese Outlook |
| Chinese Simplified headers | `发件人: ...\n日期: ...\n收件人: ...\n主题: ...` | Chinese Simplified Outlook |
| Chinese Traditional headers | `寄件人: ...\n日期: ...\n收件者: ...\n主旨: ...` | Chinese Traditional Outlook |
| Thai Outlook headers | `จาก: ...\nวันที่: ...\nถึง: ...\nหัวเรื่อง: ...` | Thai Outlook |
| Vietnamese Outlook headers | `Từ: ...\nNgày: ...\nTới: ...\nChủ đề: ...` | Vietnamese Outlook |
| Indonesian/Malay Outlook headers | `Dari: ...\nTanggal: ...\nKepada: ...\nPerihal: ...` | Indonesian/Malay Outlook |

### Critical safety guard: forwarded-only emails

Some emails have **no new content above the separator** — the entire body IS
a forwarded or quoted message:

```
---------- Forwarded message ----------
From: alice@example.com
Subject: Q3 Report
...
```

Without protection, stripping would delete the entire body. The
`has_real_content` guard in `_strip_reply_chain` prevents this: separator
patterns are only honoured once at least one non-blank line of real content
has been accumulated above them. Emails with no new top-level content are
preserved in full.

> **Approx. 4% of emails** in the sample corpus were affected by this
> before the guard was added — their bodies were being silently wiped.

---

### Reply posting styles supported

| Style | Description | Example source | Supported |
|---|---|---|---|
| **Top-posting** | New reply above quoted history | Outlook, corporate email | ✅ |
| **Bottom-posting** | New reply below `>` quoted block | Technical mailing lists, open-source | ✅ |
| **Inline / interleaved** | Reply text woven between quoted paragraphs | IETF, Linux kernel list | ❌ see below |

---

### Known gap: inline/interleaved posting

Inline posting places the reply text between individual quoted paragraphs:

```
> Alice asked: what's the timeline?

Q4, targeting end of November.

> Alice asked: who owns the review?

Bob owns it, with Alice as backup.
```

This pattern is **not currently handled**.  The stripper treats the first `>`
line as the boundary and keeps only what precedes it — for inline posts at the
top that means the full body is preserved (because `has_real_content` stays
`False`), but for mixed inline+bottom posts some reply text may be lost.

**Impact:** Inline posting is rare in corporate Outlook environments and
uncommon in general work email.  It is most prevalent in:
- IETF mailing list archives
- Linux kernel and open-source project lists
- Old-school Unix sysadmin communities

**To fix when needed:** Replace the single-boundary truncation logic in
`_strip_reply_chain` with a line-by-line classifier that identifies and
collects only non-`>`-prefixed lines throughout the full body.  Each new
pattern needs corresponding test cases in `tests/test_reply_chain_stripping.py`
and a re-run of `debug_strip_reply_chain.py` on real data.

---

## Automated validation

The debug script includes an **invariant checker** that runs automatically
whenever you call it. It checks every email where stripping fired and flags
two classes of problems:

| Check | What it means if it fires |
|---|---|
| **FIRST LINE CHANGED** | Stripping removed content from the start of the body — a definitive false positive bug |
| **NEAR-EMPTY OUTPUT** | < 5 tokens left from a > 100-token email — likely a genuine terse reply ("FYI.", "Thanks") but worth spot-checking |

Run it after any change to `_strip_reply_chain` or `_REPLY_SEPARATOR_RE`:

```bash
python scripts/debug_strip_reply_chain.py --sample 3000 --show 0
```

**Expected healthy output:**
```
  Invariant check PASSED — all 1415 stripped emails preserved their first
  line and produced non-empty output.
```

Any **FIRST LINE CHANGED** failure is a bug. **NEAR-EMPTY OUTPUT** cases
should be spot-checked manually; on the sample corpus they are all
genuine one-liner replies.

---

## Adapting to a new email source

When you point this project at a new mailbox or organisation, run the two
analysis scripts below before indexing. They take 2–5 minutes and tell you
whether the stripping is working and what chunk size to use.

### Step 1 — Profile email lengths

```bash
python scripts/analyze_email_lengths.py --sample 2000
```

This downloads a random sample of emails from your blob store (using the
saved checkpoint selection if one exists), strips reply chains, and prints
a token-length distribution.

**How to read the output:**

```
  Median (p50) :     80 tokens   ← typical short reply after stripping
  p90          :    422 tokens   ← 90% of emails are under this
  Fit in 1 chunk:   93%          ← at the simulated chunk size
  Suggested chunk_size: 448      ← set RAG_CHUNK_SIZE to this value
```

- If **p90 < 512** → set `RAG_CHUNK_SIZE=512` and you're done.
- If **p90 > 1000** → stripping is likely not catching all patterns; proceed
  to Step 2 before choosing a chunk size.
- If **mean >> median** (e.g., mean is 5× the median) → long-tail outliers
  exist; Step 2 will identify them.

### Step 2 — Debug stripping on the worst offenders

```bash
python scripts/debug_strip_reply_chain.py --sample 3000 --show 5
```

This downloads a sample, runs each body through `_strip_reply_chain` twice
(before and after), and prints:

- **Invariant validation** — flags any emails where stripping misbehaved
  (see [Automated validation](#automated-validation) above)
- Overall reduction stats (% of emails changed, total tokens removed)
- The N longest raw emails with:
  - Before/after token counts
  - The first line that triggered stripping, or **"none found"** if stripping
    had no effect on that email

**What to look for in "none found" emails:**

The raw body preview will show you exactly what separator format your email
client uses. Common patterns not yet in the code:

```
# Lotus Notes
--- <name> wrote: ---

# Chinese Outlook variant
发件人: Name
日期: Date

# Custom corporate footers acting as dividers
== Confidential =======================
```

### Step 3 — Add missing patterns

Open `src/data/loaders/mail_archive_x.py` and extend `_REPLY_SEPARATOR_RE`
or `_strip_reply_chain()`:

```python
# _REPLY_SEPARATOR_RE — for hard single-line separators:
_REPLY_SEPARATOR_RE = re.compile(
    r'^('
    r'...'
    r'|---\s+\w+\s+wrote:\s*---'   # Lotus Notes
    r')',
    re.IGNORECASE,
)

# _strip_reply_chain — for multi-line attribution patterns:
# Add an elif block analogous to the _ON_WROTE_RE check.
```

### Step 4 — Add test cases

Add corresponding test cases to `tests/test_reply_chain_stripping.py` before
shipping. The test file has **45 tests** across six classes:

| Class | What it covers |
|---|---|
| `TestStripReplyChainNoReply` | Bodies with no reply — must be returned unchanged |
| `TestStripReplyChainQuotedLines` | `>` quoted line stripping |
| `TestStripReplyChainOutlookSeparators` | `-----Original Message-----`, `___` etc. |
| `TestStripReplyChainGlobexSeparator` | Globex `---------` pattern |
| `TestStripReplyChainOnWrote` | `On <date> ... wrote:` (single and multi-line) |
| `TestStripReplyChainOutlookInlineHeader` | `From: / Date: / To: / Subject:` block |
| `TestHTMLTextExtractor` | HTML-to-text conversion |
| `TestStripReplyChainInvariants` | Correctness invariants (see below) |

Each new pattern needs at minimum:
- A test that strips correctly when the pattern is present
- A test that does NOT strip when the pattern appears mid-sentence or at the
  start of the body (false-positive guard)
- An invariant test confirming the first line is preserved

```python
def test_lotus_notes_separator(self):
    body = "My reply.\n\n--- Alice wrote: ---\n> Old content"
    result = MailArchiveXLoader._strip_reply_chain(body)
    self.assertEqual(result, "My reply.")

def test_lotus_notes_separator_first_line_not_stripped(self):
    # Separator on line 1 — no real content above it, must preserve in full.
    body = "--- Alice wrote: ---\n> Old content"
    result = MailArchiveXLoader._strip_reply_chain(body)
    self.assertEqual(result, body)
```

#### Invariant tests

`TestStripReplyChainInvariants` encodes the two rules that must hold for
**every** stripping change:

1. **First line preserved** — `stripped.first_line == raw.first_line`
2. **No length increase** — `len(stripped) <= len(raw)`
3. **Forwarded-only emails preserved** — body starting with a separator is
   returned unchanged
4. **Stops at first boundary** — content between two separators is not kept

Run the full suite:

```bash
pytest tests/test_reply_chain_stripping.py -v
```

### Step 5 — Re-run analysis to confirm

Re-run Step 1. You should see:
- `Emails where stripping changed body` percentage increase
- `Total tokens removed` percentage increase
- Median and p90 drop

Iterate Steps 2–5 until the debug script shows no large "none found" emails
and the analysis shows p90 comfortably below your target chunk size.

---

## Choosing the final chunk size

Once stripping is working well:

| Scenario | Recommendation |
|---|---|
| p90 < 300 tokens | `RAG_CHUNK_SIZE=384`, `RAG_CHUNK_OVERLAP=50` |
| p90 300–600 tokens | `RAG_CHUNK_SIZE=512`, `RAG_CHUNK_OVERLAP=64` ← **project default** |
| p90 600–900 tokens | `RAG_CHUNK_SIZE=1024`, `RAG_CHUNK_OVERLAP=100` |
| p90 > 900 tokens | Investigate remaining long emails — likely a new pattern |

The project ships with `RAG_CHUNK_SIZE=512` and `RAG_CHUNK_OVERLAP=64` as
defaults (chosen from the sample corpus analysis). Override in `.env` if
your email corpus has a different distribution:

```bash
RAG_CHUNK_SIZE=512    # set to your p90 value, rounded to nearest 64
RAG_CHUNK_OVERLAP=64  # overlap matters only for emails that exceed chunk size
```

**Why overlap=64?** Overlap only affects the ~33% of emails long enough to be
split across multiple chunks. 64 tokens gives a sentence of context at each
boundary without meaningfully inflating index size.

**Why not just use the maximum email length?** Retrieval works by scoring
chunks against a query. A 3,000-token chunk containing one relevant sentence
scores lower than a 400-token chunk containing that same sentence. Smaller
focused chunks = better retrieval precision.

---

## Re-indexing after changes

Stripping changes the *content* that gets embedded. If you change stripping
patterns or chunk size, you must wipe the existing index and re-index:

```bash
# Wipe Qdrant collection
python scripts/reset_qdrant_index.py

# Re-index from scratch
python scripts/batch_index_to_vector_store.py
```
