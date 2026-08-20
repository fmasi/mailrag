# Field report: attachment tooling, and why an agent didn't use it

**From:** JoBot (career-copilot session), 2026-08-20
**Corpus:** `work-rag-ctx-threadaware-v2` (132k points)
**Context:** verifying a candidate's recollection about a Nokia partner event for a job application.
The decisive evidence turned out to be a diagram on slide 5 of a PDF attachment. I found it by
parsing raw `.eml` off disk, not through the MCP tools. Here is what worked, what didn't, and the
more useful half: why I didn't reach for the right tools in the first place.

---

## 1. `list_attachments` and `get_attachment` are FIXED — confirmed

A note in this user's memory dated 19 Aug said both returned empty for real MIME attachments. That is
**no longer true** as of today. Verified:

```
list_attachments(thread_id="HE1PR0702MB367416282BC84D946D1D021AF22FA@HE1PR0702MB3674.eurprd07.prod.outlook.com")
→ 5 rows, including:
  sha256  71e27f95eec4726ba4d2c64317d431e129a24a885c2170d0b0738373fd1805eb
  filename "Template_Nokia Radio World_2025.pptx"
  mime     application/vnd.openxmlformats-officedocument.presentationml.presentation
  size     22,235,496
  inline   false
```

```
get_attachment(sha256="71e27f95...")
→ text_status: "extracted", with the deck's text layer returned correctly.
```

Both good. The `inline` flag is genuinely useful — it cleanly separates signature images from real
documents, and I used it to filter.

## 2. Three real problems remain

### 2a. 🔴 `search_email` always returns `attachment_names: []` — this is the important one

Every hit, every query, including threads that demonstrably carry a 22MB `.pptx` and a 1MB `.pdf`.

```
search_email("Nokia Radio World Windriver demo presentation template slides partner program")
→ 5 hits. All five: "attachment_names": []
   ...including HE1PR0702MB3674..., whose list_attachments call (above) returns the pptx.
```

**Why this matters more than the other two:** search is the discovery surface. An agent forms its
model of the corpus from what search returns. A field that is always empty does not read as "not
populated", it reads as **"this corpus has no attachments."** I never called `list_attachments`
during four separate retrieval passes today, and this is a large part of the reason. The field
actively taught me the wrong thing.

Either populate it or remove it. An always-empty field is worse than an absent one.

### 2b. Some threads with real attachments return `[]` from `list_attachments`

```
list_attachments(thread_id="AM6PR0702MB3672E3E991ABF0E71FF0DFB4F228A@AM6PR0702MB3672.eurprd07.prod.outlook.com")
→ []
```

That thread ("Nokia Radio world demo with Windriver", single message, Tomi Bergholm, 2025-08-11)
carries `Nokia Radio World 2025 Partner v3.pdf`. I extracted it from the raw `.eml` myself
(1,056,439 bytes), so it is definitely there. Possibly a thread-id resolution issue where the
attachment is indexed against a sibling message-id in a larger thread. Reproducible.

### 2c. `get_attachment` returns text only — so diagram-only facts are invisible

This is a design limit rather than a bug, but it is the one that would have cost me the answer.

The fact I needed — that Nokia's partner ecosystem slide showed exactly five names — exists **only as
a diagram**. Five labels arranged around a horseshoe graphic. There is no sentence anywhere in the
corpus that says "five partners". `get_attachment` on that PDF would have returned the surrounding
text and I would have concluded, wrongly, that the claim was uncorroborated.

I got it by extracting the PDF to disk and rendering pages 1-8 as images.

Suggestion: return a `pages` or `render` affordance for PDFs and slide decks, or at minimum flag in
the response when a page's extracted text is thin relative to its size, so an agent knows to go
look at the pixels. `text_status: "extracted"` currently reads as success even when the meaningful
content was pictorial and got dropped.

## 3. Why I used the tools I used

You asked, and this is the part likely worth more than the bug list.

**What I actually did across four retrieval passes:**
- `search_email` (hybrid, default mode) — 8 or 9 times. My default.
- `get_thread` — twice, once I knew a thread was worth reading in full.
- `grep_email` — twice, both times for exact literals: an email address, a domain.
- `list_attachments` / `get_attachment` — **zero times, until prompted.**

**The reasons, honestly:**

1. **A stale memory note.** The user's memory said the attachment tools were broken, dated the
   previous day. I trusted it and never tested. That is my error, not yours, but it is worth knowing
   that a single "this is broken" note propagates and stops the tool being exercised at all. Tools
   that were broken and are fixed need something louder than silence to re-enter an agent's rotation.

2. **`attachment_names: []` confirmed the wrong belief.** See 2a. If it had listed the pptx I would
   have called `list_attachments` on the first pass.

3. **`search_email` is the path of least resistance and it is good.** It answers "what happened" well
   enough that I stopped there repeatedly. It gave me Scott Walker's email to the ELT, the Nokia
   before-and-after, and the two-slot allocation. None of that needed grep or attachments, so nothing
   pushed me to widen. **Good semantic search suppresses tool diversity** — worth designing around.

4. **I only reach for `grep_email` on exact strings.** The description positions it as "the escape
   hatch for exact needle hunts", so that is what I used it for. Notably, when I finally ran
   `grep_email("Radio World")` it was the single highest-value call of the session: it surfaced 25
   messages **and their attachment filenames from the raw MIME headers**, which is how I learned the
   partner PDF existed. Grep is doing attachment discovery better than the attachment tools are.

**The general failure mode:** I had a mental model of the corpus as *email text*, and every tool
response reinforced it. Nothing in the interface surfaced that ~7 documents including a 22MB deck sat
behind the threads I was already reading. An agent optimises against the surface it can see.

## 4. Concrete suggestions, in priority order

1. **Populate `attachment_names` in `search_email` results.** Highest impact by a distance. It turns
   attachments from a thing you must already suspect into a thing you notice.
2. **Fix the thread-id resolution gap in `list_attachments`** (2b). Repro above.
3. **Give `get_attachment` a way to reach pictorial content** for PDFs and decks, or signal when
   extraction is thin so the agent knows to render.
4. **Consider surfacing attachment presence in `get_thread` too** — I read a thread in full and still
   did not learn it had a 22MB deck on it.
5. Minor: `grep_email` took >120s on a 2-term pattern and got backgrounded. Worth a look.

## 5. What the tooling did get right

Once pointed at it, the flow was clean: `list_attachments` gives a stable sha256, `get_attachment`
takes it directly, `inline: true/false` is exactly the filter needed, and mime and size are accurate.
The hybrid search quality is genuinely high — the Scott Walker ELT email, which was the single most
valuable document I found all week, came back as the **first hit** on a loosely-worded natural
language query. That is the hard part, and it works.
