# Roadmap

This is the public roadmap for mailrag. It is derived from the
[open issue tracker](https://github.com/fmasi/mailrag/issues) — every item below
is a real issue, and the tracker is the source of truth if the two ever
disagree. The project has a single maintainer; the cadence at the bottom is
sized accordingly.

**Where things stand.** v0.9.0 is the first tagged release. The core pipeline is
in daily production use: ingest (bodies + attachments, OCR), calibrated LLM
noise-cleaning, bge-m3 dense + learned-sparse hybrid retrieval with thread-aware
reconstruction, continuous IMAP sync, and a seven-tool MCP server. The path to
v1.0.0 is deliberately narrow: close the remaining correctness gaps, make the
headline retrieval claim independently reproducible, and grow the MCP surface
from "a query box" into a proper interface.

## Themes

The open issues cluster into six streams, which the milestones below draw from:

1. **Retrieval quality** — the ongoing precision/recall programme
   ([#91](https://github.com/fmasi/mailrag/issues/91)) and its individual
   experiments: chunking ([#14](https://github.com/fmasi/mailrag/issues/14)),
   retrieval-oriented summaries
   ([#13](https://github.com/fmasi/mailrag/issues/13)), per-thread summaries
   ([#11](https://github.com/fmasi/mailrag/issues/11)), server-side fusion
   ([#1](https://github.com/fmasi/mailrag/issues/1)).
2. **Ingest correctness and performance** — degenerate `text/plain` fallback
   ([#4](https://github.com/fmasi/mailrag/issues/4)), chunk ordering
   ([#5](https://github.com/fmasi/mailrag/issues/5)), parallel full rebuilds
   ([#102](https://github.com/fmasi/mailrag/issues/102)), splitting the
   overloaded loader ([#48](https://github.com/fmasi/mailrag/issues/48)), and
   giving the two source seams one front door
   ([#117](https://github.com/fmasi/mailrag/issues/117)).
3. **MCP surface** — the capability audit and enhancement roadmap
   ([#93](https://github.com/fmasi/mailrag/issues/93)), supported by the unified
   config layer ([#39](https://github.com/fmasi/mailrag/issues/39)).
4. **Evaluation and reproducibility** — a turnkey public benchmark
   ([#97](https://github.com/fmasi/mailrag/issues/97)) and, later, one-command
   reproduction of the full canonical eval
   ([#20](https://github.com/fmasi/mailrag/issues/20)).
5. **Code health and test depth** — the verb rename
   ([#42](https://github.com/fmasi/mailrag/issues/42)), prompt
   dedup ([#46](https://github.com/fmasi/mailrag/issues/46)), clearing the repo
   root ([#43](https://github.com/fmasi/mailrag/issues/43)), small tidy-ups
   ([#51](https://github.com/fmasi/mailrag/issues/51)), and a cold new-user
   run of the whole product
   ([#118](https://github.com/fmasi/mailrag/issues/118)).
6. **Dependency and backend health** — the qdrant-client version cap
   ([#106](https://github.com/fmasi/mailrag/issues/106)) and the periodic
   ecosystem re-check ([#95](https://github.com/fmasi/mailrag/issues/95)).

There is also a research stream around the noise-cleaning pipeline
([#28](https://github.com/fmasi/mailrag/issues/28),
[#30](https://github.com/fmasi/mailrag/issues/30),
[#9](https://github.com/fmasi/mailrag/issues/9),
[#10](https://github.com/fmasi/mailrag/issues/10)) that is explicitly not on
the 1.0 path — the current calibrate-then-sweep pipeline works, and these are
refinements to its cost/quality trade-off.

## The backend decision

mailrag commits to **Qdrant as the vector backend**. It does not maintain a
portable-across-backends posture, and 1.0 will not advertise one.

The reasoning came out of the capability audit in
[#94](https://github.com/fmasi/mailrag/issues/94): the capability that makes
retrieval work here — learned-sparse vectors stored alongside dense ones as
named vectors on the same points — is a genuinely Qdrant-specific seam, not
something every vector store supports. Once that is true, keeping alternative
backends alive buys a portability the project does not want, at the cost of the
effort that should go into using the one backend properly.

Two consequences follow. The dead branches went: `src/storage/persist.py` and
its SimpleVectorStore/Pinecone paths, the `VECTOR_STORE_PROVIDER` switch, and
the Azure Blob loader and cloud batch scripts that were their only callers were
all removed in [#49](https://github.com/fmasi/mailrag/issues/49) — removal
rather than reconciliation, since nothing in the live pipeline reached them.
And the effort redirects forward, starting with server-side fusion through
Qdrant's native Query API ([#1](https://github.com/fmasi/mailrag/issues/1))
instead of the client-side RRF callback used today.

## Milestones

### v0.9.x — patch series

Small, self-contained fixes that do not change behaviour contracts. Shipped as
patch releases as they land; none of them blocks 1.0 individually.

| Issue | What | Why now |
|---|---|---|
| [#4](https://github.com/fmasi/mailrag/issues/4) | Fall through to HTML extraction when the `text/plain` part is degenerate link-soup | Ingest correctness bug; ~1% of emails index as junk |
| [#46](https://github.com/fmasi/mailrag/issues/46) | Extract the shared prompt-formatting helper (`summary.py` ↔ `rubrics.py`) | Removes a comment-enforced lockstep coupling in a known burn area |
| [#106](https://github.com/fmasi/mailrag/issues/106) | Lift the `qdrant-client <1.19` cap once `llama-index-vector-stores-qdrant` fixes its import | Dependency health; the single-minor band blocks 1.19.x security fixes. Lands the release after upstream ships |
| [#51](https://github.com/fmasi/mailrag/issues/51) | Small code-health batch | Folded into adjacent work, not a standalone task |

### v1.0.0 — a release a stranger can trust

The bar for 1.0 is deliberately narrow: **a stranger can install it, see what it
is for, verify the headline retrieval claim themselves, and use the MCP server as
a genuinely capable interface — and the code they read matches the vocabulary the
docs use.** Everything on this list serves that bar; everything that does not is
deferred, however interesting.

"See what it is for" is new, and it is not a rewording. Shipping
[#97](https://github.com/fmasi/mailrag/issues/97) exposed that the two commands a
stranger actually runs both fall short of it from opposite directions: `make demo`
shows the pipeline working but compares it to nothing, and `make bench` measures
the hybrid layer while every lever behind the headline number is switched off. The
project is therefore *demonstrable* and *measurable* but its central claim is
neither shown nor checked. [#123](https://github.com/fmasi/mailrag/issues/123) and
[#125](https://github.com/fmasi/mailrag/issues/125) close those two halves.

| Issue | What | Why it gates 1.0 |
|---|---|---|
| [#93](https://github.com/fmasi/mailrag/issues/93) | MCP capability expansion (first slice — see below) | The MCP server is the flagship interface; 1.0 should ship more than tools-only query access |
| [#39](https://github.com/fmasi/mailrag/issues/39) | Unified config file (file < env < CLI precedence) | MCP settings are where config finally pays for itself; the issue itself targets landing alongside the MCP work |
| ~~[#97](https://github.com/fmasi/mailrag/issues/97)~~ **done** | One reproducible public recall@5 number — turnkey `make bench` on Enron-QA | Shipped in [#122](https://github.com/fmasi/mailrag/pull/122). Delivered less than the bar needs, though — see #123 below |
| [#123](https://github.com/fmasi/mailrag/issues/123) | Make **thread reconstruction** publicly measurable, by joining Enron-QA's questions to the full Enron maildir | `make bench` validates the hybrid layer only (+3.1/+4.4pp). Every lever behind the headline ladder — thread reconstruction (+29.1), summaries (+12.8), rerank (+2.5) — is switched off, so **45.6 → 93.3 remains author-reported**. The 1.0 bar says a stranger can verify the headline claim; today they verify the foundation under it |
| [#125](https://github.com/fmasi/mailrag/issues/125) | Make `make demo` showcase the value proposition — thread-aware vs naive RAG, side by side | The most-run command in the project currently proves the pipeline *runs*, not what it is *for*. Nothing is compared to anything, so the flagship claim is invisible. A stranger should learn "why this instead of a generic RAG?" in one screen |
| [#43](https://github.com/fmasi/mailrag/issues/43) | Clear the repo root: delete the orphaned `examples_advanced.py`, and relocate `main.py` out of the root as part of #125 | The first thing a visitor sees after `git clone` is a root `main.py` beside a polished CLI, which contradicts the documented entry point. Note `main.py` is **not** dead code — `scripts/quickstart.sh` runs it, so it must move rather than go; sequencing it with #125 avoids rewriting the same file twice |
| [#42](https://github.com/fmasi/mailrag/issues/42) | Finish the verb rename in code identifiers (pass1/pass2/explore/build → tag/summarize/scan/index) | Mass renames are exactly what a 1.0 boundary is for; doing it afterwards churns a supposedly stable codebase |
| [#117](https://github.com/fmasi/mailrag/issues/117) | Make the source story coherent — one front door over the loader (`.eml`, Enron) and sync (`IMAP`, Maildir) seams | 1.0 must actually be ready to take mail from more than one place, even shipping only two. Today "pluggable loaders" is advertised while all nine pipeline call sites hardcode one loader, and a user has to learn two vocabularies to answer "where can this get my mail from?" |
| [#118](https://github.com/fmasi/mailrag/issues/118) | Walk the whole product as a brand-new user on a clean machine, both sources, docs as written | Ready-for-users is a claim that has to be tested by someone starting from zero. The quickstart currently ends at the Enron demo and never mentions IMAP |
| [#34](https://github.com/fmasi/mailrag/issues/34) / [#76](https://github.com/fmasi/mailrag/issues/76) | Manual/integration passes: persona pipeline live, TUI end-to-end, doc screenshots | Release gate, not a feature — the parts the unit suite cannot reach (TTY, live LLM, GPU) get one honest human pass before 1.0 |

What is *not* in 1.0, deliberately: retrieval experiments
([#91](https://github.com/fmasi/mailrag/issues/91) and its children) — the
current numbers are already good and honestly benchmarked, and experiments
should not gate a release; rebuild parallelisation
([#102](https://github.com/fmasi/mailrag/issues/102)) — full rebuilds are now
rare events thanks to incremental sync; and all of the noise-pipeline research.

#### What "improve MCP capability" means concretely

The MCP server currently exposes seven tools (`list_collections`,
`search_email`, `get_thread`, `grep_email`, `answer_question`,
`list_attachments`, `get_attachment`) and nothing else — no structured filters,
no Prompts, no Resources. [#93](https://github.com/fmasi/mailrag/issues/93) is
the audit; its first slice is the 1.0 scope:

1. **Structured filters on search** — thread Qdrant payload filters through
   `search_email`/`get_thread` as optional arguments (`date_from`/`date_to`,
   sender, folder, `has_attachment`, `content_kind`, `thread_id`). Highest
   value, lowest effort: it turns semantic search into faceted search, and the
   index already carries the payload fields.
2. **A `compare_collections` tool** — the same query against two collections,
   diffed. Directly supports the retrieval A/B programme
   ([#91](https://github.com/fmasi/mailrag/issues/91)) and any
   v1-vs-v2 index decision.
3. **A small set of MCP Prompts** — canned retrieve-and-answer recipes
   ("summarise my week", "open commitments", "prep for meeting with X") that
   encode the pipeline's strengths as one-click flows.

The rest of the #93 roadmap — MCP Resources (`mailrag://thread/{id}`),
aggregation/facet tools, sampling/streaming — is v1.1+ material: valuable, but
none of it changes what 1.0 fundamentally is.

### v1.1 and later

Post-1.0, work shifts to the retrieval programme and throughput:

- [#91](https://github.com/fmasi/mailrag/issues/91) — the segmented retrieval
  A/B programme (rerank on pointed queries, RRF tuning), plus its individual
  experiments: [#14](https://github.com/fmasi/mailrag/issues/14) chunking sweep,
  [#13](https://github.com/fmasi/mailrag/issues/13) retrieval-oriented
  summaries, [#11](https://github.com/fmasi/mailrag/issues/11) per-thread
  summaries, [#1](https://github.com/fmasi/mailrag/issues/1) Qdrant server-side
  fusion as an opt-in fast path.
- [#102](https://github.com/fmasi/mailrag/issues/102) — parallelise `.eml`
  loading and attachment extraction for full rebuilds.
- [#5](https://github.com/fmasi/mailrag/issues/5) — store `chunk_index` for
  exact multi-chunk ordering; rides along with the next scheduled rebuild since
  it needs a re-ingest.
- [#48](https://github.com/fmasi/mailrag/issues/48) — split `mail_archive_x.py`
  into HTML→text, EML parsing and reply-stripping modules.
- [#20](https://github.com/fmasi/mailrag/issues/20) — one-command reproduction
  of the full canonical eval (the superset of
  [#97](https://github.com/fmasi/mailrag/issues/97), which ships the minimal
  public slice first).
- The remainder of [#93](https://github.com/fmasi/mailrag/issues/93): MCP
  Resources, aggregate/facet tools, citation-rich answers.

### Someday / undecided

Real ideas, honestly parked. They stay open because they document thinking, not
because they are scheduled.

- [#28](https://github.com/fmasi/mailrag/issues/28),
  [#30](https://github.com/fmasi/mailrag/issues/30),
  [#9](https://github.com/fmasi/mailrag/issues/9),
  [#10](https://github.com/fmasi/mailrag/issues/10) — noise-pipeline
  refinements (configurable drop stage, second-opinion verifier, finer LLM
  routing, spam-filtering survey). The current pipeline's precision is measured
  and acceptable; these buy marginal cost/quality improvements.
- [#6](https://github.com/fmasi/mailrag/issues/6) — wire the `thread_bound`
  token-budget knob. Explicitly deferred by design until a small-context LLM or
  an oversized thread actually forces it.
- [#95](https://github.com/fmasi/mailrag/issues/95) — periodic vector-DB
  ecosystem re-check. Re-run when a Qdrant limitation makes it relevant; no
  action otherwise.

## Release cadence and policy

One maintainer. The policy below is intentionally minimal — a process that
cannot be sustained is worse than no process.

**Versioning.** Semantic versioning, with the usual pre-1.0 caveat that minors
may break interfaces. From 1.0.0 onwards, the CLI verbs, the MCP tool
signatures, and the on-disk profile/collection formats are the public contract:
breaking any of them is a major.

**Cadence.** Milestone-driven, not calendar-driven. A minor release ships when
its milestone's scope is done — not on a date, and not with scope quietly
dropped to hit one. Patch releases ship on demand. Sequencing: v0.9.0 now;
v0.9.x patches as fixes land; v1.0.0 when the table above is closed; v1.1
planning after that. No calendar dates are promised, deliberately.

**Patch vs minor.**

- *Patch* (v0.9.x): security fixes, bug fixes, dependency bumps, documentation
  corrections. No new flags, tools, or behaviour changes.
- *Minor*: new capability (a new MCP tool, a new CLI verb, a new config layer)
  or a completed milestone.

**Security.** CI runs `pip-audit` with **zero ignore entries** — an advisory
fails the build rather than accumulating in an allow-list — and Dependabot
watches the dependency tree. Policy: an actionable advisory in a shipped
dependency triggers a patch release as soon as the fix is verified (recent
example: the pypdf CVE bump shipped within a day). If a fix is unreachable —
for instance the qdrant-client cap in
[#106](https://github.com/fmasi/mailrag/issues/106) — the exposure is
documented in an open issue and the release notes rather than silently ignored.

**Definition of done for a release.** All of:

1. Full test suite green in CI on `main` (branch protection already requires
   this for merge).
2. `pip-audit` clean, no new Dependabot alerts.
3. `docs/RELEASE_NOTES.md` entry written — what changed, what broke, what to do
   about it.
4. Docs updated for any behaviour change (per the project rules, this happens
   per-PR, so at release time it is a check, not a task).
5. Fresh-clone smoke test: install, `mailrag --help`, and the quickstart path
   run on a clean environment.
6. Tag pushed; GitHub release created from the release-notes entry.

That is the whole process. Anything heavier — release branches, RCs, backport
policies — is deferred until the project has users who need it.
