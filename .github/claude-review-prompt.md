# Review rubric: mailrag

This file is read by BOTH the CI Claude review (fmasi/.github `claude-review.yml`) and the local
`/ci-review` command, so a branch that passes `/ci-review` should pass CI's review too.

**What mailrag protects.** It turns a person's real mailbox (bodies, addresses, subjects,
attachments, calendar invites) into an index that LLMs and MCP-connected agents query. Three facts
drive this review:
- **Every email is attacker-controlled input.** Anyone can send one, so any text that came out of
  a message (body, subject, headers, attachment content or filename) is untrusted data.
- **The owner's mail must not leak.** Not into logs, errors, fixtures or git, not across corpora
  (the work and personal collections share one `.eml` root), and not to a remote endpoint the
  operator didn't choose.
- **The owner doesn't read code.** This review and the CI checks are the safety net.

Review the change, not the whole codebase. Read the surrounding code when a hunk depends on it.
Report only material problems: no praise, no summary of what the code does, no style nits that
ruff and mypy already catch.

## What to check, in priority order

1. **Correctness.** Logic errors; edge cases (empty, missing, zero, very large, unicode, time
   zones, malformed MIME, duplicate Message-IDs, concurrent callers); off-by-one; error handling
   that swallows, misreports or retries forever; resources not released; the sync ledger's state
   transitions and cursor persistence (`src/sync/`); behaviour that contradicts the PR description.
2. **Prompt injection through email content.** Text from a message that reaches an LLM prompt
   (`src/llm/`: answers, summaries, pass-2 judging, onboarding) or an MCP tool result
   (`src/mcp_server/`) must be fenced and marked as untrusted data, never spliced in as
   instructions, and stay size-bounded (`HARD_SEARCH_MAX_CHARS` and friends). The MCP server
   stays read-only: no tool may send, move, delete or write mail or files, and `get_attachment`
   never returns raw bytes. Parsers (eml, pypdf, OCR, docx/xlsx/pptx) treat content as hostile:
   limits on size and pages, and no email-derived string (an attachment filename, a subject)
   ever becomes a filesystem path, a shell argument or SQL without sanitising or binding.
3. **Personal data and secrets in logs.** No email body, address, subject or person-identifying
   id, API key, IMAP password or keychain value in `print`, logging, exception messages,
   `~/.mailrag/sync.log`, `sync_runs.message`, MCP usage logs or eval outputs. Name the setting or
   the message count, never the value (CodeQL alert #14 was a settings warning echoing an API
   key). `src/config/secrets.py` (env:, file:, keychain:) never echoes what it rejects, and
   plaintext passwords in `accounts.yaml` stay rejected. Test fixtures are synthetic
   (`example.com`/`x.com` addresses, public Enron data), never real mail.
4. **Cross-corpus leaks.** Every read path (vector search, hybrid search, `grep_email`,
   attachments, thread fetch, summaries, usage reports) is scoped to the requested collection
   through `src/mcp_server/scoping.py` or the collection filter; a refusal names both corpora.
   A new tool or query path without scoping is Critical. Scoping caches must still invalidate
   when a profile or manifest changes.
5. **Destructive and irreversible operations.** Qdrant collection deletes or recreates,
   re-index migrations, point-ID scheme changes (`src/indexing/point_ids.py`), sync-state or
   attachment-store schema changes, prune/blacklist runs: they need an explicit operator opt-in
   (never a default, never reachable from the MCP server), a dry run or a way back, and tests.
6. **Local-first and data egress.** Sending mail content to a new remote endpoint, making a
   hosted LLM or embedding provider the default, adding telemetry, or classifying a remote host
   as local (`src/llm/provenance.py`) must be opt-in and stated in the PR.
7. **Security floors and the supply chain. Never weaken them.** Don't lower or delete a floor in
   `pyproject.toml` (the "Transitive security floors" block, the pypdf, pillow, datasets and
   cryptography floors, the `tornado` dev floor) without a stated, advisory-level reason. The
   `qdrant-client <1.19` cap stays until #106 is resolved. A new `--ignore-vuln` must name an
   advisory with NO fixed release, carry its reason, and appear in `ci.yml`, the justfile's
   `audit_ignores` and `dependency-review.yml`'s `allow-ghsas`. `pyproject.toml` and `poetry.lock` change together.
8. **Tests.** New or changed behaviour has tests in the `unittest.TestCase` style of `tests/`:
   the happy path, edge cases and invalid input. A bug fix has a test that fails without it.
   LLMs, the Qdrant server, IMAP and model downloads are mocked (or use the in-memory Qdrant); no
   network in unit tests. Name the untested behaviour.
9. **Consistency and docs.** Follows `AGENTS.md`, `CLAUDE.md` and the surrounding code; no dead
   code, no duplicate of an existing helper. Behaviour changes reach docstrings and `docs/`;
   every published number goes to `docs/CLAIMS.md` with its source; nothing in the docs is now
   false.
10. **CI hygiene** (only when workflows, the justfile, lefthook.yml or `.github/` change). Draft
    skip, concurrency, `timeout-minutes`, SHA-pinned actions, no `push` on all branches, no
    `paths-ignore` on a required check (`pytest`, `pip-audit`, `CodeQL (…)`), a renamed job
    means a ruleset update, and `just ci` still mirrors CI.

## What this review cannot see

Say so when a change depends on one of these, and ask how it was verified:
- **Integration tests** (live Qdrant + LLM + bge-m3, marker `integration`) are deselected in CI
  and never run there; unit tests mock most of that stack (e.g. `test_hybrid.py` asserts the
  wiring, not the search results).
- **Real mail.** CI has only synthetic and public Enron data: MIME edge cases, encodings and
  mailbox sizes from real accounts are untested.
- **The macOS Keychain, IMAP servers and LM Studio** don't exist on the CI runner.

## Out of scope

- `eval/**`: committed eval fixtures (data, not source). A multi-megabyte JSONL diff there once
  spent the whole review budget before reaching any code (#144). Don't read them; for a
  fixture-only PR say so in one line.
- `poetry.lock` line by line: only check which packages moved and that no security floor went
  down.

## How to rank

- **Critical**: breaks the build or the sync/index pipeline, loses or corrupts mail data or the
  index, leaks a secret or personal data (logs, errors, fixtures, a new egress), lets one corpus
  read another, gives the MCP server a write or send path, splices untrusted email text into a
  prompt as instructions on a new path, adds a destructive default, or weakens a security floor
  or ignores an advisory that has a fix. Must fix before merge. One Critical finding makes the
  CI review end with `REVIEW-VERDICT: BLOCK` (label `claude-blocked`), which blocks the merge
  until a re-review passes. Don't inflate: a Critical is something the owner would roll back for.
- **Important**: a real bug on a path that will happen, missing error handling, changed
  behaviour without a test, or docs that are now wrong. Should fix before merge.
- **Minor**: worth fixing, safe to merge without.

## Output

One line per finding, most severe first:
`[Critical|Important|Minor] path/to/file:line: what is wrong, why it matters, the fix.`
If nothing material turns up, say so in one line.
