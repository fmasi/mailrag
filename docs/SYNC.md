# Continuous sync — keeping a collection live

A mailrag collection built from a backup export is a **snapshot**. Ingest everything
up to 1 January and nothing after that date is ever searchable — the index quietly
rots while looking perfectly healthy. `mailrag sync` closes that gap.

The target is **1–2 days of freshness**, not real time. That choice buys a lot of
simplicity: no daemon, no IDLE connection held open, no push infrastructure. A
one-shot command on a timer is enough.

```bash
./mailrag sync                       # fetch new mail, judge it, index the delta
./mailrag sync --status              # what's fresh, what's pending, when it last ran
./mailrag sync --fetch-only          # just spool the mail (no LLM, no Qdrant)
./mailrag sync --install-agent --conda-env mailrag --model qwen2.5:14b
```

## How it works

Every stage of the existing pipeline — folder selection, noise filter, Pass-2
judging, attachment extraction, chunking, embedding — is driven by `.eml` files on
disk. So sync doesn't reimplement any of it. It fetches new mail, writes it as
`.eml` into the corpus tree, and lets the normal stages run over the delta:

```
 provider ──fetch──▶ .eml spool ──▶ pass1 tag ──▶ pass2 judge ──▶ incremental index
 (IMAP/Maildir)      <root>/incoming/           (Pass-2 cache:      (delete-by-key +
                       YYYY/MM/                  only NEW mail       deterministic ids,
                                                 costs LLM calls)    no rebuild)
```

Two existing properties make this cheap and safe:

- **The Pass-2 cache is content-addressed.** Re-running the judge stage over
  old+new mail costs LLM calls only for the new mail — and because the delta runs
  through the same profile and rubric, the cleaning stays consistent with the
  already-cleaned corpus. No separate "sync rubric" to drift.
- **Point ids are deterministic** ([`VERBS.md § index is incremental`](VERBS.md#index-is-incremental)).
  Indexing 40 new emails into a 20,000-email collection replaces exactly those
  emails' chunks and touches nothing else.

## Configuring an account

Accounts live in `~/.mailrag/accounts.yaml`. Each entry says where mail comes
from, which collection it feeds, and how often to look:

```yaml
accounts:
  - id: personal
    source: imap                       # or: maildir
    host: imap.mail.me.com
    port: 993
    login: you@example.com
    secret: keychain:mailrag.imap.personal
    collection: personal               # the Qdrant collection to keep fresh
    profile: ~/.mailrag/personal.json  # the cleaning profile / rubric to apply
    spool_root: ~/mail/personal/incoming
    exclude_roles: [junk, trash]
    cadence: 12h
    start_from: "2026-08-01"   # optional — see below
```

### `start_from`: don't re-download history you already have

With no `start_from`, the first run treats sync **as** the backfill and downloads
every message in every in-scope folder. That is correct when sync is how the
collection gets populated, and wrong when a backup export already covers history
— an iCloud inbox alone can be thousands of messages.

Set `start_from` to the date your export ends and the first run fetches only what
the export missed. It is resolved **server-side** (`UID SEARCH SINCE`), so the
watermark is placed without downloading anything first. If a folder has nothing
since that date it is marked caught up at its newest message; if the search fails,
mailrag falls back to a full sync rather than silently skipping mail it cannot
bound.

Multi-account falls out of this: several entries each get their own cursor and
ledger, and can target their own collection or share one. The same message
arriving at two of your addresses is kept per-account — that is meaningful
provenance, not a duplicate to be silently merged.

### Secrets are references, never literals

`secret:` holds a *reference*, and mailrag dereferences it at connect time:

| Form | Use |
|------|-----|
| `keychain:<service>` | macOS Keychain (default on macOS) |
| `env:<VAR>` | CI, containers |
| `file:<path>` | Linux; first line of a `0600` file |

A plaintext password is **rejected**, not merely discouraged — `accounts.yaml`
gets copied into repos, pasted into issues and swept into backups. Store one with:

```bash
security add-generic-password -U -a you@example.com -s mailrag.imap.personal -w
```

### Folder scope is expressed in *roles*

iCloud has no `SPECIAL-USE` and uses literal names (`Sent Messages`); Gmail has
labels; Dovecot advertises `\Sent`. So scope is set in roles — `inbox`, `sent`,
`archive`, `drafts`, `junk`, `trash`, `other` — and means the same thing
everywhere. The default is **everything except junk and trash**, which includes
`sent` (a thread without your own replies reads one-sided) and `other`
(arbitrary filed folders). Override a specific folder when the guess is wrong:

```yaml
    folder_roles:
      "Newsletters": junk
```

An unrecognised folder resolves to `other` — which is *in* scope — rather than
being guessed into `junk` and silently dropped.

## What happens when something is down

A scheduled sync runs unattended on a laptop that sleeps, changes networks, and
has Docker stopped half the time. So each stage runs only if its backend answers,
and anything it couldn't do is picked up next time:

| Unavailable | Behaviour |
|-------------|-----------|
| Network / IMAP | warn, exit cleanly, retry on the next tick |
| LLM endpoint | mail is still fetched and spooled; judging deferred |
| Qdrant | mail is still fetched and judged; indexing deferred |

This deliberately inverts `onboard`'s fail-fast, which is right for a six-hour
build and wrong for a background refresh. Nothing is lost, because the ledger
tracks each stage separately; nothing is repeated, because the cache and the
deterministic ids make re-running a stage free.

The run is also resumable at any point: the cursor is committed every 20 messages,
and a message that cannot be parsed is **parked with its error while the cursor
advances past it**. One poison message must never wedge a folder forever.

## Scheduling

`mailrag sync` is the portable unit. `--install-agent` writes the platform's
scheduler unit:

- **macOS** — a launchd LaunchAgent with `StartInterval`. Not cron: cron silently
  skips a window that passes while the machine is asleep; launchd runs the job on
  wake. For a laptop that spends nights closed, that's the difference between a
  fresh index and a stale one.
- **Linux** — a systemd user timer with `Persistent=true`, which behaves the same way.

The most common way this feature fails is a scheduler running mailrag outside its
conda environment, dying on the first import, silently, for weeks. So pass
`--conda-env`, and note that the unit always writes a log file. The backstop is
`sync --status`, which warns when the last successful sync was over 48 hours ago.

## Deletes

Deleting mail on the server does **not** remove it from the index. This is an
archive: the collection is built from a backup and keeps what it has seen. There
is no reconciliation sweep.

## Adding a provider

A source contributes exactly two provider-specific things — **how you enumerate
new mail** and **what a cursor is**. Everything else is already generic. Implement
[`MessageSource`](../src/sync/sources.py) and register it in
[`factory.py`](../src/sync/factory.py):

```python
class MessageSource(Protocol):
    name: str
    def capabilities(self) -> SourceCaps: ...
    def list_folders(self) -> list[Folder]: ...
    def open_folder(self, folder) -> Folder: ...   # returns it with a live generation
    def initial_cursor(self, folder) -> Cursor: ...
    def fetch_delta(self, folder, cursor) -> Iterator[RawMessage]: ...
    def advance(self, cursor, message) -> Cursor: ...
    def close(self) -> None: ...
```

Cursors are **opaque**: `(kind, value)` where the source owns `value` entirely, plus
a **generation** — the provider's "everything you knew is void" signal. That's
`UIDVALIDITY` for IMAP, a 404-on-`historyId` for Gmail, `cannotCalculateChanges`
for JMAP. One concept, one reset path, and a reset costs bandwidth alone because
the ledger recognises re-enumerated mail.

Shipped today: `imap` and `maildir`. The seam is designed for Gmail (`historyId`),
JMAP (`Email/changes`) and Microsoft Graph (delta tokens) without schema changes.

### Notes on iCloud specifically

- Endpoint `imap.mail.me.com:993`; an **app-specific password** is required.
- CONDSTORE/QRESYNC/IDLE are advertised **only after login** — a frontend proxy
  rewrites the pre-auth `CAPABILITY` response. mailrag re-reads capabilities post
  login, and treats CONDSTORE as an optimisation the UID watermark never needs.
- No `SPECIAL-USE`; map by name (handled by the role table).
- Roughly five concurrent connections, and Mail.app is probably holding some — so
  mailrag uses exactly one, with no pipelining.
- Fetches use `BODY.PEEK[]`, so syncing never marks your mail read.

Verified against a live account (2026-07-28): pre-auth `CAPABILITY` advertises 8
capabilities, post-auth 19 — CONDSTORE, QRESYNC, UIDPLUS, IDLE and ESEARCH all
appear only after login, and `MOVE` is absent. Of 26 folders, only two carried a
SPECIAL-USE flag (`\Sent`, `\Trash`); everything else — including `Junk`,
`Drafts` and `Archive` — was classified by the name table.

## Reference

| Path | What it is |
|------|------------|
| `~/.mailrag/accounts.yaml` | account config |
| `~/.mailrag/sync_state.db` | cursors, message ledger, run history (safe to delete — costs a re-enumerate) |
| `~/.mailrag/sync.log` | scheduled-run output |
| `<spool_root>/YYYY/MM/*.eml` | the spooled mail itself |

> The spool must be visible to the profile's `selection_rules`, or the delta is
> silently invisible to `resolve_index_files` and nothing new ever gets indexed.
