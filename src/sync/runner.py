"""Orchestrate one sync run: fetch -> judge -> index, skipping what is unavailable.

The defining property here is **spool-first stage skipping**. Onboarding is
right to fail fast — if Qdrant is down there is no point starting a six-hour
build. A scheduled sync is the opposite: it runs unattended on a laptop that
sleeps, moves between networks, and has Docker stopped half the time. So each
stage runs only if its backend answers, and a message that could not be judged
or indexed this run is simply picked up by the next one:

===================  =========================================================
Unavailable          Behaviour
===================  =========================================================
Network / IMAP       warn, exit cleanly, retry next tick
LLM endpoint         mail is still fetched and spooled; ``judged_at`` stays NULL
Qdrant               mail is still fetched and judged; ``indexed_at`` stays NULL
===================  =========================================================

Nothing is lost because the ledger records each stage separately, and nothing is
repeated because the Pass-2 cache and deterministic point ids make re-running a
stage over already-processed mail free.

The cursor is committed **per batch**, and it advances past a message that fails
to spool. A poison message parks with its error rather than blocking every later
message behind it forever — the one lesson worth taking from msgvault's sync.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from src.sync.accounts import AccountConfig
from src.sync.sources import Folder, MessageSource
from src.sync.spool import Spool, SpoolError
from src.sync.state import STATUS_FAILED, STATUS_OK, STATUS_PARTIAL, SyncState

log = logging.getLogger(__name__)

# Commit the cursor every N spooled messages. Small enough that an interruption
# costs little re-fetching, large enough to keep SQLite writes off the hot path.
CURSOR_COMMIT_EVERY = 20


class PermanentIndexError(RuntimeError):
    """The index stage refused for a reason retrying cannot fix.

    A legacy collection or an index-policy mismatch needs an operator decision
    (usually one ``--recreate``). Treating it as a transient outage would hide a
    permanent failure behind a "deferred" line, silently, every cadence tick.
    """


@dataclass
class SyncReport:
    """What one run did — the payload behind the CLI's summary line."""

    account_id: str
    fetched: int = 0
    already_had: int = 0
    judged: int = 0
    indexed: int = 0
    errors: int = 0
    folders_synced: int = 0
    folders_reset: int = 0
    skipped_stages: List[str] = field(default_factory=list)
    messages: List[str] = field(default_factory=list)

    @property
    def status(self) -> str:
        if self.skipped_stages or self.errors:
            return STATUS_PARTIAL
        return STATUS_OK

    def summary(self) -> str:
        parts = [
            f"fetched {self.fetched} (had {self.already_had})",
            f"judged {self.judged}",
            f"indexed {self.indexed}",
        ]
        if self.folders_reset:
            parts.append(f"{self.folders_reset} folder(s) re-enumerated")
        if self.errors:
            parts.append(f"{self.errors} error(s)")
        if self.skipped_stages:
            parts.append(f"skipped: {', '.join(self.skipped_stages)}")
        return "; ".join(parts)


def fetch_account(
    account: AccountConfig,
    source: MessageSource,
    state: SyncState,
    *,
    report: Optional[SyncReport] = None,
    limit: Optional[int] = None,
) -> SyncReport:
    """Fetch every in-scope folder into the spool, advancing cursors as it goes."""
    report = report or SyncReport(account_id=account.id)
    spool = Spool(account.resolved_spool_root())

    for listed in source.list_folders():
        if not account.wants(listed.role):
            continue
        try:
            folder = source.open_folder(listed)
        except Exception as exc:  # noqa: BLE001 — one bad folder must not end the run
            log.warning("skipping folder %s: %s", listed.name, exc)
            report.messages.append(f"folder {listed.name}: {exc}")
            report.errors += 1
            continue

        cursor, stored_generation = state.get_cursor(account.id, folder.name)
        if folder.generation and stored_generation and folder.generation != stored_generation:
            # The provider voided everything it previously told us (IMAP
            # UIDVALIDITY bump, mailbox recreated). Drop the cursor, keep the
            # ledger: re-enumeration then costs bandwidth, not LLM calls.
            log.info(
                "folder %s generation %s -> %s; re-enumerating",
                folder.name,
                stored_generation,
                folder.generation,
            )
            state.reset_folder(account.id, folder.name, folder.generation)
            cursor = None
            report.folders_reset += 1
        if cursor is None:
            cursor = source.initial_cursor(folder)

        report.folders_synced += 1
        cursor = _fetch_folder(account, source, state, spool, folder, cursor, report, limit=limit)
        state.set_cursor(
            account.id,
            folder.name,
            cursor,
            generation=folder.generation,
            role=folder.role.value,
        )
    return report


def _fetch_folder(account, source, state, spool, folder: Folder, cursor, report, *, limit=None):
    """Spool one folder's delta. Returns the cursor reached, committed periodically."""
    since_commit = 0
    for message in source.fetch_delta(folder, cursor):
        try:
            result = spool.write(message.raw)
        except SpoolError as exc:
            # Park it and move the cursor on regardless. A message that cannot be
            # parsed will never parse, and leaving the watermark behind it would
            # re-fetch it — and stall everything after it — on every future run.
            log.warning("could not spool %s/%s: %s", folder.name, message.source_uid, exc)
            # Record it before moving on, or the message vanishes: no ledger row,
            # no .eml, no UID anywhere, and --status honestly reporting 0 errors.
            # judged_at/indexed_at are pre-set so it never enters those stages.
            state.record_poison(
                account.id, folder=folder.name, source_uid=message.source_uid, error=str(exc)
            )
            report.messages.append(f"{folder.name}/{message.source_uid}: {exc}")
            report.errors += 1
            cursor = source.advance(cursor, message)
            continue

        if state.have_message(account.id, result.message_key):
            report.already_had += 1
        else:
            report.fetched += 1
        # Recorded either way: a second sighting retargets the row to the folder
        # it was most recently seen in without re-triggering any stage.
        state.record_fetched(
            account.id,
            message_key=result.message_key,
            content_sha256=result.content_sha256,
            message_id=result.message_id,
            folder=folder.name,
            source_uid=message.source_uid,
            eml_path=result.path,
            internal_date=message.internal_date.isoformat() if message.internal_date else None,
        )
        cursor = source.advance(cursor, message)

        since_commit += 1
        if since_commit >= CURSOR_COMMIT_EVERY:
            state.set_cursor(
                account.id,
                folder.name,
                cursor,
                generation=folder.generation,
                role=folder.role.value,
            )
            since_commit = 0
        if limit is not None and (report.fetched + report.already_had) >= limit:
            break
    return cursor


def judge_pending(
    account: AccountConfig,
    state: SyncState,
    *,
    profile,
    model: str,
    workers: int = 1,
    report: Optional[SyncReport] = None,
    run_pass_fn: Optional[Callable] = None,
) -> SyncReport:
    """Run the Pass-2 summarize/judge sweep over spooled-but-unjudged mail.

    Costs one LLM call per *new* email and nothing for anything the cache has
    already seen, so re-running after a failure is cheap. A dead endpoint marks
    the stage skipped rather than failing the run — the mail is already safely on
    disk, and the next tick will judge it.
    """
    report = report or SyncReport(account_id=account.id)
    pending = state.pending(account.id, "judged")
    if not pending:
        return report

    by_path = {r["eml_path"]: r["message_key"] for r in pending if r["eml_path"]}
    # A spool file that no longer exists is deterministic: leaving it in the list
    # means the same failure at the same position on every future run, stalling
    # the account permanently now that indexing waits on judging.
    missing = [p for p in by_path if not os.path.exists(p)]
    for path in missing:
        state.record_error(account.id, by_path[path], "spooled file is missing")
        report.errors += 1
        del by_path[path]
    if missing:
        report.messages.append(f"{len(missing)} spooled file(s) missing; skipped")
    if not by_path:
        return report

    # Collected per path, because the aggregate counts cannot say WHICH files
    # succeeded — and with workers > 1 results do not arrive in input order.
    # Inferring it positionally (as this did) marks failed messages judged and
    # leaves successful ones pending, permanently, since judged_at is never
    # cleared. Found in review of #101.
    succeeded: List[str] = []
    errored_paths: List[str] = []

    def on_outcome(path: str, outcome: str) -> None:
        if outcome in ("done", "cached"):
            succeeded.append(path)
        else:
            errored_paths.append(path)

    try:
        (run_pass_fn or _default_run_pass)(
            profile=profile,
            paths=list(by_path),
            model=model,
            workers=workers,
            on_outcome=on_outcome,
        )
    except Exception as exc:  # noqa: BLE001 — an unreachable LLM is expected, not exceptional
        log.warning("judge stage unavailable (%s); %d message(s) deferred", exc, len(by_path))
        report.skipped_stages.append("judge")
        report.messages.append(f"judge deferred: {exc}")
        # Whatever completed before the failure is still durably cached, so
        # crediting it costs nothing and avoids re-paying for it next run.
        if succeeded:
            report.judged += state.mark_judged(
                account.id, [by_path[p] for p in succeeded if p in by_path]
            )
        return report

    newly_judged = [by_path[p] for p in succeeded if p in by_path]
    # Anything already indexed WITHOUT a summary must go back through indexing now
    # that it has one — indexed_at is otherwise write-once, so the vector would
    # stay summary-less forever (found in review of the #101 fixes).
    requeued = state.clear_indexed(account.id, state.indexed_keys(account.id, newly_judged))
    if requeued:
        report.messages.append(f"{requeued} message(s) re-queued for indexing with their summary")
    report.judged += state.mark_judged(account.id, newly_judged)
    for path in errored_paths:
        key = by_path.get(path)
        if key:
            state.record_error(account.id, key, "pass-2 judge failed")
    report.errors += len(errored_paths)
    return report


def _default_run_pass(*, profile, paths, model, workers, on_outcome=None):
    """Bridge to the existing Pass-2 machinery, scoped to the delta's files."""
    from src.llm import client as llm_client  # noqa: PLC0415
    from src.llm import rubrics, summary  # noqa: PLC0415
    from src.llm.cache import Pass2Cache  # noqa: PLC0415
    from src.llm.pass2 import run_pass  # noqa: PLC0415
    from src.pipeline.pass2 import _make_load_email  # noqa: PLC0415

    cache = Pass2Cache(profile.pass2_cache)
    try:
        cl = llm_client.make_client()

        def summarize(email):
            return summary.parse_response(
                llm_client.chat(cl, model, rubrics.build_prompt(profile.rubric, email, 4000))
            )

        return run_pass(
            paths,
            cache,
            _make_load_email(4000),
            summarize,
            model,
            progress=False,
            workers=workers,
            on_outcome=on_outcome,
        )
    finally:
        cache.close()


def index_pending(
    account: AccountConfig,
    state: SyncState,
    *,
    profile,
    embedder,
    report: Optional[SyncReport] = None,
    index_fn: Optional[Callable] = None,
    require_judged: bool = True,
    embed_summary: bool = True,
) -> SyncReport:
    """Index spooled-but-unindexed mail into the account's collection, incrementally.

    Uses ``recreate=False``, which is only safe because point ids are
    deterministic and each email's existing points are deleted before its new
    ones land (issue #101, slice 1).
    """
    report = report or SyncReport(account_id=account.id)
    pending = state.pending(account.id, "indexed")

    if require_judged:
        # Indexing unjudged mail is permanent: indexed_at is set once and never
        # cleared, so an email indexed during an LLM outage would keep its
        # summary-less vector forever even after the summary arrived. Wait for
        # the judge stage instead — the mail is safe on disk meanwhile.
        deferred = [r for r in pending if r["judged_at"] is None]
        pending = [r for r in pending if r["judged_at"] is not None]
        if deferred:
            report.messages.append(f"{len(deferred)} message(s) awaiting judge before indexing")

    by_path = {r["eml_path"]: r["message_key"] for r in pending if r["eml_path"]}
    if not by_path:
        return report

    try:
        result = (index_fn or _default_index)(
            profile=profile,
            embedder=embedder,
            paths=list(by_path),
            collection=account.collection,
            embed_summary=embed_summary,
        )
    except PermanentIndexError as exc:
        # An operator-fixable refusal (legacy collection, policy mismatch). Filing
        # it as a transient outage would repeat it silently every cadence tick.
        log.error("index stage refused: %s", exc)
        report.messages.append(f"index REFUSED (needs operator action): {exc}")
        report.errors += 1
        return report
    except Exception as exc:  # noqa: BLE001 — Qdrant in Docker is routinely down
        log.warning("index stage unavailable (%s); %d message(s) deferred", exc, len(by_path))
        report.skipped_stages.append("index")
        report.messages.append(f"index deferred: {exc}")
        return report

    # (chunks, handled_keys) — handled means "the indexer is done with it",
    # which includes emails it deliberately DROPPED (confident noise) as well as
    # those it wrote. Marking only what was written would leave every pruned
    # noise email pending forever, re-loaded on every tick: a backlog that grows
    # and never drains. Only the dedup losers are left pending, and those
    # self-heal because their surviving twin is not in the next delta.
    chunks, handled_keys = result
    keys = [k for k in by_path.values() if k in handled_keys]
    skipped = len(by_path) - len(keys)
    report.indexed += state.mark_indexed(account.id, keys)
    report.messages.append(f"indexed {chunks} chunk(s)")
    if skipped:
        report.messages.append(
            f"{skipped} message(s) deduped against existing mail (retry next run)"
        )
    return report


def _default_index(*, profile, embedder, paths, collection, embed_summary=True):
    """Load the delta's files and append them to the collection.

    Returns ``(chunks, indexed_message_keys)``. The guards are checked BEFORE the
    expensive load/pass-2/OCR work, so a permanent refusal costs one round trip
    rather than re-doing the whole delta on every scheduled tick.
    """
    from src.data.loaders.mail_archive_x import MailArchiveXLoader  # noqa: PLC0415
    from src.data.noise_filter import NoiseFilter  # noqa: PLC0415
    from src.indexing.attachment_docs import build_attachment_documents  # noqa: PLC0415
    from src.indexing.contextual_index import build_contextual_index  # noqa: PLC0415
    from src.indexing.policy import describe_mismatch, policy_fingerprint  # noqa: PLC0415
    from src.ingest import hybrid_qdrant as hq  # noqa: PLC0415
    from src.llm.cache import Pass2Cache  # noqa: PLC0415
    from src.llm.pass2 import apply_pass2  # noqa: PLC0415
    from src.pipeline import pass1  # noqa: PLC0415

    # Fail fast on the operator-fixable refusals, before any work is done.
    client = hq.get_client(profile.qdrant_url)
    if hq.has_legacy_points(client, collection):
        raise PermanentIndexError(
            f"collection '{collection}' predates deterministic point ids; appending would "
            "duplicate every chunk. Rebuild once with `mailrag index --recreate`."
        )
    existing_policy = hq.collection_policy(client, collection)
    incoming = policy_fingerprint(
        chunk_size=profile.chunk_size,
        chunk_overlap=profile.chunk_overlap,
        embed_summary=embed_summary,
        embedder_name=type(embedder).__name__,
        dim=getattr(embedder, "dim", 1024),
    )
    if existing_policy and existing_policy != incoming:
        raise PermanentIndexError(describe_mismatch(collection, existing_policy, incoming))

    emails = MailArchiveXLoader(eml_files=list(paths)).load()
    loaded_keys = {e.message_key() for e in emails}
    emails, _stats = pass1.run(emails, NoiseFilter.from_project_rules())

    if profile.pass2_cache:
        cache = Pass2Cache(profile.pass2_cache)
        try:
            emails, _dropped = apply_pass2(emails, cache)
        finally:
            cache.close()

    # Everything that reached the indexer. Anything in `loaded_keys` but not here
    # was deliberately dropped (confident noise, pass-1 rules) and is DONE — not
    # pending. Anything here that produces no chunks lost the corpus-wide dedup
    # and should be retried, so it is excluded from `handled` below.
    reached_indexer = {e.message_key() for e in emails}

    attachment_docs = build_attachment_documents(
        [e.source_id for e in emails if getattr(e, "source_id", None)],
        extractor_name="tesseract",
    )
    result = build_contextual_index(
        emails,
        collection=collection,
        embedder=embedder,
        chunk_size=profile.chunk_size,
        chunk_overlap=profile.chunk_overlap,
        embed_summary=embed_summary,
        recreate=False,  # incremental — see issue #101 slice 1
        qdrant_url=profile.qdrant_url,
        apply_noise_filter=False,
        extra_docs=attachment_docs,
    )
    # handled = written  ∪  deliberately dropped  ∪  unparseable-at-load.
    # Only the dedup losers stay pending.
    deduped_away = reached_indexer - set(result.indexed_message_keys)
    handled = (loaded_keys | reached_indexer) - deduped_away
    return result.chunks, handled


def sync_account(
    account: AccountConfig,
    *,
    state: SyncState,
    source_factory: Callable[[AccountConfig], MessageSource],
    profile=None,
    embedder=None,
    model: str = "",
    workers: int = 1,
    fetch_only: bool = False,
    limit: Optional[int] = None,
    embed_summary: bool = True,
) -> SyncReport:
    """Run one full sync for one account. Never raises for an expected outage.

    The run record is opened first and always closed, so ``--status`` can tell
    "never ran" from "ran and failed" — and a run killed mid-flight is superseded
    by the next one rather than blocking it.
    """
    report = SyncReport(account_id=account.id)
    run_id = state.start_run(account.id)
    source = None
    try:
        try:
            source = source_factory(account)
            fetch_account(account, source, state, report=report, limit=limit)
        except Exception as exc:  # noqa: BLE001 — no network is normal on a laptop
            log.warning("fetch stage unavailable (%s)", exc)
            report.skipped_stages.append("fetch")
            report.messages.append(f"fetch deferred: {exc}")
        finally:
            if source is not None:
                source.close()

        # Stages below still run on whatever is already spooled, so an outage in
        # one stage never strands mail that an earlier run fetched.
        judged_configured = bool(model)
        if not fetch_only and profile is not None:
            if judged_configured:
                judge_pending(
                    account, state, profile=profile, model=model, workers=workers, report=report
                )
            elif state.pending(account.id, "judged"):
                # Without a model there IS no judge stage. Say so — reporting a
                # clean "ok" while silently indexing unjudged mail hides the
                # single most consequential thing about the run.
                report.skipped_stages.append("judge (no --model)")
        if not fetch_only and profile is not None and embedder is not None:
            index_pending(
                account,
                state,
                profile=profile,
                embedder=embedder,
                report=report,
                # With no judge stage configured, waiting for judgments that will
                # never come would mean never indexing at all.
                require_judged=judged_configured,
                embed_summary=embed_summary,
            )

        state.finish_run(
            run_id,
            status=report.status,
            fetched=report.fetched,
            judged=report.judged,
            indexed=report.indexed,
            errors=report.errors,
            message="; ".join(report.messages)[:2000],
        )
    except Exception as exc:  # noqa: BLE001 — last resort: never leave a run 'running'
        state.finish_run(run_id, status=STATUS_FAILED, message=str(exc))
        raise
    return report
