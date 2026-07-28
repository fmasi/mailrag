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

    paths = [r["eml_path"] for r in pending if r["eml_path"]]
    if not paths:
        return report

    try:
        counts = (run_pass_fn or _default_run_pass)(
            profile=profile, paths=paths, model=model, workers=workers
        )
    except Exception as exc:  # noqa: BLE001 — an unreachable LLM is expected, not exceptional
        log.warning("judge stage unavailable (%s); %d message(s) deferred", exc, len(paths))
        report.skipped_stages.append("judge")
        report.messages.append(f"judge deferred: {exc}")
        return report

    # Only mark what actually got a judgment; errors stay pending so the next run
    # retries them rather than indexing them unjudged forever.
    errored = int(counts.get("error", 0))
    judged_rows = pending if not errored else pending[: max(0, len(pending) - errored)]
    report.judged += state.mark_judged(account.id, [r["message_key"] for r in judged_rows])
    report.errors += errored
    return report


def _default_run_pass(*, profile, paths, model, workers):
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
) -> SyncReport:
    """Index spooled-but-unindexed mail into the account's collection, incrementally.

    Uses ``recreate=False``, which is only safe because point ids are
    deterministic and each email's existing points are deleted before its new
    ones land (issue #101, slice 1).
    """
    report = report or SyncReport(account_id=account.id)
    pending = state.pending(account.id, "indexed")
    paths = [r["eml_path"] for r in pending if r["eml_path"]]
    if not paths:
        return report

    try:
        chunks = (index_fn or _default_index)(
            profile=profile, embedder=embedder, paths=paths, collection=account.collection
        )
    except Exception as exc:  # noqa: BLE001 — Qdrant in Docker is routinely down
        log.warning("index stage unavailable (%s); %d message(s) deferred", exc, len(paths))
        report.skipped_stages.append("index")
        report.messages.append(f"index deferred: {exc}")
        return report

    report.indexed += state.mark_indexed(account.id, [r["message_key"] for r in pending])
    report.messages.append(f"indexed {chunks} chunk(s)")
    return report


def _default_index(*, profile, embedder, paths, collection):
    """Load the delta's files and append them to the collection."""
    from src.data.loaders.mail_archive_x import MailArchiveXLoader  # noqa: PLC0415
    from src.data.noise_filter import NoiseFilter  # noqa: PLC0415
    from src.indexing.attachment_docs import build_attachment_documents  # noqa: PLC0415
    from src.indexing.contextual_index import build_contextual_index  # noqa: PLC0415
    from src.llm.cache import Pass2Cache  # noqa: PLC0415
    from src.llm.pass2 import apply_pass2  # noqa: PLC0415
    from src.pipeline import pass1  # noqa: PLC0415

    emails = MailArchiveXLoader(eml_files=list(paths)).load()
    emails, _stats = pass1.run(emails, NoiseFilter.from_project_rules())

    if profile.pass2_cache:
        cache = Pass2Cache(profile.pass2_cache)
        try:
            emails, _dropped = apply_pass2(emails, cache)
        finally:
            cache.close()

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
        embed_summary=True,
        recreate=False,  # incremental — see issue #101 slice 1
        qdrant_url=profile.qdrant_url,
        apply_noise_filter=False,
        extra_docs=attachment_docs,
    )
    return result.chunks


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
        if not fetch_only and profile is not None and model:
            judge_pending(
                account, state, profile=profile, model=model, workers=workers, report=report
            )
        if not fetch_only and profile is not None and embedder is not None:
            index_pending(account, state, profile=profile, embedder=embedder, report=report)

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
