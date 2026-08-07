# src/llm/pass2.py
"""Orchestration for the LLM Pass-2 file sweep.

The core takes ``load_email`` and ``summarize`` callables so it can be tested
without a network or a real loader. ``file_sha256`` ties cache rows to the same
content-addressed hash space as the blacklist.
"""

from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Optional

from src.data.blacklist import file_sha256
from src.data.identity import email_identity
from src.llm.cache import Pass2Cache


def _identity_from(email: Dict):
    """Derive (message_id, content_sha256) from a load_email dict, tolerating
    missing keys (older loaders may not surface a Message-ID)."""
    return email_identity(
        sender=email.get("sender", ""),
        subject=email.get("subject", ""),
        date=email.get("date"),
        body=email.get("body", ""),
        message_id=email.get("message_id", "") or "",
    )


# Failures of the ENDPOINT rather than of the message. These must never consume
# a message's retry budget: a closed LM Studio reports identically for every
# message, and counting those as per-message failures abandoned entire backlogs.
_TRANSIENT_EXC_NAMES = (
    "APIConnectionError",
    "APITimeoutError",
    "InternalServerError",
    "RateLimitError",
    "ServiceUnavailable",
    # Auth failures are endpoint-level and operator-fixable — LM Studio now
    # requires a token, and a missing key returns 401 for EVERY message. Charging
    # that to the messages would abandon a whole backlog over a config mistake.
    "AuthenticationError",
    "PermissionDeniedError",
    "LLMHealthcheckError",
)


def _prov_kwargs(model: str, provenance) -> Dict[str, str]:
    """Cache columns describing the judge. Falls back to the bare model id."""
    if provenance is None:
        return {"model": model}
    from src.llm.provenance import model_fingerprint  # noqa: PLC0415

    return {
        "model": model_fingerprint(provenance) or model,
        "quant": provenance.quant,
        "endpoint": provenance.endpoint,
        "source": provenance.source,
    }


def classify_failure(exc: BaseException) -> str:
    """``"unavailable"`` if *exc* means the endpoint is down, else ``"error"``.

    Walks the ``__cause__``/``__context__`` chain because the OpenAI-compatible
    clients wrap the underlying socket error.
    """
    seen = 0
    cur: BaseException | None = exc
    while cur is not None and seen < 10:
        if isinstance(cur, (ConnectionError, TimeoutError)) or type(cur).__name__ in (
            _TRANSIENT_EXC_NAMES
        ):
            return "unavailable"
        if isinstance(cur, OSError) and not isinstance(cur, FileNotFoundError):
            return "unavailable"
        cur = cur.__cause__ or cur.__context__
        seen += 1
    return "error"


def process_file(
    path: str,
    cache: Pass2Cache,
    load_email: Callable[[str], Dict],
    summarize: Callable[[Dict], Dict],
    model: str,
    provenance=None,
) -> str:
    """Summarize+judge one file unless cached. Returns 'cached' | 'done' | 'error'."""
    try:
        sha = file_sha256(path)
    except OSError as exc:
        # A missing/unreadable .eml must not abort the sweep. It is deterministic,
        # so aborting means the same file kills every future run at the same
        # position and nothing after it is ever processed.
        print(f"  pass2 error on {path}: {exc}")
        return "error"
    if cache.has(sha):
        return "cached"
    try:
        email = load_email(path)
        record = summarize(email)
        mid, chash = _identity_from(email)
        cache.put(
            sha, record, message_id=mid, content_sha256=chash, **_prov_kwargs(model, provenance)
        )
        return "done"
    except Exception as exc:  # leave uncached so the next run retries it
        outcome = classify_failure(exc)
        print(f"  pass2 {outcome} on {path}: {exc}")
        return outcome


def run_pass(
    paths: Iterable[str],
    cache: Pass2Cache,
    load_email: Callable[[str], Dict],
    summarize: Callable[[Dict], Dict],
    model: str,
    limit: Optional[int] = None,
    progress: bool = False,
    workers: int = 1,
    on_outcome: Optional[Callable[[str, str], None]] = None,
    provenance=None,
) -> Dict[str, int]:
    """Sweep *paths*, summarizing uncached files. Returns outcome counts.

    When *progress* is true, show a tqdm bar over the whole corpus (rate + ETA)
    with a running done/cached/err breakdown. Already-cached files advance the
    bar quickly, so a resumed run shows true overall position.

    When *workers* > 1, load+summarize (network-bound LLM calls) run in a thread
    pool with that many in-flight requests; all SQLite cache reads/writes stay on
    the calling thread so the single connection is never touched concurrently.

    *on_outcome* is called as ``on_outcome(path, outcome)`` for every path, with
    outcome in ``{"cached", "done", "error"}``.  Aggregate counts alone cannot
    tell a caller WHICH files succeeded — and with ``workers > 1`` results do not
    even arrive in input order — so any caller that records per-message state
    must use this rather than inferring from the counts.
    """
    counts = {"cached": 0, "done": 0, "error": 0, "unavailable": 0}

    def _record(path: str, outcome: str) -> None:
        counts[outcome] += 1
        if on_outcome is not None:
            on_outcome(path, outcome)

    paths = list(paths)
    if limit is not None:
        paths = paths[:limit]

    bar = None
    if progress:
        try:
            from tqdm import tqdm

            bar = tqdm(total=len(paths), unit="email", desc="pass2", smoothing=0.05)
        except ImportError:  # bar is optional; fall back to silent sweep
            bar = None

    def _tick():
        if bar is not None:
            bar.update(1)
            bar.set_postfix(
                done=counts["done"], cached=counts["cached"], err=counts["error"], refresh=False
            )

    if workers and workers > 1:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        # Split cached vs. todo on this thread (serial cache.has), then fan the
        # LLM work out; cache.put happens here as each future lands.
        todo = []
        for path in paths:
            try:
                sha = file_sha256(path)
            except OSError as exc:  # see process_file: never abort the sweep
                print(f"  pass2 error on {path}: {exc}")
                _record(path, "error")
                _tick()
                continue
            if cache.has(sha):
                _record(path, "cached")
                _tick()
            else:
                todo.append((path, sha))

        def _work(item):
            path, sha = item
            email = load_email(path)
            record = summarize(email)
            mid, chash = _identity_from(email)
            return path, sha, record, mid, chash

        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(_work, item): item for item in todo}
            for fut in as_completed(futures):
                try:
                    path, sha, record, mid, chash = fut.result()
                    cache.put(
                        sha,
                        record,
                        message_id=mid,
                        content_sha256=chash,
                        **_prov_kwargs(model, provenance),
                    )
                    _record(path, "done")
                except Exception as exc:  # leave uncached so a rerun retries it
                    outcome = classify_failure(exc)
                    print(f"  pass2 {outcome} on {futures[fut][0]}: {exc}")
                    _record(futures[fut][0], outcome)
                _tick()
        if bar is not None:
            bar.close()
        return counts

    for path in paths:
        _record(path, process_file(path, cache, load_email, summarize, model, provenance))
        _tick()
    if bar is not None:
        bar.close()
    return counts


def noise_hashes(cache: Pass2Cache, min_confidence: float) -> List[str]:
    """File shas judged noise at/above *min_confidence* (for blacklisting)."""
    return [row["sha256"] for row in cache.iter_noise(min_confidence)]


def agreement_rate(local: Dict[str, bool], ref: Dict[str, bool]) -> float:
    """Fraction of shared keys where the two is_noise judgments agree."""
    keys = set(local) & set(ref)
    if not keys:
        return 0.0
    return sum(1 for k in keys if local[k] == ref[k]) / len(keys)


def sample_files(paths: Iterable[str], n: Optional[int], seed: int = 0) -> List[str]:
    """Deterministic random sample of *n* paths (all of them if n is None or >= len).

    Used by ``run --sample`` to spot-check a representative subset, unlike
    ``--limit`` which takes the first N in resolution order.
    """
    import random

    items = list(paths)
    if n is None or n >= len(items):
        return items
    return random.Random(seed).sample(items, n)


def apply_pass2(emails, cache: Pass2Cache, min_confidence: float = 0.7):
    """Apply cached Pass-2 judgments to *emails* before indexing.

    Drops emails judged noise at/above *min_confidence* (the confident-noise
    prune that saves DB space) and injects the cached summary onto each kept,
    non-noise email. Emails with no cache row — or noise below the threshold —
    are KEPT untouched, so a partial/resumable cache never silently loses mail
    (zero-loss stays the default; only confident noise is removed). The raw
    ``.eml`` files remain on disk, so a drop is reversible by rebuilding.

    Returns ``(kept_emails, dropped_count)``. ``source_id`` is the file path.
    """
    kept = []
    dropped = 0
    for email in emails:
        mid, chash = email_identity(
            sender=getattr(email, "sender", "") or "",
            subject=getattr(email, "subject", "") or "",
            date=getattr(email, "date", None),
            body=getattr(email, "body", "") or "",
            message_id=getattr(email, "message_id", "") or "",
        )
        row = cache.get_resilient(
            file_sha256(email.source_id), message_id=mid, content_sha256=chash
        )
        if row is not None and row["is_noise"] and row["confidence"] >= min_confidence:
            dropped += 1
            continue
        if row is not None and not row["is_noise"] and row["summary"]:
            email.summary = row["summary"]
        kept.append(email)
    return kept, dropped


def inject_summaries(emails, cache: Pass2Cache) -> int:
    """Set ``.summary`` on each email whose source file has a kept (non-noise)
    cached summary. Returns how many were set. ``source_id`` is the file path."""
    n = 0
    for email in emails:
        mid, chash = email_identity(
            sender=getattr(email, "sender", "") or "",
            subject=getattr(email, "subject", "") or "",
            date=getattr(email, "date", None),
            body=getattr(email, "body", "") or "",
            message_id=getattr(email, "message_id", "") or "",
        )
        # Exact file-bytes match first, then fall back to stable identifiers so a
        # re-exported mailbox still reuses the cached summary.
        row = cache.get_resilient(
            file_sha256(email.source_id), message_id=mid, content_sha256=chash
        )
        if row is not None and not row["is_noise"] and row["summary"]:
            email.summary = row["summary"]
            n += 1
    return n
