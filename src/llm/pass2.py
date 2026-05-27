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
        sender=email.get("sender", ""), subject=email.get("subject", ""),
        date=email.get("date"), body=email.get("body", ""),
        message_id=email.get("message_id", "") or "",
    )


def process_file(path: str, cache: Pass2Cache, load_email: Callable[[str], Dict],
                 summarize: Callable[[Dict], Dict], model: str) -> str:
    """Summarize+judge one file unless cached. Returns 'cached' | 'done' | 'error'."""
    sha = file_sha256(path)
    if cache.has(sha):
        return "cached"
    try:
        email = load_email(path)
        record = summarize(email)
        mid, chash = _identity_from(email)
        cache.put(sha, record, model=model, message_id=mid, content_sha256=chash)
        return "done"
    except Exception as exc:  # leave uncached so the next run retries it
        print(f"  pass2 error on {path}: {exc}")
        return "error"


def run_pass(paths: Iterable[str], cache: Pass2Cache,
             load_email: Callable[[str], Dict], summarize: Callable[[Dict], Dict],
             model: str, limit: Optional[int] = None,
             progress: bool = False) -> Dict[str, int]:
    """Sweep *paths*, summarizing uncached files. Returns outcome counts.

    When *progress* is true, show a tqdm bar over the whole corpus (rate + ETA)
    with a running done/cached/err breakdown. Already-cached files advance the
    bar quickly, so a resumed run shows true overall position.
    """
    counts = {"cached": 0, "done": 0, "error": 0}
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

    for path in paths:
        counts[process_file(path, cache, load_email, summarize, model)] += 1
        if bar is not None:
            bar.update(1)
            bar.set_postfix(done=counts["done"], cached=counts["cached"],
                            err=counts["error"], refresh=False)
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
        row = cache.get_resilient(file_sha256(email.source_id),
                                  message_id=mid, content_sha256=chash)
        if row is not None and not row["is_noise"] and row["summary"]:
            email.summary = row["summary"]
            n += 1
    return n
