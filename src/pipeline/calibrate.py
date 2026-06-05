"""Calibrate stage: judge a deterministic sample with the profile's rubric and
bucket the suspected mistakes — the forcing function before a full Pass-2.

It reuses the Pass-2 loader shape, ``sample_files`` for a deterministic subset,
and the same parse path, but collects records directly and writes NOTHING to the
real Pass-2 cache (it is a throwaway look, not a sweep).
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List

from src.ingest.local_source import resolve_index_files
from src.llm.pass2 import sample_files
from src.llm import client as llm_client
from src.llm import summary, rubrics, calibration


def _make_load_email(body_chars: int) -> Callable[[str], Dict[str, Any]]:
    # body_chars is accepted for signature parity with the pass2 loader; the body
    # is actually truncated later in rubrics.build_prompt, not at load time.
    from src.data.loaders.mail_archive_x import MailArchiveXLoader

    def load_email(path: str) -> Dict[str, Any]:
        emails = list(MailArchiveXLoader(eml_files=[path], verbose=False).load())
        if not emails:
            raise ValueError("no email parsed")
        e = emails[0]
        return {"sender": e.sender, "subject": e.subject,
                "date": e.date.isoformat() if e.date else "unknown",
                "body": e.body, "message_id": e.message_id or ""}
    return load_email


def judge_sample(paths: List[str], load_email: Callable[[str], Dict[str, Any]],
                 judge: Callable[[Dict[str, Any]], Dict[str, Any]],
                 workers: int = 4, progress: bool = False) -> List[Dict[str, Any]]:
    """Judge each path into a flat record; errored paths are skipped.

    ``judge(email_dict) -> {is_noise, confidence, summary, reason}``. Parallelized
    with a thread pool when *workers* > 1 (network-bound LLM calls).
    """
    def _one(path: str) -> Dict[str, Any]:
        e = load_email(path)
        j = judge(e)
        return {"sender": e.get("sender", ""), "subject": e.get("subject", ""),
                # is_noise is guaranteed by summary.parse_response (raises if absent); a
                # missing key here is a real error and is counted as a skip by the caller.
                "is_noise": bool(j["is_noise"]), "confidence": j.get("confidence", 0.0),
                "summary": j.get("summary", ""), "reason": j.get("reason", "")}

    records: List[Dict[str, Any]] = []
    skipped = 0
    bar = None
    if progress:
        try:
            from tqdm import tqdm
            bar = tqdm(total=len(paths), unit="email", desc="calibrate", smoothing=0.05)
        except ImportError:
            bar = None

    def _tick():
        if bar is not None:
            bar.update(1)

    if workers and workers > 1:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_one, p) for p in paths]
            for f in as_completed(futs):
                try:
                    records.append(f.result())
                except Exception:  # skip unparseable/erroring emails in the sample
                    skipped += 1
                _tick()
    else:
        for p in paths:
            try:
                records.append(_one(p))
            except Exception:
                skipped += 1
            _tick()
    if bar is not None:
        bar.close()
    if skipped:
        print(f"  calibrate: skipped {skipped}/{len(paths)} emails (load/judge errors)")
    return records


def run(profile, *, model: str, sample: int = 200, seed: int = 11, workers: int = 4,
        body_chars: int = 4000, progress: bool = False) -> calibration.CalibrationReport:
    """Sample *sample* files deterministically (by *seed*), judge each with the
    profile's rubric, and return a :class:`CalibrationReport`. Writes no cache."""
    kept, _ = resolve_index_files(profile.resolved_root(), profile.selection_rules, None)
    paths = sample_files(kept, sample, seed=seed)
    cl = llm_client.make_client()
    load_email = _make_load_email(body_chars)

    def judge(email: Dict[str, Any]) -> Dict[str, Any]:
        return summary.parse_response(
            llm_client.chat(cl, model,
                            rubrics.build_prompt(profile.rubric, email, body_chars)))

    records = judge_sample(paths, load_email, judge, workers=workers, progress=progress)
    return calibration.CalibrationReport(
        rubric=profile.rubric,
        sample=len(records),
        noise_rate=calibration.noise_rate(records),
        false_noise=calibration.false_noise(records),
        false_keep=calibration.false_keep(records),
    )
