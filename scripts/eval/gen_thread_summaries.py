"""Generate summaries over the spike slice into a Pass2Cache (#11/#13).

--mode thread   : summarize each email with its PRECEDING thread context (the method)
--mode isolated : summarize each email alone (control / today's prompt, new model)

Walks the slice's threads chronologically (append-only, = the live-ingest order),
calls RAG_SUMMARY_MODEL per email, and writes {summary,is_noise,...} keyed by file
sha256 (+ message_id/content_sha256) so build_local_eml_rag --summary-cache injects
it. Emits a tqdm progress bar. Real content -> ~/rag_pass2 + eval/out (gitignored).

Run on the HOST (rag env; LM Studio loaded with the summary model):
  RAG_LLM_BASE_URL=http://localhost:1234/v1 RAG_SUMMARY_MODEL=<model> \\
    conda run -n rag --no-capture-output python scripts/eval/gen_thread_summaries.py \\
    --slice eval/out/spike_slice.txt --mode thread --out ~/rag_pass2/spike_A.db \\
    | tee eval/out/gen_thread_summaries.log
"""
import argparse
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from dotenv import load_dotenv; load_dotenv()
except ImportError:
    pass

from src.data.blacklist import file_sha256
from src.data.identity import email_identity
from src.data.threading import compute_thread_id
from src.llm.cache import Pass2Cache
from src.llm.client import make_client, chat, default_model
from src.llm.summary import build_prompt, build_thread_aware_prompt, parse_response


def _as_dict(e):
    """Convert a NormalizedEmail to the dict shape expected by build_prompt /
    build_thread_aware_prompt / _format_preceding (all use .get())."""
    return {
        "sender": getattr(e, "sender", "") or "",
        "date": getattr(e, "date", None),
        "subject": getattr(e, "subject", "") or "",
        "body": getattr(e, "body", "") or "",
        "message_id": getattr(e, "message_id", "") or "",
    }


def _tid(e):
    """Derive the thread_id the same way NormalizedEmail.to_document() does."""
    return compute_thread_id(
        getattr(e, "message_id", "") or "",
        getattr(e, "in_reply_to", "") or "",
        getattr(e, "references", "") or "",
    )


def _dkey(e):
    d = getattr(e, "date", None)
    return d if isinstance(d, datetime) else datetime.min


def run(slice_path, out_path, mode, model):
    from src.data.loaders.mail_archive_x import MailArchiveXLoader

    paths = [l.strip() for l in open(slice_path) if l.strip()]
    emails = MailArchiveXLoader(eml_files=paths).load()

    # Group by thread_id using compute_thread_id (NormalizedEmail has NO thread_id
    # attribute — it is only materialised inside to_document() via metadata).
    by_thread = {}
    for e in emails:
        by_thread.setdefault(_tid(e), []).append(e)
    for tid in by_thread:
        by_thread[tid].sort(key=_dkey)

    client = make_client()
    cache = Pass2Cache(out_path)
    print(
        f"summary model: {model}  mode: {mode}  threads: {len(by_thread)}"
        f"  emails: {len(emails)}",
        flush=True,
    )

    try:
        from tqdm import tqdm
        bar = tqdm(total=len(emails), unit="email", desc=f"summ-{mode}", smoothing=0.05)
    except ImportError:
        bar = None
    counts = {"done": 0, "cached": 0, "error": 0}

    for tid, thread in by_thread.items():
        # preceding is a list of *dicts* — _format_preceding calls .get() on each
        # item, so NormalizedEmail objects would silently return empty strings for
        # every field.  Convert before accumulating.
        preceding = []
        for e in thread:
            sha = file_sha256(e.source_id)
            if cache.has(sha):
                counts["cached"] += 1
            else:
                ed = _as_dict(e)
                prompt = (
                    build_thread_aware_prompt(ed, preceding)
                    if mode == "thread"
                    else build_prompt(ed)
                )
                try:
                    record = parse_response(chat(client, model, prompt))
                    mid, chash = email_identity(
                        sender=ed["sender"],
                        subject=ed["subject"],
                        date=ed["date"],
                        body=ed["body"],
                        message_id=ed["message_id"],
                    )
                    cache.put(
                        sha, record,
                        model=model,
                        message_id=mid,
                        content_sha256=chash,
                    )
                    counts["done"] += 1
                except Exception as exc:  # noqa: BLE001 — keep going on single-email errors
                    print(f"  error on {e.source_id}: {exc}", flush=True)
                    counts["error"] += 1
            # Append the dict (not the NormalizedEmail) so _format_preceding can
            # call .get() safely on it.
            preceding.append(_as_dict(e))
            if bar is not None:
                bar.update(1)
                bar.set_postfix(**counts, refresh=False)

    if bar is not None:
        bar.close()
    print(f"done: {counts} -> {out_path}", flush=True)


def main():
    ap = argparse.ArgumentParser(
        description="Generate Pass-2 summaries over the spike slice (thread-aware or isolated)."
    )
    ap.add_argument("--slice", default="eval/out/spike_slice.txt",
                    help="Newline-delimited list of .eml paths (from select_spike_slice.py)")
    ap.add_argument("--out", required=True,
                    help="Pass2Cache sqlite path (e.g. ~/rag_pass2/spike_A.db)")
    ap.add_argument("--mode", choices=["thread", "isolated"], default="thread",
                    help="thread = build_thread_aware_prompt; isolated = build_prompt (control)")
    args = ap.parse_args()
    model = os.getenv("RAG_SUMMARY_MODEL", "").strip() or default_model()
    run(os.path.expanduser(args.slice), os.path.expanduser(args.out), args.mode, model)


if __name__ == "__main__":
    main()
