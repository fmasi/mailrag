#!/usr/bin/env python3
"""LLM Pass-2 CLI: summarize+judge emails with local Gemma, then report/apply.

Subcommands:
  run     heavy resumable sweep -> populate the SQLite cache
  report  dry-run: counts + samples from the cache (no mutation)
  apply   append noise file-hashes (>= threshold) to the blacklist
  eval    dev-only: compare local Gemma vs a reference model (added later)
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load .env if present. find_dotenv walks UP the tree, so running this from a
# git worktree (which lives under the main checkout's .claude/worktrees/) still
# picks up the main checkout's gitignored .env — no need to copy it per worktree.
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # dotenv optional; env can still be exported manually
    pass

from src.data.blacklist import append_to_blacklist  # noqa: E402
from src.llm import client as llm_client  # noqa: E402
from src.llm import pass2, summary  # noqa: E402
from src.llm.cache import Pass2Cache  # noqa: E402


def _load_selection(selection_path):
    import json
    with open(selection_path) as fh:
        return json.load(fh)


def _make_load_email(body_chars):
    """Return a load_email(path) -> dict using the single-file MailArchiveXLoader."""
    from src.data.loaders.mail_archive_x import MailArchiveXLoader

    def load_email(path):
        # The loader prints "Loading… Found 1… Loaded 1" per file; silence it so
        # the progress bar stays clean.
        import contextlib, io
        with contextlib.redirect_stdout(io.StringIO()):
            emails = list(MailArchiveXLoader(eml_files=[path]).load())
        if not emails:
            raise ValueError("no email parsed")
        e = emails[0]
        return {"sender": e.sender, "subject": e.subject,
                "date": e.date.isoformat() if e.date else "unknown",
                "body": e.body, "message_id": e.message_id or ""}
    return load_email


def cmd_run(args):
    from src.ingest.local_source import resolve_index_files
    sel = _load_selection(args.selection)
    kept, _ = resolve_index_files(sel["root"], sel["selection_rules"], args.blacklist)
    kept = pass2.sample_files(kept, args.sample, args.seed)
    cache = Pass2Cache(args.cache)
    cl = llm_client.make_client()
    model = args.model or llm_client.default_model()
    if not model:
        print("Error: set --model or RAG_LLM_MODEL"); sys.exit(1)
    load_email = _make_load_email(args.body_chars)

    def summarize(email):
        return summary.parse_response(
            llm_client.chat(cl, model, summary.build_prompt(email, args.body_chars)))

    print(f"sweeping {len(kept)} file(s) with {model} "
          f"({cache.stats()['total']} already cached)", flush=True)
    counts = pass2.run_pass(kept, cache, load_email, summarize, model,
                            limit=args.limit, progress=not args.no_progress,
                            workers=args.workers)
    print(f"run: {counts}; cache now {cache.stats()}")
    cache.close()


def cmd_report(args):
    cache = Pass2Cache(args.cache)
    stats = cache.stats()
    print(f"cache: {stats}")
    print(f"\nnoise candidates >= conf {args.min_confidence}:")
    shown = 0
    for row in cache.iter_noise(args.min_confidence):
        print(f"  [{row['confidence']:.2f}] {row['reason']}")
        shown += 1
        if shown >= args.samples:
            break
    print(f"\nsample kept summaries:")
    shown = 0
    for row in cache.iter_kept():
        if row["summary"]:
            print(f"  - {row['summary']}")
            shown += 1
            if shown >= args.samples:
                break
    cache.close()


def cmd_apply(args):
    cache = Pass2Cache(args.cache)
    shas = pass2.noise_hashes(cache, args.min_confidence)
    cache.close()
    if args.dry_run:
        print(f"would blacklist {len(shas)} file(s) at conf >= {args.min_confidence}")
        return
    added = append_to_blacklist(args.blacklist, shas)
    print(f"blacklisted {added} new file-hash(es) (of {len(shas)} candidates) "
          f"-> {args.blacklist}; rebuild to apply")


def cmd_eval(args):
    """DEV-ONLY: compare local Gemma vs a cloud reference on a sample.

    Offline production paths never call this. Requires ANTHROPIC_API_KEY.
    """
    import random
    from src.ingest.local_source import resolve_index_files
    sel = _load_selection(args.selection)
    kept, _ = resolve_index_files(sel["root"], sel["selection_rules"], args.blacklist)
    random.seed(args.seed)
    sample = random.sample(kept, min(args.sample, len(kept)))

    load_email = _make_load_email(args.body_chars)
    cl = llm_client.make_client()
    model = args.model or llm_client.default_model()
    if not model:
        print("Error: set --model or RAG_LLM_MODEL"); sys.exit(1)

    import anthropic
    ref = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY

    def ref_summarize(email):
        msg = ref.messages.create(
            model=args.reference_model, max_tokens=512,
            messages=[{"role": "user", "content": summary.build_prompt(email, args.body_chars)}])
        return summary.parse_response(msg.content[0].text)

    local_j, ref_j = {}, {}
    for path in sample:
        email = load_email(path)
        try:
            lr = summary.parse_response(
                llm_client.chat(cl, model, summary.build_prompt(email, args.body_chars)))
            rr = ref_summarize(email)
        except Exception as exc:
            print(f"  skip {path}: {exc}"); continue
        local_j[path], ref_j[path] = lr["is_noise"], rr["is_noise"]
        flag = "" if lr["is_noise"] == rr["is_noise"] else "  <-- DISAGREE"
        print(f"\n{os.path.basename(path)}{flag}")
        print(f"  gemma : noise={lr['is_noise']} ({lr['confidence']:.2f}) {lr['summary'][:80]}")
        print(f"  ref   : noise={rr['is_noise']} ({rr['confidence']:.2f}) {rr['summary'][:80]}")

    print(f"\nagreement on is_noise: {pass2.agreement_rate(local_j, ref_j):.0%} "
          f"over {len(local_j)} email(s)")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="llm_pass2.py")
    sub = ap.add_subparsers(dest="command", required=True)

    pr = sub.add_parser("run", help="Heavy resumable summarize+judge sweep")
    pr.add_argument("--selection", required=True, help="selection JSON from select_local_eml.py")
    pr.add_argument("--cache", required=True, help="SQLite cache path")
    pr.add_argument("--blacklist", default=None)
    pr.add_argument("--model", default=None)
    pr.add_argument("--body-chars", type=int, default=4000)
    pr.add_argument("--limit", type=int, default=None,
                    help="process the first N resolved files (in order)")
    pr.add_argument("--sample", type=int, default=None,
                    help="randomly sample N of the resolved files (representative spot-check)")
    pr.add_argument("--seed", type=int, default=0, help="random seed for --sample")
    pr.add_argument("--no-progress", action="store_true",
                    help="disable the tqdm progress bar")
    pr.add_argument("--workers", type=int, default=1,
                    help="parallel in-flight LLM requests (cache writes stay serial)")
    pr.set_defaults(func=cmd_run)

    rp = sub.add_parser("report", help="Dry-run report from the cache")
    rp.add_argument("--cache", required=True)
    rp.add_argument("--min-confidence", type=float, default=0.7)
    rp.add_argument("--samples", type=int, default=20)
    rp.set_defaults(func=cmd_report)

    ap_ = sub.add_parser("apply", help="Append noise hashes to the blacklist")
    ap_.add_argument("--cache", required=True)
    ap_.add_argument("--blacklist", required=True)
    ap_.add_argument("--min-confidence", type=float, default=0.7)
    ap_.add_argument("--dry-run", action="store_true")
    ap_.set_defaults(func=cmd_apply)

    ev = sub.add_parser("eval", help="DEV-ONLY: compare local Gemma vs a reference model")
    ev.add_argument("--selection", required=True)
    ev.add_argument("--blacklist", default=None)
    ev.add_argument("--model", default=None)
    ev.add_argument("--reference-model", default="claude-opus-4-7")
    ev.add_argument("--sample", type=int, default=20)
    ev.add_argument("--body-chars", type=int, default=4000)
    ev.add_argument("--seed", type=int, default=0)
    ev.set_defaults(func=cmd_eval)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
