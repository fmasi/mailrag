# scripts/eval/select_spike_slice.py
"""Select the spike slice: every gold thread + a random distractor sample (#11).

Maps thread_id -> .eml paths by loading the cleaned corpus once, then writes the
union of (gold threads from the eval queries) + (random distractor threads) capped
at --max-emails to a newline path list consumed by build_local_eml_rag --only-files
and gen_thread_summaries --slice. Real content -> eval/out (gitignored).

Run on the HOST (rag env):
  conda run -n rag --no-capture-output python scripts/eval/select_spike_slice.py \
    --queries eval/out/queries.jsonl --distractor-threads 900 --max-emails 2000 \
    --out eval/out/spike_slice.txt | tee eval/out/spike_slice.log
"""
import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def run(queries_path, selection_path, blacklist, distractor_threads, max_emails, seed, out_path):
    from src.ingest.local_source import resolve_index_files
    from src.data.loaders.mail_archive_x import MailArchiveXLoader
    from src.data.threading import compute_thread_id

    gold_tids = {json.loads(l)["thread_id"] for l in open(queries_path) if l.strip()}
    print(f"gold threads: {len(gold_tids)}", flush=True)

    sel = json.load(open(selection_path))
    kept, _ = resolve_index_files(sel["root"], sel["selection_rules"], blacklist)
    print(f"corpus files: {len(kept)} — loading to map thread_id->paths (slow)...", flush=True)
    emails = MailArchiveXLoader(eml_files=kept).load()

    by_thread = {}
    for e in emails:
        # NormalizedEmail has no thread_id field; derive it from the RFC 5322
        # headers exactly as NormalizedEmail.to_document() does.
        tid = compute_thread_id(
            e.message_id or "", e.in_reply_to or "", e.references or ""
        )
        if not tid:
            continue
        by_thread.setdefault(tid, []).append(e.source_id)

    gold_present = [t for t in gold_tids if t in by_thread]
    print(f"gold threads found in corpus: {len(gold_present)}/{len(gold_tids)}", flush=True)

    rng = random.Random(seed)
    distractor_pool = [t for t in by_thread if t not in gold_tids]
    rng.shuffle(distractor_pool)

    chosen = list(gold_present)
    paths = []
    for t in chosen:
        paths.extend(by_thread[t])
    for t in distractor_pool[:distractor_threads]:
        if len(paths) >= max_emails:
            break
        chosen.append(t)
        paths.extend(by_thread[t])

    paths = paths[:max_emails]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        fh.write("\n".join(paths) + "\n")
    print(f"slice: {len(chosen)} threads, {len(paths)} emails -> {out_path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", default="eval/out/queries.jsonl")
    ap.add_argument("--selection", default=os.path.expanduser("~/rag_eml.selection.json"))
    ap.add_argument("--blacklist", default=os.path.expanduser("~/rag_pass2/work-rag.blacklist"))
    ap.add_argument("--distractor-threads", type=int, default=900)
    ap.add_argument("--max-emails", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="eval/out/spike_slice.txt")
    args = ap.parse_args()
    run(args.queries, args.selection, args.blacklist, args.distractor_threads,
        args.max_emails, args.seed, args.out)


if __name__ == "__main__":
    main()
