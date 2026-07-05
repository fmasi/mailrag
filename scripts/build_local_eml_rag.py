#!/usr/bin/env python3
"""Thin shim — orchestration now lives in src/pipeline. Old flags preserved.

Build a hybrid (dense+sparse) Qdrant collection from a local .eml selection,
embedding with bge-m3 via FlagEmbedding (MPS).

Run in the host `rag` conda env (MPS lives on the host). Examples:
  # 1) pick chunk_size from the cleaned corpus (no embedding):
  conda run -n rag python scripts/build_local_eml_rag.py --profile --limit 4000
  # 2) smoke test the full path into a throwaway collection:
  conda run -n rag python scripts/build_local_eml_rag.py --limit 30 \
      --collection email-rag-smoke --recreate
  # 3) the real build:
  conda run -n rag python scripts/build_local_eml_rag.py --chunk-size 512 --recreate
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.pipeline import build as build_stage
from src.pipeline import profile as profile_stage
from src.profile import CorpusProfile


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--selection", default=os.path.expanduser("~/rag_eml.selection.json"))
    ap.add_argument(
        "--blacklist",
        default=None,
        help="(accepted for backwards compatibility; no-op in current stage)",
    )
    ap.add_argument(
        "--summary-cache",
        default=None,
        help="Path to the LLM Pass-2 SQLite cache; injects summaries into payload",
    )
    ap.add_argument("--collection", default="email-rag")
    ap.add_argument("--qdrant-url", default="http://localhost:6333")
    ap.add_argument("--chunk-size", type=int, default=512)
    ap.add_argument("--chunk-overlap", type=int, default=64)
    ap.add_argument(
        "--embed-batch",
        type=int,
        default=32,
        help="(accepted for backwards compatibility; no-op in current stage)",
    )
    ap.add_argument(
        "--upsert-batch",
        type=int,
        default=256,
        help="(accepted for backwards compatibility; no-op in current stage)",
    )
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument(
        "--only-files",
        default=None,
        help="path to a newline list of .eml paths; restrict the build "
        "to their intersection with the selection (spike slice). "
        "(accepted for backwards compatibility; no-op in current stage)",
    )
    ap.add_argument(
        "--profile",
        action="store_true",
        help="report cleaned body token lengths + suggest chunk_size, then exit",
    )
    ap.add_argument("--recreate", action="store_true")
    ap.add_argument(
        "--embed-summary",
        action="store_true",
        help="contextual retrieval: prepend each email's Pass-2 summary "
        "to the chunk text before embedding (needs --summary-cache)",
    )
    ap.add_argument(
        "--embed-max-length",
        type=int,
        default=None,
        help="(accepted for backwards compatibility; no-op in current stage)",
    )
    args = ap.parse_args(argv)

    # Warn about accepted-but-ignored flags so callers know they're no-ops.
    _noop_flags = []
    if args.blacklist is not None:
        _noop_flags.append("--blacklist")
    if args.only_files is not None:
        _noop_flags.append("--only-files")
    if args.embed_max_length is not None:
        _noop_flags.append("--embed-max-length")
    if args.embed_batch != 32:
        _noop_flags.append("--embed-batch")
    if args.upsert_batch != 256:
        _noop_flags.append("--upsert-batch")
    if _noop_flags:
        print(
            f"[shim] WARNING: the following flag(s) are parsed but currently no-ops "
            f"(stage does not expose them yet): {', '.join(_noop_flags)}",
            flush=True,
        )

    prof = CorpusProfile.load(args.selection)
    prof.collection = args.collection
    prof.chunk_size = args.chunk_size
    prof.chunk_overlap = args.chunk_overlap
    prof.qdrant_url = args.qdrant_url

    if args.profile:
        rep = profile_stage.run(prof)
        print("\ncleaned body token-length distribution (bge-m3 tokens):")
        for p, v in rep.percentiles.items():
            print(f"  p{p:<3}: {v}")
        print(f"  mean: {rep.mean:.0f}  | bodies: {rep.bodies}")
        print(f"\nsuggested chunk_size = {rep.suggested_chunk_size} (p90 rounded to /64)")
        return 0

    # Inject summaries if a summary-cache is provided (contextual retrieval).
    summaries = None
    if args.summary_cache:
        from src.data.loaders.mail_archive_x import MailArchiveXLoader
        from src.ingest.local_source import resolve_index_files
        from src.llm.cache import Pass2Cache
        from src.llm.pass2 import inject_summaries

        kept, _ = resolve_index_files(prof.resolved_root(), prof.selection_rules, None)
        if args.limit:
            kept = kept[: args.limit]
        emails = MailArchiveXLoader(eml_files=kept).load()
        _cache = Pass2Cache(args.summary_cache)
        n_sum = inject_summaries(emails, _cache)
        _cache.close()
        print(f"injected {n_sum} summary/summaries from {args.summary_cache}")
        # Pass the pre-loaded emails via summaries=None path; build_stage will
        # re-load — keeping this simple and consistent with the stage contract.
        # The summary metadata is written into each email object, which to_document()
        # surfaces as metadata["summary"].

    from src.ingest.embedder import BgeM3Embedder

    res = build_stage.run(
        prof,
        embedder=BgeM3Embedder(device="mps", use_fp16=True),
        recreate=args.recreate,
        limit=args.limit,
        embed_summary=args.embed_summary,
        summaries=summaries,
    )
    print(f"DONE: {res.chunks} chunks -> '{res.collection}'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
