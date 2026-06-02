#!/usr/bin/env python3
"""Build a hybrid (dense+sparse) Qdrant collection from a local .eml selection,
embedding with bge-m3 via FlagEmbedding (MPS).

Pipeline:
  selection JSON -> resolve_index_files (selection minus blacklist)
    -> MailArchiveXLoader(eml_files)        [captures threading headers,
                                             collapses calendar invites]
    -> to_document -> SentenceSplitter (bge-m3-tokenizer-aligned)
    -> exact-text chunk dedup
    -> bge-m3 dense + sparse embed
    -> upsert (named vectors: dense, sparse) to Qdrant.

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
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selection", default=os.path.expanduser("~/rag_eml.selection.json"))
    ap.add_argument("--blacklist", default=None)
    ap.add_argument("--summary-cache", default=None,
                    help="Path to the LLM Pass-2 SQLite cache; injects summaries into payload")
    ap.add_argument("--collection", default="email-rag")
    ap.add_argument("--qdrant-url", default="http://localhost:6333")
    ap.add_argument("--chunk-size", type=int, default=512)
    ap.add_argument("--chunk-overlap", type=int, default=64)
    ap.add_argument("--embed-batch", type=int, default=32)
    ap.add_argument("--upsert-batch", type=int, default=256)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--only-files", default=None,
                    help="path to a newline list of .eml paths; restrict the build "
                         "to their intersection with the selection (spike slice)")
    ap.add_argument("--profile", action="store_true", help="report cleaned body token lengths + suggest chunk_size, then exit")
    ap.add_argument("--recreate", action="store_true")
    ap.add_argument("--embed-summary", action="store_true",
                    help="contextual retrieval: prepend each email's Pass-2 summary "
                         "to the chunk text before embedding (needs --summary-cache)")
    ap.add_argument("--embed-max-length", type=int, default=None,
                    help="token ceiling for embedding; default = chunk_size "
                         "(body-only) or chunk_size + summary headroom when --embed-summary "
                         "(so the summary augments, not displaces, the body chunk)")
    args = ap.parse_args(argv)

    from src.ingest.local_source import resolve_index_files
    from src.data.loaders.mail_archive_x import MailArchiveXLoader

    sel = json.load(open(args.selection))
    kept, skipped = resolve_index_files(sel["root"], sel["selection_rules"], args.blacklist)
    if args.only_files:
        only = {l.strip() for l in open(args.only_files) if l.strip()}
        kept = [p for p in kept if p in only]
        print(f"--only-files: restricted to {len(kept)} of {len(only)} slice paths")
    if args.limit:
        kept = kept[: args.limit]
    print(f"selected {len(kept)} file(s); {len(skipped)} blacklisted")

    emails = MailArchiveXLoader(eml_files=kept).load()

    # Pass 1 noise filter (cheap, pre-embed): drop obvious junk (LinkedIn,
    # newsletters, ...) by the project's noise_rules.yaml before embedding.
    from src.data.noise_filter import NoiseFilter

    nf = NoiseFilter.from_project_rules()
    n_before = len(emails)
    emails = [e for e in emails if not nf.is_noise(e)]
    print(f"noise filter: kept {len(emails)}/{n_before} "
          f"(dropped {n_before - len(emails)}; categories: {nf.category_names()})")

    if args.summary_cache:
        from src.llm.cache import Pass2Cache
        from src.llm.pass2 import inject_summaries
        _cache = Pass2Cache(args.summary_cache)
        n_sum = inject_summaries(emails, _cache)
        _cache.close()
        print(f"injected {n_sum} summary/summaries from {args.summary_cache}")
    docs = [e.to_document(doc_id=f"{e.source}_{i}") for i, e in enumerate(emails)]
    print(f"{len(docs)} email(s) to index")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("BAAI/bge-m3")
    encode_len = lambda text: tok.encode(text, add_special_tokens=False)

    # ---- profile mode: choose chunk_size from cleaned body token lengths ----
    if args.profile:
        from src.ingest.profile import percentiles, suggest_chunk_size
        lens = [len(encode_len(d.text)) for d in docs if d.text.strip()]
        pct = percentiles(lens)
        print("\ncleaned body token-length distribution (bge-m3 tokens):")
        for p, v in pct.items():
            print(f"  p{p:<3}: {v}")
        print(f"  mean: {statistics.mean(lens):.0f}  | bodies: {len(lens)}")
        suggested = suggest_chunk_size(lens)
        split = sum(1 for L in lens if L > suggested)
        print(f"\nsuggested chunk_size = {suggested} (p90 rounded to /64); "
              f"{split} bodies ({100*split/len(lens):.1f}%) would still split")
        return 0

    # ---- build mode ----
    from src.ingest.embedder import BgeM3Embedder
    from src.indexing.contextual_index import build_contextual_index

    if args.embed_summary:
        print("contextual retrieval ON: prepending Pass-2 summaries to embedded text")

    # summaries=None: inject_summaries() already set e.summary on each email;
    # to_document() surfaces that as metadata["summary"], and build_contextual_index
    # reads it from there. Passing summaries=None avoids a second injection pass.
    embedder = BgeM3Embedder(device="mps", use_fp16=True)
    res = build_contextual_index(
        emails,
        collection=args.collection,
        embedder=embedder,
        summaries=None,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        embed_summary=args.embed_summary,
        embed_max_length_override=args.embed_max_length,
        embed_batch=args.embed_batch,
        upsert_batch=args.upsert_batch,
        recreate=args.recreate,
        qdrant_url=args.qdrant_url,
    )

    print(f"DONE: {res.chunks} chunks -> '{res.collection}'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
