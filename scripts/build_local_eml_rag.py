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
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _percentiles(values, ps=(50, 90, 95, 99, 100)):
    s = sorted(values)
    return {p: s[min(len(s) - 1, int(len(s) * p / 100))] for p in ps}


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
    ap.add_argument("--profile", action="store_true", help="report cleaned body token lengths + suggest chunk_size, then exit")
    ap.add_argument("--recreate", action="store_true")
    args = ap.parse_args(argv)

    from src.ingest.local_source import resolve_index_files
    from src.data.loaders.mail_archive_x import MailArchiveXLoader

    sel = json.load(open(args.selection))
    kept, skipped = resolve_index_files(sel["root"], sel["selection_rules"], args.blacklist)
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
        lens = [len(encode_len(d.text)) for d in docs if d.text.strip()]
        pct = _percentiles(lens)
        print("\ncleaned body token-length distribution (bge-m3 tokens):")
        for p, v in pct.items():
            print(f"  p{p:<3}: {v}")
        print(f"  mean: {statistics.mean(lens):.0f}  | bodies: {len(lens)}")
        # round p90 up to nearest 64, clamp to [256, 1024]
        import math
        suggested = min(1024, max(256, int(math.ceil(pct[90] / 64) * 64)))
        split = sum(1 for L in lens if L > suggested)
        print(f"\nsuggested chunk_size = {suggested} (p90 rounded to /64); "
              f"{split} bodies ({100*split/len(lens):.1f}%) would still split")
        return 0

    # ---- build mode ----
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.core.schema import MetadataMode

    splitter = SentenceSplitter(
        chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap, tokenizer=encode_len
    )
    nodes = splitter.get_nodes_from_documents(docs, show_progress=False)
    print(f"{len(nodes)} chunks before dedup")

    from src.data.dedup import dedup_by_content
    nodes = dedup_by_content(nodes, key=lambda n: n.get_content(metadata_mode=MetadataMode.NONE))
    print(f"{len(nodes)} chunks after dedup")
    if not nodes:
        print("nothing to index")
        return 0

    from src.ingest.embedder import BgeM3Embedder
    from src.ingest.sparse import lexical_weights_to_sparse
    from src.ingest import hybrid_qdrant as hq

    client = hq.get_client(args.qdrant_url)
    hq.ensure_hybrid_collection(client, args.collection, dim=1024, recreate=args.recreate)
    embedder = BgeM3Embedder(device="mps", use_fp16=True)

    total = len(nodes)
    done = 0
    t_start = time.time()
    for i in range(0, total, args.upsert_batch):
        batch = nodes[i : i + args.upsert_batch]
        embed_texts = [n.get_content(metadata_mode=MetadataMode.EMBED) for n in batch]
        dense, sparse = embedder.encode(embed_texts, batch_size=args.embed_batch, max_length=args.chunk_size)
        points = []
        for n, dv, lw in zip(batch, dense, sparse):
            idx, val = lexical_weights_to_sparse(lw)
            payload = dict(n.metadata)
            payload["text"] = n.get_content(metadata_mode=MetadataMode.NONE)
            points.append(hq.make_point(n.node_id, dv, idx, val, payload))
        hq.upsert(client, args.collection, points)
        done += len(batch)
        rate = done / (time.time() - t_start)
        print(f"  upserted {done}/{total}  ({rate:.0f} chunks/s)")

    print(f"DONE: {done} chunks -> '{args.collection}' in {time.time()-t_start:.1f}s")
    info = client.get_collection(args.collection)
    print(f"collection points_count={info.points_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
