"""`mailrag` command-line entry: `onboard` and `query` subcommands.

Run as `python -m src.cli ...` (the repo-root `mailrag` shim wraps this). Poetry
stays package-mode=false, so no console_scripts are installed.
"""
import argparse
import sys


def _cmd_onboard(args):
    from src.onboard import run_onboard
    report = run_onboard(
        args.source, collection=args.collection, chunk_size=args.chunk_size,
        queries_path=args.queries, validate=not args.no_validate, limit=args.limit,
        noise_min_confidence=args.noise_confidence, model=args.model,
        qdrant_url=args.qdrant_url)
    print(report.one_line())
    return 0


def _cmd_query(args):
    from src.onboard import latest_manifest_collection
    collection = args.collection or latest_manifest_collection()
    if not collection:
        print("no collection given and no manifest found; run `mailrag onboard` first",
              file=sys.stderr)
        return 2
    from src.query.hybrid import build_hybrid_searcher
    from src.llm.answer import answer_from_threads
    searcher = build_hybrid_searcher(collection, mode="hybrid")
    contexts = searcher.search_threads(args.text)
    print(answer_from_threads(args.text, contexts, k=args.k))
    return 0


def build_parser():
    p = argparse.ArgumentParser(prog="mailrag")
    sub = p.add_subparsers(dest="cmd", required=True)

    ob = sub.add_parser("onboard", help="build a validated assistant from an .eml dir")
    ob.add_argument("source")
    ob.add_argument("--collection", default=None)
    ob.add_argument("--chunk-size", type=int, default=None)
    ob.add_argument("--queries", default=None)
    ob.add_argument("--no-validate", action="store_true")
    ob.add_argument("--limit", type=int, default=None)
    ob.add_argument("--noise-confidence", type=float, default=0.7)
    ob.add_argument("--model", default=None)
    ob.add_argument("--qdrant-url", default="http://localhost:6333")
    ob.set_defaults(func=_cmd_onboard)

    q = sub.add_parser("query", help="ask a question against an onboarded collection")
    q.add_argument("text")
    q.add_argument("--collection", default=None)
    q.add_argument("--k", type=int, default=3)
    q.set_defaults(func=_cmd_query)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
