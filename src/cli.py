"""`mailrag` command-line entry: `onboard`, `query`, and pipeline subcommands.

Run as `python -m src.cli ...` (the repo-root `mailrag` shim wraps this). Poetry
stays package-mode=false, so no console_scripts are installed.
"""
import argparse
import sys

from dotenv import load_dotenv

from src.profile import CorpusProfile
from src.ingest.local_source import resolve_index_files
from src.data.loaders.mail_archive_x import MailArchiveXLoader
from src.data.noise_filter import NoiseFilter
from src.pipeline import pass1, build as build_stage, profile as profile_stage
from src.pipeline import pass2 as pass2_stage, select as select_stage
from src.ingest.embedder import BgeM3Embedder  # module-level so tests patch src.cli.BgeM3Embedder


def _add_profile_arg(p):
    p.add_argument("--profile", required=True, help="path to the corpus profile JSON")


def _cmd_pass1(args):
    prof = CorpusProfile.load(args.profile)
    kept, _ = resolve_index_files(prof.resolved_root(), prof.selection_rules, None)
    emails = MailArchiveXLoader(eml_files=kept).load()
    _, stats = pass1.run(emails, NoiseFilter.from_project_rules())
    print(f"pass1: dropped {stats.dropped}; kept {stats.kept}; tagged {stats.tagged}")
    return 0


def _cmd_build(args):
    prof = CorpusProfile.load(args.profile)
    res = build_stage.run(prof, embedder=BgeM3Embedder(device="mps", use_fp16=True),
                          recreate=args.recreate, limit=args.limit)
    prof.save(args.profile)
    print(f"DONE: {res.chunks} chunks -> '{res.collection}'")
    return 0


def _cmd_profile(args):
    prof = CorpusProfile.load(args.profile)
    rep = profile_stage.run(prof, set_profile=True)
    prof.save(args.profile)
    print(f"suggested chunk_size = {rep.suggested_chunk_size} (bodies {rep.bodies})")
    return 0


def _cmd_pass2(args):
    prof = CorpusProfile.load(args.profile)
    counts = pass2_stage.run(prof, model=args.model, workers=args.workers)
    print(f"pass2: {counts}")
    return 0


def _cmd_select(args):
    prof = CorpusProfile.load(args.profile)
    select_stage.run(prof)
    prof.save(args.profile)
    print(f"selected {len(prof.selection_rules)} rule(s) -> {args.profile}")
    return 0


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

    sp = sub.add_parser("pass1", help="preview noise partition from a corpus profile")
    _add_profile_arg(sp)
    sp.set_defaults(func=_cmd_pass1)

    sb = sub.add_parser("build", help="embed and index a corpus from a profile")
    _add_profile_arg(sb)
    sb.add_argument("--recreate", action="store_true")
    sb.add_argument("--limit", type=int, default=None)
    sb.set_defaults(func=_cmd_build)

    spr = sub.add_parser("profile", help="measure corpus and suggest chunk_size")
    _add_profile_arg(spr)
    spr.set_defaults(func=_cmd_profile)

    sp2 = sub.add_parser("pass2", help="run LLM noise classification over a corpus")
    _add_profile_arg(sp2)
    sp2.add_argument("--model", required=True)
    sp2.add_argument("--workers", type=int, default=1)
    sp2.set_defaults(func=_cmd_pass2)

    ss = sub.add_parser("select", help="interactively build selection rules for a corpus")
    _add_profile_arg(ss)
    ss.set_defaults(func=_cmd_select)

    return p


def main(argv=None):
    load_dotenv()
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
