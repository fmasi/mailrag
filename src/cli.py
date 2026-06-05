"""`mailrag` command-line entry: `onboard`, `ask`, and pipeline subcommands.

Run as `python -m src.cli ...` (the repo-root `mailrag` shim wraps this). Poetry
stays package-mode=false, so no console_scripts are installed.

Verbs form a cost-ordered ladder (see docs/VERBS.md):
    scope · measure · tag · scan · judge · calibrate · summarize · prune · index · ask
Older names (select/profile/pass1/explore/pass2/build/query) remain as hidden
aliases for one release.
"""
import argparse
import datetime
import sys

from dotenv import load_dotenv

from src.profile import CorpusProfile
from src.ingest.local_source import resolve_index_files
from src.data.loaders.mail_archive_x import MailArchiveXLoader
from src.data.noise_filter import NoiseFilter
from src.pipeline import pass1, build as build_stage, profile as profile_stage
from src.pipeline import pass2 as pass2_stage, select as select_stage
from src.pipeline import calibrate as calibrate_stage
from src.pipeline import explore as explore_stage
from src.persona import registry as persona_registry
from src.persona import executor as persona_executor
from src.persona import runner as persona_runner
from src.llm import calibration as calibration_lib
from src.ingest.embedder import BgeM3Embedder  # module-level so tests patch src.cli.BgeM3Embedder


def _add_profile_arg(p):
    p.add_argument("--profile", required=True, help="path to the corpus profile JSON")


def _cmd_tag(args):
    prof = CorpusProfile.load(args.profile)
    kept, _ = resolve_index_files(prof.resolved_root(), prof.selection_rules, None)
    emails = MailArchiveXLoader(eml_files=kept).load()
    _, stats = pass1.run(emails, NoiseFilter.from_project_rules())
    print(f"tag: dropped {stats.dropped}; kept {stats.kept}; tagged {stats.tagged}")
    return 0


def _cmd_index(args):
    prof = CorpusProfile.load(args.profile)
    res = build_stage.run(prof, embedder=BgeM3Embedder(device="mps", use_fp16=True),
                          recreate=args.recreate, limit=args.limit,
                          embed_summary=args.embed_summary,
                          noise_min_confidence=args.noise_confidence)
    prof.save(args.profile)
    print(f"DONE: {res.chunks} chunks -> '{res.collection}'")
    return 0


def _cmd_measure(args):
    prof = CorpusProfile.load(args.profile)
    rep = profile_stage.run(prof, set_profile=True)
    prof.save(args.profile)
    print(f"suggested chunk_size = {rep.suggested_chunk_size} (bodies {rep.bodies})")
    return 0


def _cmd_summarize(args):
    prof = CorpusProfile.load(args.profile)
    if not prof.rubric:
        raise ValueError("profile has no rubric set")
    cal = prof.calibration
    calibrated = bool(cal) and cal.get("rubric") == prof.rubric and cal.get("passed")
    if not args.force and not calibrated:
        print(f"error: rubric '{prof.rubric}' is not calibrated; run "
              f"`mailrag calibrate --profile {args.profile} --model {args.model}` "
              f"first (or pass --force)", file=sys.stderr)
        return 2
    counts = pass2_stage.run(prof, model=args.model, workers=args.workers)
    print(f"summarize: {counts}")
    return 0


def _cmd_calibrate(args):
    prof = CorpusProfile.load(args.profile)
    if not prof.rubric:
        raise ValueError("profile has no rubric set")
    report = calibrate_stage.run(prof, model=args.model, sample=args.sample,
                                 seed=args.seed, workers=args.workers, progress=True)
    print(calibration_lib.format_report(report))
    # passed=True records only that calibration was RUN for this rubric (the gate is
    # a forcing-function to make the human read the buckets above, not an automatic
    # quality verdict). The human reads false-noise/false-keep and decides; --force
    # skips the gate entirely.
    if prof.calibration:
        print(f"  (overwriting previous calibration for rubric "
              f"'{prof.calibration.get('rubric')}')")
    prof.calibration = {
        "rubric": report.rubric, "passed": True, "noise_rate": report.noise_rate,
        "sample": report.sample, "false_noise": len(report.false_noise),
        "false_keep": len(report.false_keep),
        "at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    prof.save(args.profile)
    print(f"\ncalibration recorded for rubric '{report.rubric}' -> {args.profile}")
    return 0


def _cmd_scope(args):
    prof = CorpusProfile.load(args.profile)
    select_stage.run(prof)
    prof.save(args.profile)
    print(f"selected {len(prof.selection_rules)} rule(s) -> {args.profile}")
    return 0


def _cmd_scan(args):
    prof = CorpusProfile.load(args.profile)
    default_json = args.profile.rsplit(".", 1)[0] + ".scan.json"
    out = args.json or default_json
    try:
        report = explore_stage.run(
            prof, json_path=out, clusters=args.clusters, seed=args.seed,
            limit=args.limit, force_fresh=args.fresh, qdrant_url=args.qdrant_url,
            top=args.top, profile_path=args.profile)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(explore_stage.format_report(report, top=args.top))
    print(f"wrote {out}")
    return 0


_LLM_STEPS = {"calibrate", "summarize", "judge"}


def _cmd_run(args):
    reg = persona_registry.load_registry()
    try:
        persona = reg.get(args.persona)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    needs_model = any(s.verb in _LLM_STEPS for s in persona.steps)
    if needs_model and not args.model:
        print(f"error: persona '{persona.name}' has LLM steps; pass --model",
              file=sys.stderr)
        return 2
    handlers = persona_runner.build_handlers(
        profile_path=args.profile, model=args.model, workers=args.workers)
    missing = persona_executor.missing_handlers(persona, handlers)
    if missing:
        print(f"error: persona '{persona.name}' needs verb(s) not yet implemented: "
              f"{', '.join(missing)} (coming in the persona engine). "
              f"Try 'llm-none' or 'llm-all'.", file=sys.stderr)
        return 2
    prof = CorpusProfile.load(args.profile)
    print(f"running persona '{persona.name}' ({persona.label})")
    persona_executor.run_persona(
        prof, persona, handlers,
        on_step=lambda s: print(f"  ▶ {s.verb}"),
        on_skip=lambda s: print(f"  – skip {s.verb} (optional)"))
    prof.save(args.profile)
    print(f"persona '{persona.name}' complete -> {args.profile}")
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


def _cmd_ask(args):
    from src.onboard import latest_manifest_collection
    prof = CorpusProfile.load(args.profile) if args.profile else None
    collection = (args.collection or (prof.collection if prof else None)
                  or latest_manifest_collection())
    if not collection:
        print("no collection given and no manifest found; run `mailrag onboard` first",
              file=sys.stderr)
        return 2
    # Resolve the Qdrant URL explicitly rather than letting the query path read
    # QDRANT_URL from .env: that var targets the container (host.docker.internal)
    # and is unresolvable on the host (#29). Precedence: flag > profile > localhost.
    qdrant_url = (args.qdrant_url or (prof.qdrant_url if prof else None)
                  or "http://localhost:6333")
    from src.query.hybrid import build_hybrid_searcher
    from src.llm.answer import answer_from_threads
    searcher = build_hybrid_searcher(collection, mode="hybrid", qdrant_url=qdrant_url)
    contexts = searcher.search_threads(args.text)
    print(answer_from_threads(args.text, contexts, k=args.k))
    return 0


def _configure_onboard(p):
    p.add_argument("source")
    p.add_argument("--collection", default=None)
    p.add_argument("--chunk-size", type=int, default=None)
    p.add_argument("--queries", default=None)
    p.add_argument("--no-validate", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--noise-confidence", type=float, default=0.7)
    p.add_argument("--model", default=None)
    p.add_argument("--qdrant-url", default="http://localhost:6333")


def _configure_ask(p):
    p.add_argument("text")
    p.add_argument("--collection", default=None)
    p.add_argument("--profile", default=None,
                   help="corpus profile JSON; supplies collection + qdrant_url")
    p.add_argument("--qdrant-url", default=None,
                   help="override Qdrant URL (default: profile, else http://localhost:6333)")
    p.add_argument("--k", type=int, default=3)


def _configure_index(p):
    _add_profile_arg(p)
    p.add_argument("--recreate", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--embed-summary", action="store_true",
                   help="consume the summarize cache: drop confident noise and embed summaries")
    p.add_argument("--noise-confidence", type=float, default=0.7,
                   help="with --embed-summary, drop noise at/above this confidence")


def _configure_summarize(p):
    _add_profile_arg(p)
    p.add_argument("--model", required=True)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--force", action="store_true",
                   help="run even if the rubric is not calibrated")


def _configure_calibrate(p):
    _add_profile_arg(p)
    p.add_argument("--model", required=True)
    p.add_argument("--sample", type=int, default=200)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--workers", type=int, default=4)


def _configure_scan(p):
    _add_profile_arg(p)
    p.add_argument("--clusters", type=int, default=None,
                   help="number of clusters (default: heuristic from corpus size)")
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--fresh", action="store_true",
                   help="force fresh BGE-M3 embed, skip reusing the Qdrant collection")
    p.add_argument("--qdrant-url", default=None)
    p.add_argument("--json", default=None,
                   help="output path (default: <profile-stem>.scan.json)")
    p.add_argument("--top", type=int, default=15, help="rows printed")


def _configure_run(p):
    _add_profile_arg(p)
    p.add_argument("--persona", required=True,
                   help="persona name (see personas.yaml / docs/VERBS.md)")
    p.add_argument("--model", default=None,
                   help="LLM model for calibrate/summarize steps")
    p.add_argument("--workers", type=int, default=1)


def _add_verb(sub, name, configure, func, *, help, aliases=()):
    """Register a verb plus hidden aliases (old names) sharing one handler."""
    p = sub.add_parser(name, help=help)
    configure(p)
    p.set_defaults(func=func)
    for alias in aliases:
        # No help kwarg -> argparse adds the alias to `choices` (so it still works)
        # but not to the help listing; the subparsers metavar keeps it out of usage.
        ap = sub.add_parser(alias)
        configure(ap)
        ap.set_defaults(func=func)
    return p


def build_parser():
    p = argparse.ArgumentParser(prog="mailrag")
    sub = p.add_subparsers(dest="cmd", required=True, metavar="<verb>")

    _add_verb(sub, "onboard", _configure_onboard, _cmd_onboard,
              help="build a validated assistant from an .eml dir")
    _add_verb(sub, "ask", _configure_ask, _cmd_ask, aliases=["query"],
              help="ask a question against an indexed collection")
    _add_verb(sub, "scope", _add_profile_arg, _cmd_scope, aliases=["select"],
              help="choose which folders/accounts are in play")
    _add_verb(sub, "measure", _add_profile_arg, _cmd_measure, aliases=["profile"],
              help="measure corpus and suggest chunk_size")
    _add_verb(sub, "tag", _add_profile_arg, _cmd_tag, aliases=["pass1"],
              help="regex/header tag of obvious bulk noise (no LLM)")
    _add_verb(sub, "scan", _configure_scan, _cmd_scan, aliases=["explore"],
              help="cluster embeddings to surface noise pockets (no LLM)")
    _add_verb(sub, "calibrate", _configure_calibrate, _cmd_calibrate,
              help="judge a sample with the rubric and bucket mistakes")
    _add_verb(sub, "summarize", _configure_summarize, _cmd_summarize, aliases=["pass2"],
              help="LLM summary + noise judgement over a corpus")
    _add_verb(sub, "index", _configure_index, _cmd_index, aliases=["build"],
              help="embed and index a corpus from a profile")
    _add_verb(sub, "run", _configure_run, _cmd_run,
              help="run a persona recipe end-to-end")
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
