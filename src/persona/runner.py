"""Build the verb->handler map that executes a persona recipe in-process.

Both the headless ``run`` verb and the interactive TUI call :func:`build_handlers`
and hand the result to :func:`src.persona.executor.run_persona`, so the wiring
from recipe steps to the existing pipeline stages lives in one place.

Only the implemented verbs are returned. ``judge`` and ``prune`` (the
drop-before-LLM savings path) are intentionally absent until the engine ships, so
:func:`src.persona.executor.missing_handlers` reports them and personas that need
them (``llm-verify``) fail friendly rather than half-running. See docs/VERBS.md.
"""

from __future__ import annotations

import datetime
from typing import Any, Callable, Dict, Optional


def build_handlers(
    *,
    profile_path: str,
    model: Optional[str] = None,
    workers: int = 1,
    embedder: Any = None,
    prune_confirm: Optional[Callable] = None,
    limit: Optional[int] = None,
) -> Dict[str, Callable]:
    """Return ``{verb: handler(profile, **params)}`` for the implemented verbs.

    Heavy imports/objects (loaders, the BGE-M3 embedder) are created lazily inside
    the handlers, so constructing the map is cheap and side-effect-free.
    ``prune_confirm`` (preview -> bool) gates prune's blacklist write; defaults to
    auto-yes (headless). The wizard passes an interactive confirm.
    ``limit`` caps the corpus the scan/summarize/index steps touch, for a fast
    end-to-end test run instead of a full (multi-hour) rebuild."""
    from src.pipeline import build as build_stage
    from src.pipeline import calibrate as calibrate_stage
    from src.pipeline import explore as explore_stage
    from src.pipeline import judge as judge_stage
    from src.pipeline import pass1
    from src.pipeline import pass2 as pass2_stage
    from src.pipeline import profile as profile_stage
    from src.pipeline import prune as prune_stage
    from src.pipeline import select as select_stage

    _confirm = prune_confirm or (lambda preview: True)
    _scan_json = profile_path.rsplit(".", 1)[0] + ".scan.json"

    def _require_model() -> str:
        """The model id, or a clear error if an LLM step was reached without one.

        ``model`` is optional because non-LLM personas do not need one, but the
        LLM steps below cannot run without it. Both callers (the wizard and the
        TUI) already prompt for a model before an LLM verb — this makes that
        invariant explicit at the point of use, so a caller that forgets fails
        here with an actionable message instead of somewhere inside the LLM
        client with an opaque one.
        """
        if not model:
            raise ValueError(
                "this step needs an LLM model: pass model= to build_handlers, "
                "use --model, or set RAG_LLM_MODEL"
            )
        return model

    def _default_blacklist(prof):
        if not getattr(prof, "blacklist", None):
            prof.blacklist = profile_path.rsplit(".", 1)[0] + ".blacklist.txt"
        return prof.blacklist

    def _scope(prof, **_):
        return select_stage.run(prof)

    def _measure(prof, **_):
        return profile_stage.run(prof, set_profile=True)

    def _tag(prof, **_):
        from src.data.loaders.mail_archive_x import MailArchiveXLoader
        from src.data.noise_filter import NoiseFilter
        from src.ingest.local_source import resolve_index_files

        kept, _drop = resolve_index_files(prof.resolved_root(), prof.selection_rules, None)
        emails = MailArchiveXLoader(eml_files=kept).load()
        _, stats = pass1.run(emails, NoiseFilter.from_project_rules())
        return stats

    def _scan(prof, **_):
        out = profile_path.rsplit(".", 1)[0] + ".scan.json"
        return explore_stage.run(prof, json_path=out, profile_path=profile_path, limit=limit)

    def _calibrate(prof, **_):
        report = calibrate_stage.run(prof, model=_require_model(), workers=workers, progress=True)
        prof.calibration = {
            "rubric": report.rubric,
            "passed": True,
            "noise_rate": report.noise_rate,
            "sample": report.sample,
            "false_noise": len(report.false_noise),
            "false_keep": len(report.false_keep),
            "at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        return report

    def _summarize(prof, target="all", **_):
        return pass2_stage.run(prof, model=_require_model(), workers=workers, limit=limit)

    def _index(prof, embed="summary", **_):
        from src.ingest.embedder import BgeM3Embedder

        emb = embedder or BgeM3Embedder(device="mps", use_fp16=True)
        return build_stage.run(prof, embedder=emb, embed_summary=(embed == "summary"), limit=limit)

    def _judge(prof, min_score=0.6, **_):
        return judge_stage.run(
            prof, model=_require_model(), scan_json=_scan_json, min_score=min_score, workers=workers
        )

    def _prune(prof, **params):
        _default_blacklist(prof)
        source = params.get("from", "judge")
        return prune_stage.run(prof, source=source, confirm=_confirm)

    return {
        "scope": _scope,
        "measure": _measure,
        "tag": _tag,
        "scan": _scan,
        "calibrate": _calibrate,
        "summarize": _summarize,
        "index": _index,
        "judge": _judge,
        "prune": _prune,
    }
