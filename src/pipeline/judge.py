"""Judge stage: cheap LLM verdict-only sweep over the `scan` suspects.

Reuses the Pass-2 machinery (resumable cache, resilient identity, thread pool) but
with the rubric's verdict-only `judge_template`, so each call's output is short. Run
ONLY on the scan-flagged suspects (clusters at/above a score cutoff) — the threads we
intend to drop — so `prune` can blacklist them before the expensive summary pass.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

from src.llm import client as llm_client
from src.llm import rubrics, summary
from src.llm.cache import Pass2Cache
from src.llm.pass2 import run_pass
from src.llm.provenance import describe_backend
from src.pipeline.pass2 import _make_load_email


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def select_suspects(scan_json: Dict[str, Any], min_score: float) -> List[str]:
    """Deduped `.eml` paths from clusters scoring at/above *min_score* (in order)."""
    seen: Dict[str, None] = {}
    for cluster in scan_json.get("clusters", []):
        if cluster.get("score", 0.0) < min_score:
            continue
        for member in cluster.get("members", []):
            for path in member.get("paths", []):
                if path and path not in seen:
                    seen[path] = None
    return list(seen)


def run(
    profile,
    *,
    model: str,
    scan_json: str,
    min_score: float = 0.6,
    workers: int = 1,
    body_chars: int = 4000,
    progress: bool = True,
) -> Dict[str, int]:
    if not os.path.exists(scan_json):
        raise ValueError(f"scan JSON not found: {scan_json}; run `mailrag scan` first")
    suspects = select_suspects(_load_json(scan_json), min_score)
    cache = Pass2Cache(profile.pass2_cache)
    cl = llm_client.make_client()
    load_email = _make_load_email(body_chars)

    def judge_fn(email: Dict[str, Any]) -> Dict[str, Any]:
        return summary.parse_response(
            llm_client.chat(
                cl, model, rubrics.build_judge_prompt(profile.rubric, email, body_chars)
            )
        )

    prov = describe_backend(model=model, api_base=getattr(cl, "base_url", ""))
    print(f"judge: {prov.label()}")
    counts = run_pass(
        suspects,
        cache,
        load_email,
        judge_fn,
        model,
        progress=progress,
        workers=workers,
        provenance=prov,
    )
    cache.close()
    return {**counts, "suspects": len(suspects)}
