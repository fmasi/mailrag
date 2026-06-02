"""ZTH onboarding: build a validated thread-aware contextual assistant from an
.eml directory in one bounded LLM pass. See
docs/superpowers/specs/2026-06-02-zth-onboard-design.md.
"""
import json
import os
import re
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path


def collection_slug(source_dir):
    """Default collection name for a source directory: ``mailrag-<slug>``."""
    name = Path(source_dir).resolve().name or "corpus"
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "corpus"
    return f"mailrag-{slug}"


def load_eml_dir(source_dir, *, limit=None):
    """Load every ``.eml`` under ``source_dir`` (recursively) into NormalizedEmail
    objects. Raises ValueError when the directory is missing or has no .eml files."""
    root = Path(source_dir)
    if not root.is_dir():
        raise ValueError(f"not a directory: {source_dir}")
    paths = sorted(str(p) for p in root.rglob("*.eml"))
    if limit:
        paths = paths[:limit]
    if not paths:
        raise ValueError(f"no .eml files found under {source_dir}")
    from src.data.loaders.mail_archive_x import MailArchiveXLoader
    return MailArchiveXLoader(eml_files=paths).load()


def filter_kept(emails, judgments, *, min_confidence=0.7):
    """Drop emails judged noise with confidence >= ``min_confidence``; set
    ``.summary`` on the kept ones from their judgment. Emails with no judgment are
    kept (conservative). Returns ``(kept_emails, n_noise_dropped)``."""
    kept, dropped = [], 0
    for e in emails:
        mid = getattr(e, "message_id", "") or ""
        rec = judgments.get(mid)
        if rec and rec["is_noise"] and rec["confidence"] >= min_confidence:
            dropped += 1
            continue
        if rec and rec.get("summary"):
            try:
                e.summary = rec["summary"]
            except (AttributeError, TypeError):
                pass
        kept.append(e)
    return kept, dropped


MANIFEST_DIR = Path(os.path.expanduser("~/.mailrag"))


@dataclass
class OnboardReport:
    collection: str
    kept: int
    noise_dropped: int
    llm_failures: int
    chunks: int
    chunk_size: int
    coverage_at3: "float | None"
    n_queries: int
    validated: bool

    def one_line(self):
        cov = (f"coverage@3 = {self.coverage_at3:.0%} on {self.n_queries} queries"
               if self.validated else "validation skipped")
        fails = f", {self.llm_failures} llm-failures" if self.llm_failures else ""
        return (f"Onboarded {self.kept} emails ({self.noise_dropped} noise dropped"
                f"{fails}) -> {self.chunks} chunks in '{self.collection}'; {cov}. "
                f"Query it: mailrag query --collection {self.collection} '...'")


def write_manifest(report, *, source, model):
    """Persist a reproducibility manifest to ``~/.mailrag/<collection>.json`` and
    return its path."""
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    path = MANIFEST_DIR / f"{report.collection}.json"
    data = asdict(report)
    data.update(source=source, model=model,
                created_at=datetime.now(timezone.utc).isoformat(),
                defaults={"fusion": "rrf", "sparse_weight": 1,
                          "thread_aware": True, "reranker": False})
    path.write_text(json.dumps(data, indent=2))
    return str(path)


def latest_manifest_collection():
    """Collection name from the most-recently-written manifest, or None."""
    if not MANIFEST_DIR.is_dir():
        return None
    manifests = sorted(MANIFEST_DIR.glob("*.json"),
                       key=lambda p: (p.stat().st_mtime, p.name))
    if not manifests:
        return None
    return json.loads(manifests[-1].read_text()).get("collection")


def _load_queries(path):
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _coverage_at3(searcher, queries):
    """covered@3 over thread-distinct ranks. ``queries``: list of dicts with
    ``query`` and ``thread_id``. Returns ``(coverage, n_queries)``."""
    from src.query.thread_expand import _node_metadata
    from src.eval.coverage_diag import distinct_thread_rank
    if not queries:
        return None, 0
    covered = 0
    for q in queries:
        hits = []
        for node in searcher.search(q["query"]):
            md = _node_metadata(node)
            hits.append({"thread_id": md.get("thread_id"),
                         "message_id": md.get("message_id")})
        rank = distinct_thread_rank(hits, q["thread_id"])
        if rank is not None and rank < 3:
            covered += 1
    return covered / len(queries), len(queries)


def validate_coverage(collection, *, searcher=None, queries_path=None,
                      counts=None, seed=7):
    """Return ``(coverage_at3, n_queries)``. Uses ``queries_path`` if given, else
    auto-generates a small validated set via gen_queries.run. Best-effort: returns
    ``(None, 0)`` on any failure (the index is already usable). Query rows must
    carry ``query`` and ``thread_id`` (the gen_queries.run schema)."""
    counts = counts or {"terse": 5, "content": 5, "spanning": 3}
    try:
        if queries_path:
            queries = _load_queries(queries_path)
        else:
            import tempfile
            from scripts.eval.gen_queries import run as gen_run
            tmp = os.path.join(tempfile.mkdtemp(), "queries.jsonl")
            gen_run(collection, counts, tmp, seed)
            queries = _load_queries(tmp)
        if searcher is None:
            from src.query.hybrid import build_hybrid_searcher
            searcher = build_hybrid_searcher(collection, mode="hybrid")
        return _coverage_at3(searcher, queries)
    except Exception as exc:  # best-effort validation
        print(f"validation skipped: {exc}")
        return None, 0
