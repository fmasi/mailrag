"""ZTH onboarding: build a validated thread-aware contextual assistant from an
.eml directory in one bounded LLM pass.
"""

import json
import os
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from src.data.threading import assign_subject_fallback_thread_ids


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


# Resolved per call rather than at import so tests (and anyone with a
# non-default home) can redirect it. A module-level constant meant the suite
# wrote manifests into the developer's real ~/.mailrag.
def manifest_dir() -> Path:
    return Path(os.path.expanduser(os.environ.get("MAILRAG_HOME") or "~/.mailrag"))


class _ManifestDirProxy:
    """Backwards-compatible stand-in for the old module constant."""

    def _p(self) -> Path:
        return manifest_dir()

    def mkdir(self, **kw):
        return self._p().mkdir(**kw)

    def is_dir(self) -> bool:
        return self._p().is_dir()

    def glob(self, pattern):
        return self._p().glob(pattern)

    def __truediv__(self, other):
        return self._p() / other

    def __fspath__(self):
        return os.fspath(self._p())

    def __repr__(self):
        return repr(self._p())


MANIFEST_DIR = _ManifestDirProxy()


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
        cov = (
            f"coverage@3 = {self.coverage_at3:.0%} on {self.n_queries} queries"
            if self.validated
            else "validation skipped"
        )
        fails = f", {self.llm_failures} llm-failures" if self.llm_failures else ""
        return (
            f"Onboarded {self.kept} emails ({self.noise_dropped} noise dropped"
            f"{fails}) -> {self.chunks} chunks in '{self.collection}'; {cov}. "
            f"Query it: mailrag query --collection {self.collection} '...'"
        )


def write_manifest(report, *, source, model, profile_path=None):
    """Persist a reproducibility manifest to ``~/.mailrag/<collection>.json`` and
    return its path.

    ``profile_path`` records WHICH corpus profile produced this collection. That
    mapping is load-bearing rather than informational: scoping a grep to a
    collection means walking exactly the files that collection's profile
    selects, and until now nothing recorded the link — it was recovered by
    scanning ``*.profile.json`` in a directory and reading the ``collection``
    each one names. That works until profiles move, get renamed, or two name the
    same collection. Written here so the collection carries its own provenance.
    """
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    path = MANIFEST_DIR / f"{report.collection}.json"
    data = asdict(report)
    data.update(
        source=source,
        model=model,
        created_at=datetime.now(timezone.utc).isoformat(),
        defaults={"fusion": "rrf", "sparse_weight": 1, "thread_aware": True, "reranker": False},
    )
    if profile_path:
        data["profile_path"] = os.path.abspath(os.path.expanduser(profile_path))
    path.write_text(json.dumps(data, indent=2))
    return str(path)


def record_profile_for_collection(collection: str, profile_path: str) -> None:
    """Record which corpus profile a collection was built from.

    Written by the profile-driven commands (``index``, ``attachments build``)
    because ``onboard`` is directory-driven and has no profile to name. Merges
    into any existing manifest rather than replacing it, so registering the
    mapping never costs the reproducibility record.

    The mapping is load-bearing: scoping a grep to a collection means walking
    exactly the files that collection's profile selects. Recovering it by
    scanning a directory for ``*.profile.json`` works until a profile is moved,
    renamed, or two of them name the same collection — so the collection should
    carry its own provenance.
    """
    if not isinstance(collection, str) or not isinstance(profile_path, str):
        return  # nothing useful to record (and nothing JSON-serialisable)
    if not collection or not profile_path:
        return
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    path = MANIFEST_DIR / f"{collection}.json"
    data = {}
    if path.exists():
        try:
            data = json.loads(path.read_text())
        except Exception:
            data = {}
    data["collection"] = collection
    data["profile_path"] = os.path.abspath(os.path.expanduser(profile_path))
    data["profile_recorded_at"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(data, indent=2))


def manifest_profile_paths() -> dict:
    """``{collection: profile path}`` from manifests that recorded one."""
    out: dict = {}
    if not MANIFEST_DIR.is_dir():
        return out
    for f in sorted(MANIFEST_DIR.glob("*.json")):
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        coll, prof = d.get("collection"), d.get("profile_path")
        if coll and prof and os.path.exists(os.path.expanduser(prof)):
            out[coll] = os.path.expanduser(prof)
    return out


def latest_manifest_collection():
    """Collection name from the most-recently-written manifest, or None."""
    if not MANIFEST_DIR.is_dir():
        return None
    manifests = sorted(MANIFEST_DIR.glob("*.json"), key=lambda p: (p.stat().st_mtime, p.name))
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
    from src.eval.coverage_diag import distinct_thread_rank
    from src.query.thread_expand import _node_metadata

    if not queries:
        return None, 0
    covered = 0
    for q in queries:
        hits = []
        for node in searcher.search(q["query"]):
            md = _node_metadata(node)
            hits.append({"thread_id": md.get("thread_id"), "message_id": md.get("message_id")})
        rank = distinct_thread_rank(hits, q["thread_id"])
        if rank is not None and rank < 3:
            covered += 1
    return covered / len(queries), len(queries)


def validate_coverage(collection, *, searcher=None, queries_path=None, counts=None, seed=7):
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

            with tempfile.TemporaryDirectory() as td:
                tmp = os.path.join(td, "queries.jsonl")
                gen_run(collection, counts, tmp, seed)
                queries = _load_queries(tmp)
        if searcher is None:
            from src.query.hybrid import build_hybrid_searcher

            searcher = build_hybrid_searcher(collection, mode="hybrid")
        return _coverage_at3(searcher, queries)
    except Exception as exc:  # best-effort validation
        print(f"validation skipped: {exc}", file=sys.stderr)
        return None, 0


def _require_qdrant(qdrant_url):
    """Fail fast (before the expensive LLM pass) if Qdrant is unreachable."""
    from src.ingest import hybrid_qdrant as hq

    try:
        hq.get_client(qdrant_url).get_collections()
    except Exception as e:
        raise ValueError(
            f"Qdrant not reachable at {qdrant_url}. Start it with "
            f"`docker compose up -d qdrant`. ({e})"
        )


def _profile_chunk_size(emails):
    from transformers import AutoTokenizer

    from src.ingest.profile import suggest_chunk_size

    tok = AutoTokenizer.from_pretrained("BAAI/bge-m3")
    lengths = [
        len(tok.encode(e.body or "", add_special_tokens=False))
        for e in emails
        if (e.body or "").strip()
    ]
    return suggest_chunk_size(lengths)


def run_onboard(
    source_dir,
    *,
    collection=None,
    chunk_size=None,
    queries_path=None,
    validate=True,
    limit=None,
    noise_min_confidence=0.7,
    model=None,
    qdrant_url="http://localhost:6333",
    embedder=None,
    cache_path=None,
):
    """Profile -> single resumable LLM pass -> filter+build -> validate -> report."""
    from src.indexing.contextual_index import build_contextual_index
    from src.llm.cache import Pass2Cache
    from src.llm.onboard_pass import generate_thread_judgments

    collection = collection or collection_slug(source_dir)
    _require_qdrant(qdrant_url)

    emails = load_eml_dir(source_dir, limit=limit)
    assign_subject_fallback_thread_ids(emails)

    if chunk_size is None:
        chunk_size = _profile_chunk_size(emails)

    cache_path = cache_path or str(MANIFEST_DIR / f"{collection}.pass.db")
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    cache = Pass2Cache(cache_path)
    from tqdm import tqdm

    bar = tqdm(total=len(emails), desc="clean+summarize")
    try:
        judgments = generate_thread_judgments(emails, cache=cache, model=model, progress=bar.update)
    finally:
        bar.close()
        cache.close()

    llm_failures = sum(
        1 for r in judgments.values() if str(r.get("reason", "")).startswith("llm_error")
    )
    kept, noise_dropped = filter_kept(emails, judgments, min_confidence=noise_min_confidence)
    if not kept:
        raise ValueError("all emails were filtered as noise; nothing to index")

    if embedder is None:
        from src.ingest.embedder import BgeM3Embedder

        embedder = BgeM3Embedder(use_fp16=True)
    res = build_contextual_index(
        kept,
        collection=collection,
        embedder=embedder,
        summaries=None,  # summaries already set on each e.summary by filter_kept()
        embed_summary=True,
        chunk_size=chunk_size,
        recreate=True,
        qdrant_url=qdrant_url,
    )

    coverage, n_queries = (None, 0)
    if validate:
        coverage, n_queries = validate_coverage(collection, queries_path=queries_path)

    report = OnboardReport(
        collection=collection,
        kept=len(kept),
        noise_dropped=noise_dropped,
        llm_failures=llm_failures,
        chunks=res.chunks,
        chunk_size=chunk_size,
        coverage_at3=coverage,
        n_queries=n_queries,
        validated=(coverage is not None),
    )
    write_manifest(report, source=str(source_dir), model=(model or ""))
    return report
