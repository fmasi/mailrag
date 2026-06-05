"""Pure clustering + noise-pocket scoring for the `explore` verb.

No I/O: callers pass a vector matrix (one mean-pooled vector per thread) and a
parallel list of :class:`ThreadMeta`; this module L2-normalizes, runs
MiniBatchKMeans, scores each cluster, and returns a :class:`ClusterReport`.

The noise score combines three transparent components, each in ``[0, 1]``:
  * ``tag_rate``  - mean per-thread pass1-tag fraction in the cluster
  * ``sender_concentration`` - mean per-thread top-sender share
  * ``tightness`` - mean cosine of members to the cluster centroid
``score = mean(tag_rate, sender_concentration, tightness)``. It is a sortable
heuristic to guide the eye, not a verdict.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
from sklearn.cluster import MiniBatchKMeans

_EPS = 1e-9
_K_CAP = 50
# A cluster scoring at/above this is treated as an "obvious" noise pocket for the
# persona advisor; corpora with a large obvious fraction are worth trimming.
_OBVIOUS_SCORE = 0.6
_OBVIOUS_FRACTION_FOR_TRIM = 0.25


@dataclass
class ThreadMeta:
    """Per-thread metadata, parallel to a row of the vector matrix."""
    thread_id: str
    n_emails: int
    dominant_sender: str
    top_sender_share: float
    n_senders: int
    tag_fraction: float
    sample_subjects: List[str]
    paths: List[str]


@dataclass
class ClusterInfo:
    id: int
    size: int
    n_emails: int
    score: float
    tag_rate: float
    tag_lift: float
    sender_concentration: float
    tightness: float
    top_senders: List[Tuple[str, float]]
    n_senders: int
    sample_subjects: List[str]
    members: List[dict] = field(default_factory=list)


@dataclass
class ClusterReport:
    k: int
    seed: int
    n_threads: int
    n_emails: int
    corpus_baseline_tag_rate: float
    clusters: List[ClusterInfo]
    vector_source: str = "fresh"

    @property
    def obvious_noise_fraction(self) -> float:
        """Share of threads sitting in 'obvious' noise pockets (score >= cutoff)."""
        if not self.n_threads:
            return 0.0
        obvious = sum(c.size for c in self.clusters if c.score >= _OBVIOUS_SCORE)
        return round(obvious / self.n_threads, 4)

    def to_json_dict(self, *, profile: str, collection: str,
                     vector_source: str) -> dict:
        persona, reason = recommend_persona(self)
        return {
            "profile": profile,
            "collection": collection,
            "vector_source": vector_source,
            "k": self.k,
            "seed": self.seed,
            "n_threads": self.n_threads,
            "n_emails": self.n_emails,
            "corpus_baseline_tag_rate": self.corpus_baseline_tag_rate,
            "obvious_noise_fraction": self.obvious_noise_fraction,
            "recommended_persona": persona,
            "recommendation_reason": reason,
            "clusters": [
                {
                    "id": c.id,
                    "size": c.size,
                    "n_emails": c.n_emails,
                    "score": c.score,
                    "components": {
                        "tag_rate": c.tag_rate,
                        "tag_lift": c.tag_lift,
                        "sender_concentration": c.sender_concentration,
                        "tightness": c.tightness,
                    },
                    "top_senders": [list(t) for t in c.top_senders],
                    "n_senders": c.n_senders,
                    "sample_subjects": c.sample_subjects,
                    "members": c.members,
                }
                for c in self.clusters
            ],
        }


def recommend_persona(report: "ClusterReport") -> Tuple[str, str]:
    """Recommend a persona from the obvious-noise fraction.

    A large obvious-noise fraction means there is a lot to drop cheaply, so the
    verified-trim persona pays off; otherwise there is little to trim and judging
    everything (``llm-all``) is the simpler call. (``llm-none`` is the no-LLM
    choice — a budget preference rather than a noise-driven recommendation.)"""
    f = report.obvious_noise_fraction
    if f >= _OBVIOUS_FRACTION_FOR_TRIM:
        return ("llm-verify",
                f"{f:.0%} of threads sit in obvious noise pockets — trim them with a "
                f"cheap LLM check (llm-verify), or skip the LLM entirely (llm-none)")
    return ("llm-all",
            f"only {f:.0%} obvious noise — little to trim safely, so judge everything "
            f"for max quality (llm-all)")


def default_k(n_threads: int, requested: int | None = None) -> int:
    """Heuristic cluster count, clamped to ``[2, min(50, n_threads)]``."""
    k = requested if requested else max(2, round(math.sqrt(n_threads / 2)))
    return max(2, min(k, _K_CAP, n_threads))


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / np.maximum(norms, _EPS)


def cluster_threads(vectors: np.ndarray, metas: List[ThreadMeta], *,
                    k: int | None = None, seed: int = 11) -> ClusterReport:
    """Cluster ``vectors`` (one row per thread) and score each cluster."""
    if len(metas) != len(vectors):
        raise ValueError("vectors and metas must be parallel")
    n = len(metas)
    if n == 0:
        raise ValueError("no threads to cluster")
    k = default_k(n, requested=k)

    unit = _l2_normalize(np.asarray(vectors, dtype=float))
    km = MiniBatchKMeans(n_clusters=k, random_state=seed, n_init=3, batch_size=256)
    labels = km.fit_predict(unit)

    baseline = float(np.mean([m.tag_fraction for m in metas]))
    clusters: List[ClusterInfo] = []
    for cid in range(k):
        idx = [i for i in range(n) if labels[i] == cid]
        if not idx:
            continue
        members = [metas[i] for i in idx]
        sub = unit[idx]
        centroid = _l2_normalize(sub.mean(axis=0, keepdims=True))[0]
        tightness = float(np.clip(np.mean(sub @ centroid), 0.0, 1.0))
        tag_rate = float(np.mean([m.tag_fraction for m in members]))
        sender_conc = float(np.mean([m.top_sender_share for m in members]))
        score = float(np.mean([tag_rate, sender_conc, tightness]))
        dom = Counter(m.dominant_sender for m in members)
        top_senders = [(s, round(c / len(members), 4)) for s, c in dom.most_common(3)]
        subjects: List[str] = []
        for m in members:
            for s in m.sample_subjects:
                if s and s not in subjects:
                    subjects.append(s)
                if len(subjects) >= 3:
                    break
            if len(subjects) >= 3:
                break
        clusters.append(ClusterInfo(
            id=cid, size=len(members),
            n_emails=sum(m.n_emails for m in members),
            score=round(score, 4), tag_rate=round(tag_rate, 4),
            tag_lift=round(tag_rate / max(baseline, _EPS), 4),
            sender_concentration=round(sender_conc, 4),
            tightness=round(tightness, 4), top_senders=top_senders,
            n_senders=len(dom),
            sample_subjects=subjects,
            members=[{"thread_id": m.thread_id, "paths": m.paths} for m in members],
        ))

    clusters.sort(key=lambda c: c.score, reverse=True)
    return ClusterReport(
        k=k, seed=seed, n_threads=n,
        n_emails=sum(m.n_emails for m in metas),
        corpus_baseline_tag_rate=round(baseline, 6), clusters=clusters)


def format_report(report: ClusterReport, *, top: int = 15) -> str:
    """Render the top clusters as a rich table; return it as plain text."""
    from io import StringIO
    from rich.console import Console
    from rich.table import Table

    table = Table(title=None, show_lines=False)
    for col in ("id", "threads", "emails", "score", "tag_lift",
                "top_sender (share)", "n_send", "tight", "sample subjects"):
        table.add_column(col, overflow="fold")
    for c in report.clusters[:top]:
        ts = c.top_senders[0] if c.top_senders else ("-", 0.0)
        table.add_row(
            str(c.id), str(c.size), str(c.n_emails), f"{c.score:.3f}",
            f"{c.tag_lift:.2f}", f"{ts[0]} ({ts[1]:.0%})", str(c.n_senders),
            f"{c.tightness:.2f}", " | ".join(c.sample_subjects[:2]),
        )
    buf = StringIO()
    console = Console(file=buf, width=140)
    console.print(
        f"scan: {report.n_threads} threads / {report.n_emails} emails, "
        f"k={report.k}, seed={report.seed}, "
        f"baseline tag rate={report.corpus_baseline_tag_rate:.3f}")
    console.print(table)
    persona, reason = recommend_persona(report)
    console.print(f"recommended persona: [bold]{persona}[/bold] — {reason}")
    return buf.getvalue()
