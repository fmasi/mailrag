# src/eval/metrics.py
"""Graded-relevance ranking metrics + per-category aggregation.

Pool-relative recall: `total_relevant` is the count of grade>=rel emails in the
judged pool for that query, not absolute corpus recall (documented in §3.3).
"""

from __future__ import annotations

import math
from typing import Dict, List


def dcg(grades: List[float]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(grades))


def ndcg_at_k(grades: List[int], k: int) -> float:
    g = grades[:k]
    ideal = sorted(grades, reverse=True)[:k]
    idcg = dcg(ideal)
    return dcg(g) / idcg if idcg > 0 else 0.0


def precision_at_k(grades: List[int], k: int, rel: int) -> float:
    g = grades[:k]
    if not g:
        return 0.0
    return sum(1 for x in g if x >= rel) / k


def recall_at_k(grades: List[int], k: int, rel: int, total_relevant: int) -> float:
    if total_relevant <= 0:
        return 0.0
    found = sum(1 for x in grades[:k] if x >= rel)
    return found / total_relevant


def mrr(grades: List[int], rel: int) -> float:
    for i, g in enumerate(grades):
        if g >= rel:
            return 1.0 / (i + 1)
    return 0.0


def success_at_k(ranked_message_ids: List[str], target_id: str, k: int) -> int:
    return 1 if target_id in ranked_message_ids[:k] else 0


def aggregate(rows: List[Dict], metric_keys: List[str]) -> Dict[str, Dict[str, float]]:
    """Mean each metric overall and per `category`. Adds `n` (row count) per group."""
    groups: Dict[str, List[Dict]] = {"overall": list(rows)}
    for r in rows:
        groups.setdefault(r["category"], []).append(r)
    out: Dict[str, Dict[str, float]] = {}
    for name, grp in groups.items():
        d = {"n": len(grp)}
        for key in metric_keys:
            vals = [r[key] for r in grp if key in r]
            d[key] = sum(vals) / len(vals) if vals else 0.0
        out[name] = d
    return out
