# src/eval/pooling.py
"""TREC-style pooling: union arm rankings into a judge pool; map grades back."""

from __future__ import annotations

from typing import Dict, List

from src.eval.flatten import EmailHit


def build_pool(arms: Dict[str, List[EmailHit]]) -> List[EmailHit]:
    """Union of all arms' hits, one per message_id (first arm/rank seen wins)."""
    seen: Dict[str, EmailHit] = {}
    for hits in arms.values():
        for h in hits:
            if h.message_id not in seen:
                seen[h.message_id] = h
    return list(seen.values())


def graded_rankings(
    arms: Dict[str, List[EmailHit]], grades: Dict[str, int]
) -> Dict[str, List[int]]:
    """Each arm's ranked message_ids → ranked grade vector (missing grade = 0)."""
    return {arm: [grades.get(h.message_id, 0) for h in hits] for arm, hits in arms.items()}


def total_relevant(grades: Dict[str, int], rel: int) -> int:
    """Count of pool emails graded relevant (>= rel) — denominator for recall."""
    return sum(1 for g in grades.values() if g >= rel)
