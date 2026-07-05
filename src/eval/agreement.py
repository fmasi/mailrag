# src/eval/agreement.py
"""Calibration stats: Cohen's kappa, Spearman rho, and decision-flip detection.

Dependency-free (no numpy/scipy) so it runs in the lightweight test env.
"""

from __future__ import annotations

from typing import Dict, List


def cohen_kappa(a: List[int], b: List[int]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    n = len(a)
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    labels = set(a) | set(b)
    pe = sum((a.count(l) / n) * (b.count(l) / n) for l in labels)
    return 1.0 if pe == 1.0 else (po - pe) / (1 - pe)


def _rank(xs: List[float]) -> List[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # average rank (1-based) for ties
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(a: List[float], b: List[float]) -> float:
    if len(a) != len(b) or len(a) < 2:
        return 0.0
    ra, rb = _rank(a), _rank(b)
    n = len(a)
    ma = sum(ra) / n
    mb = sum(rb) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    va = sum((x - ma) ** 2 for x in ra) ** 0.5
    vb = sum((y - mb) ** 2 for y in rb) ** 0.5
    return cov / (va * vb) if va > 0 and vb > 0 else 0.0


def decision_flips(local: Dict[str, bool], ref: Dict[str, bool]) -> List[str]:
    """Names of decisions whose boolean outcome differs between the two judges."""
    keys = set(local) | set(ref)
    return [k for k in keys if local.get(k) != ref.get(k)]
