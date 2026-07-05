# src/eval/decisions.py
"""Encode the pre-registered eval decision rules (spec §1).

retire_cprime: thread-aware (C+rerank+thread) must tie-or-beat the better C' arm
  on terse/spanning categories AND beat it on content/entity categories
  (mean nDCG@10), i.e. dominate-or-match everywhere.
auto_bound: > 10% of retrieved threads exceed the 8k-token budget.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from src.eval.metrics import aggregate, ndcg_at_k

THREAD_ARM = "C+rerank+thread"
CPRIME_ARMS = ("Cprime", "Cprime+rerank")
_TERSE = {"terse", "spanning"}
_TIE_EPS = 0.02  # within-noise tolerance for "tie-or-beat"


def _per_query_ndcg(rankings: List[Dict], grades: Dict, arm: str, k: int = 10) -> List[Dict]:
    rows = []
    for r in rankings:
        ids = r["arms"].get(arm, [])
        g = [grades.get(r["query"], {}).get(mid, 0) for mid in ids]
        rows.append({"category": r["category"], "ndcg@10": ndcg_at_k(g, k)})
    return rows


def decide(
    rankings: List[Dict], grades: Dict, pct_over_8k: Optional[float] = None
) -> Dict[str, bool]:
    thread = aggregate(_per_query_ndcg(rankings, grades, THREAD_ARM), ["ndcg@10"])
    # Best (max) C' arm per group is the bar to clear.
    cprime_groups = [
        aggregate(_per_query_ndcg(rankings, grades, a), ["ndcg@10"]) for a in CPRIME_ARMS
    ]

    def cprime_best(group):
        return max(g.get(group, {}).get("ndcg@10", 0.0) for g in cprime_groups)

    terse_ok = thread.get("terse", {}).get("ndcg@10", 0.0) + _TIE_EPS >= cprime_best("terse")
    span_ok = thread.get("spanning", {}).get("ndcg@10", 0.0) + _TIE_EPS >= cprime_best("spanning")
    content_ok = thread.get("content", {}).get("ndcg@10", 0.0) >= cprime_best(
        "content"
    )  # strict beat
    retire = bool(terse_ok and span_ok and content_ok)

    auto_bound = bool(pct_over_8k is not None and pct_over_8k > 10.0)
    return {"retire_cprime": retire, "auto_bound": auto_bound}
