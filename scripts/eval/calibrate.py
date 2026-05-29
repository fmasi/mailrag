# scripts/eval/calibrate.py
"""Compare local vs reference judge on the calibration subset; decide the route.

Reads local grades.json + calib_ref_grades.json (same shape), computes Cohen's
kappa + Spearman on the overlapping (query,message_id) grades, recomputes the two
pre-registered decisions under each judge, and reports whether either flips.

Run (any env with src importable):
  conda run -n rag --no-capture-output \
    python scripts/eval/calibrate.py --local eval/out/grades.json \
    --ref eval/out/calib_ref_grades.json --rankings eval/out/arm_rankings.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.eval.agreement import cohen_kappa, spearman, decision_flips
from src.eval.decisions import decide  # added in Task 12


def _paired(local, ref):
    a, b = [], []
    for q, mids in ref.items():
        for mid, g in mids.items():
            if q in local and mid in local[q]:
                a.append(local[q][mid]); b.append(g)
    return a, b


def run(local_path, ref_path, rankings_path):
    local = json.load(open(local_path))
    ref = json.load(open(ref_path))
    rankings = json.load(open(rankings_path))

    a, b = _paired(local, ref)
    print(f"paired judgments: {len(a)}")
    print(f"cohen_kappa: {cohen_kappa(a, b):.3f}")
    print(f"spearman:    {spearman([float(x) for x in a], [float(x) for x in b]):.3f}")

    # Restrict rankings to the calibrated queries, decide under each judge.
    calib_qs = set(ref.keys())
    sub = [r for r in rankings if r["query"] in calib_qs]
    d_local = decide(sub, local)
    # For the reference decision, fill missing grades from local (ref only covers the subset's pool partially)
    merged = {q: {**local.get(q, {}), **ref.get(q, {})} for q in calib_qs}
    d_ref = decide(sub, merged)
    flips = decision_flips(d_local, d_ref)
    print(f"decisions(local): {d_local}")
    print(f"decisions(ref):   {d_ref}")
    print(f"DECISION FLIPS: {flips or 'none'}")
    print("ROUTE: " + ("all-local OK" if not flips else "ESCALATE judge to Opus for full run"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local", default="eval/out/grades.json")
    ap.add_argument("--ref", default="eval/out/calib_ref_grades.json")
    ap.add_argument("--rankings", default="eval/out/arm_rankings.json")
    args = ap.parse_args()
    run(args.local, args.ref, args.rankings)


if __name__ == "__main__":
    main()
