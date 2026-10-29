#!/usr/bin/env python3
"""
run_composite_sensitivity.py -- legacy composite score, equal-weight variant, and
+/-25% weight-perturbation ranking stability (paper Sec. IV-E).

The composite is a legacy, study-specific summary built from ontology-dependent diagnostics.
It is NOT used for any conclusion in the paper; this script exists so the reported
ranking statement can be checked.

    python run_composite_sensitivity.py --components composite_components.csv --sims 10000

composite_components.csv: one row per condition with columns
    condition,S,C,H,O,RE,RR,RL,B1,SS,U,T,penalty
  S  legacy clinical-safety score      C  rule-based CheXbert-style F1
  H  detector-defined unsupported rate O  detector-defined omission rate
  RE/RR legacy RadGraph-style proxies  RL ROUGE-L   B1 BLEU-1
  SS structured-output rate            U  unique-output ratio
  T  top-output repetition             penalty  post-scoring penalty
     (-0.15 retrieval-only, -0.05 non-structured generation, else 0)
Fill it from the archived diagnostics file; do not estimate missing values.
"""
import argparse
import csv
import numpy as np

W = {"S": 0.22, "C": 0.22, "H": 0.14, "O": 0.10, "RE": 0.08, "RR": 0.04,
     "RL": 0.06, "B1": 0.04, "SS": 0.05, "U": 0.05, "T": 0.08}
POSITIVE = ["S", "C", "H", "O", "RE", "RR", "RL", "B1", "SS", "U"]
COLS = POSITIVE + ["T", "penalty"]


def score(c, w):
    return (w["S"] * c["S"] + w["C"] * c["C"] + w["H"] * (1 - c["H"]) + w["O"] * (1 - c["O"])
            + w["RE"] * c["RE"] + w["RR"] * c["RR"] + w["RL"] * c["RL"] + w["B1"] * c["B1"]
            + w["SS"] * c["SS"] + w["U"] * c["U"] - w["T"] * c["T"] + c["penalty"])


def ranking(conds, w):
    s = {k: score(v, w) for k, v in conds.items()}
    return s, sorted(s, key=s.get, reverse=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--components", required=True)
    ap.add_argument("--target", default="Gated RAG++")
    ap.add_argument("--sims", type=int, default=10000)
    ap.add_argument("--pct", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    conds = {}
    with open(a.components) as f:
        for row in csv.DictReader(f):
            missing = [c for c in COLS if row.get(c, "") == ""]
            if missing:
                raise SystemExit(f"{row['condition']}: missing {missing}; fill from archived diagnostics")
            conds[row["condition"]] = {c: float(row[c]) for c in COLS}
    if a.target not in conds:
        raise SystemExit(f"target '{a.target}' not in {list(conds)}")

    for title, w in [("Original weights", W),
                     ("Equal positive weights",
                      {k: (1 / len(POSITIVE) if k in POSITIVE else W["T"]) for k in W})]:
        s, order = ranking(conds, w)
        print(f"\n{title}")
        for i, k in enumerate(order, 1):
            print(f"  {i}. {k:<24} {s[k]:.4f}")

    rng = np.random.default_rng(a.seed)
    ranks = []
    for _ in range(a.sims):
        w = {k: v * (1 + rng.uniform(-a.pct, a.pct)) for k, v in W.items()}
        ranks.append(ranking(conds, w)[1].index(a.target) + 1)
    ranks = np.asarray(ranks)
    print(f"\n{a.sims} simulations, +/-{a.pct:.0%} per weight: '{a.target}' ranked first in "
          f"{(ranks == 1).mean():.1%}; mean rank {ranks.mean():.3f}; worst {ranks.max()}")


if __name__ == "__main__":
    main()
