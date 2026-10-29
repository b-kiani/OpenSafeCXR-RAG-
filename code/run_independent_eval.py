#!/usr/bin/env python3
"""
run_independent_eval.py -- F1-CheXbert and official F1-RadGraph for all conditions.

Regenerates the paper's Table II and the paired-bootstrap CIs in Sec. IV-A from the
archived generated reports.

    pip install "radeval>=2.2" "radgraph==0.1.18" numpy pandas
    python run_independent_eval.py --dir ./conditions --out ./results

Input: one CSV per condition in --dir, columns uid,generated,reference.
File stems name the conditions; the two compared by the bootstrap default to
combined_rag.csv and gated_ragpp.csv (override with --base / --gated).
The script refuses to run unless every condition has identical uid and reference lists.

APIs used (checked against the released package sources):
  radeval 2.x : RadEval(metrics=["f1chexbert"])(refs=..., hyps=...) ->
                keys f1chexbert_{5,all}_{micro,macro,weighted}_f1
  radgraph 0.1.18 : F1RadGraph(reward_level="all", model_type="radgraph-xl")(refs=, hyps=)
                -> (mean_triplet, (entity_list, entity_relation_list, complete_list), ...)
Note the radgraph positional order in forward() is (refs, hyps); we pass keywords.
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd


def load_conditions(d):
    conds, ref = {}, None
    for f in sorted(glob.glob(os.path.join(d, "*.csv"))):
        name = os.path.splitext(os.path.basename(f))[0]
        df = pd.read_csv(f).sort_values("uid").reset_index(drop=True)
        for c in ("generated", "reference"):
            df[c] = df[c].fillna("").astype(str)
        if ref is None:
            ref = df[["uid", "reference"]].copy()
        else:
            if df["uid"].tolist() != ref["uid"].tolist():
                raise SystemExit(f"{name}: uid list differs from the first condition")
            if df["reference"].tolist() != ref["reference"].tolist():
                raise SystemExit(f"{name}: reference reports differ -- not a common test set")
        conds[name] = df
        print(f"  {name}: {len(df)} records")
    if not conds:
        raise SystemExit(f"no CSV files in {d}")
    return conds


def paired_bootstrap(a, b, n_boot=10000, seed=42):
    """Mean of (b - a) over cases with a percentile 95% CI."""
    rng = np.random.default_rng(seed)
    d = np.asarray(b, float) - np.asarray(a, float)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boot = d[idx].mean(axis=1)
    return float(d.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def chexbert(evaluator, hyps, refs):
    out = evaluator(refs=refs, hyps=hyps)
    return {k: float(v) for k, v in out.items() if k.startswith("f1chexbert_")}


def radgraph_cases(scorer, hyps, refs):
    mean, rewards, _, _ = scorer(refs=refs, hyps=hyps)
    per_case = np.column_stack([np.asarray(r, float) for r in rewards])   # (n, 3)
    return [float(x) for x in mean], per_case


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--base", default="combined_rag")
    ap.add_argument("--gated", default="gated_ragpp")
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    print("Loading conditions")
    conds = load_conditions(a.dir)

    from radeval import RadEval
    from radgraph import F1RadGraph
    cx = RadEval(metrics=["f1chexbert"], show_progress=False)
    rg = F1RadGraph(reward_level="all", model_type="radgraph-xl")

    results, cases = {}, {}
    for name, df in conds.items():
        hyps, refs = df["generated"].tolist(), df["reference"].tolist()
        r = {"chexbert": chexbert(cx, hyps, refs)}
        mean, per_case = radgraph_cases(rg, hyps, refs)
        r["radgraph"] = dict(zip(["entity", "entity_relation", "complete"], mean))
        results[name], cases[name] = r, per_case
        np.save(os.path.join(a.out, f"radgraph_percase_{name}.npy"), per_case)
        print(f"{name:<24} 5mic={r['chexbert'].get('f1chexbert_5_micro_f1', float('nan')):.3f} "
              f"allmic={r['chexbert'].get('f1chexbert_all_micro_f1', float('nan')):.3f} "
              f"RG={mean[0]:.3f}/{mean[1]:.3f}/{mean[2]:.3f}")

    if a.base in cases and a.gated in cases:
        boot = {}
        for i, lvl in enumerate(["entity", "entity_relation", "complete"]):
            d, lo, hi = paired_bootstrap(cases[a.base][:, i], cases[a.gated][:, i], a.bootstrap)
            boot[lvl] = {"delta": d, "ci95": [lo, hi], "excludes_zero": bool(lo * hi > 0)}
            print(f"  {lvl:<16} delta={d:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")
        results["_paired_bootstrap"] = {"base": a.base, "gated": a.gated,
                                        "resamples": a.bootstrap, **boot}
    else:
        print(f"(bootstrap skipped: need {a.base}.csv and {a.gated}.csv)")

    with open(os.path.join(a.out, "independent_eval.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {a.out}/independent_eval.json. If any value differs from Table II, "
          f"correct the paper, not the output.")


if __name__ == "__main__":
    main()
