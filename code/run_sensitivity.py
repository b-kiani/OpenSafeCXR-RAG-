#!/usr/bin/env python3
"""
run_sensitivity.py -- retrieval-quality grid (paper Sec. IV-D) and detector operating-point sweep (Sec. IV-C)

Two subcommands:

  retrieval   alpha/K sweep scored by independent CheXbert label agreement
              

  threshold   detector operating-point sweep quantifying the
              unsupported-positive vs omission trade-off

Install:
    pip install numpy pandas

Usage:
    python run_sensitivity.py retrieval \
        --train-emb train_img.npy --train-abn train_abn.npy --train-labels train_chexbert.npy \
        --test-emb  test_img.npy  --test-abn  test_abn.npy  --test-labels  test_chexbert.npy \
        --train-uid train_uid.npy --test-uid test_uid.npy --out retrieval_sensitivity.csv

    python run_sensitivity.py threshold \
        --probs iu_test_probs.npy --labels iu_test_chexbert7.npy --out threshold_sweep.csv
"""
import argparse
import numpy as np
import pandas as pd

from common import SEVEN_NIH as SEVEN, NIH_THRESHOLDS_14 as NIH_THRESHOLDS


def l2norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(n, 1e-12, None)


def jaccard(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return 1.0 if union == 0 else inter / union


def cmd_retrieval(a):
    te, ta = l2norm(np.load(a.train_emb)), l2norm(np.load(a.train_abn))
    qe, qa = l2norm(np.load(a.test_emb)),  l2norm(np.load(a.test_abn))
    tl, ql = np.load(a.train_labels).astype(bool), np.load(a.test_labels).astype(bool)
    tu, qu = np.load(a.train_uid, allow_pickle=True), np.load(a.test_uid, allow_pickle=True)

    sim_img, sim_abn = qe @ te.T, qa @ ta.T
    block = (qu[:, None] == tu[None, :])          # same-UID exclusion
    rows = []
    for alpha in [0.0, 0.25, 0.50, 0.75, 1.0]:
        s = alpha * sim_abn + (1 - alpha) * sim_img
        s = np.where(block, -np.inf, s)
        order = np.argsort(-s, axis=1)
        for K in [1, 3, 5, 10]:
            js, hs = [], []
            for i in range(len(qe)):
                for j in order[i, :K]:
                    js.append(jaccard(ql[i], tl[j]))
                    hs.append((ql[i] == tl[j]).mean())
            rows.append({"alpha": alpha, "K": K,
                         "mean_chexbert_jaccard": float(np.mean(js)),
                         "mean_hamming_agreement": float(np.mean(hs))})
            print(f"  alpha={alpha:<5} K={K:<3} Jaccard={np.mean(js):.4f} "
                  f"Hamming={np.mean(hs):.4f}")
    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)
    best = df.loc[df["mean_chexbert_jaccard"].idxmax()]
    print(f"\nBest configuration: alpha={best.alpha}, K={int(best.K)}, "
          f"Jaccard={best.mean_chexbert_jaccard:.4f}")
    print("If the best config is alpha=0, abnormality weighting does NOT improve")
    print("retrieval quality -- report that, do not bury it.")


def cmd_threshold(a):
    probs = np.load(a.probs)          # (n, 7) detector probabilities, SEVEN order
    labels = np.load(a.labels).astype(bool)   # (n, 7) CheXbert-derived IU proxy labels
    n = len(probs)
    rows = []

    def score(pred, tag):
        tp = np.logical_and(pred, labels).sum()
        fp = np.logical_and(pred, ~labels).sum()
        fn = np.logical_and(~pred, labels).sum()
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f = 2 * p * r / (p + r) if (p + r) else 0.0
        rows.append({"setting": tag, "precision": p, "recall": r, "micro_f1": f,
                     "unsupported_pos_per_case": fp / n,
                     "omitted_pos_per_case": fn / n})
        print(f"  {tag:<22} P={p:.3f} R={r:.3f} F1={f:.3f} "
              f"FP/case={fp/n:.3f} FN/case={fn/n:.3f}")

    thr = np.array([NIH_THRESHOLDS[l] for l in SEVEN])
    score(probs >= thr[None, :], "retained NIH thr")
    for g in np.arange(0.05, 1.00, 0.05):
        score(probs >= g, f"global {g:.2f}")

    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)
    g = df[df.setting.str.startswith("global")]
    b = g.loc[g["micro_f1"].idxmax()]
    print(f"\nBest global threshold: {b.setting} -> F1={b.micro_f1:.3f}, "
          f"FP/case={b.unsupported_pos_per_case:.3f}, FN/case={b.omitted_pos_per_case:.3f}")
    print("Confirm that lowering the threshold raises FP/case -- that trade-off IS the finding.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("retrieval")
    for x in ["train-emb", "train-abn", "train-labels", "test-emb", "test-abn",
              "test-labels", "train-uid", "test-uid"]:
        r.add_argument(f"--{x}", required=True)
    r.add_argument("--out", default="retrieval_sensitivity.csv")
    r.set_defaults(func=cmd_retrieval)

    t = sub.add_parser("threshold")
    t.add_argument("--probs", required=True)
    t.add_argument("--labels", required=True)
    t.add_argument("--out", default="threshold_sweep.csv")
    t.set_defaults(func=cmd_threshold)

    args = ap.parse_args()
    args.func(args)
