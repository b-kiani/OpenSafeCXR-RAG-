#!/usr/bin/env python3
"""
recalibrate_thresholds.py -- target-domain operating points for the NIH-trained detector.

Addresses ICTACS Reviewer #1 (item 1) and Reviewer #3 (item 2): choose the detector's
per-label thresholds on the IU *validation* split (CheXbert-derived labels), freeze them,
and measure the effect on the IU *test* split. The test split is never used for selection.

Inputs (all numpy .npy, rows aligned, columns in common.SEVEN_NIH order):
    --val-probs   (n_val, 7)   detector probabilities on IU validation images
    --val-labels  (n_val, 7)   0/1 labels from label_reports_chexbert.py on IU validation reports
    --test-probs  (n_test, 7)  detector probabilities on IU test images
    --test-labels (n_test, 7)  0/1 labels on IU test reports

Strategies (--strategy):
    f1           per-label threshold maximising validation F1
    precision    lowest threshold reaching validation precision >= --min-precision;
                 a label that never reaches it is DISABLED (never asserted)
    youden       per-label threshold maximising sensitivity + specificity - 1

Optional --calibrate {platt,isotonic} fits a probability calibrator per label on validation
and reports Brier score / ECE on test before and after (thresholds are then chosen on the
calibrated probabilities).

Outputs in --out-dir:
    recalibration_per_label.csv   per-label thresholds and test P/R/F1 (NIH vs recalibrated)
    recalibration_summary.json    micro metrics, errors per case, paired bootstrap CI
    recalibration_snippet.tex     a LaTeX table + paragraph filled with THESE numbers

Example:
    python recalibrate_thresholds.py --val-probs iu_val_probs.npy --val-labels iu_val_labels.npy \
        --test-probs iu_test_probs.npy --test-labels iu_test_labels.npy \
        --strategy f1 --out-dir recal_f1

Sanity check: with --strategy nih the script must reproduce the paper's Table IV IU F1
column and the micro P/R/F1 of 0.292/0.155/0.202. If it does not, the inputs differ from
the archived run and the recalibration numbers must not be reported alongside Table IV.
"""
import argparse
import json
import os
import numpy as np

from common import SEVEN_NIH, NIH_THRESHOLDS_7

GRID = np.round(np.arange(0.01, 1.00, 0.01), 2)
MIN_POS_WARN = 10


def prf(pred, y):
    tp = int(np.sum(pred & y)); fp = int(np.sum(pred & ~y)); fn = int(np.sum(~pred & y))
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f, tp, fp, fn


def choose_threshold(p, y, strategy, min_precision):
    if y.sum() == 0:
        return np.inf, "no validation positives: disabled"
    best_t, best_v = None, -1.0
    for t in GRID:
        pred = p >= t
        P, R, F, tp, fp, fn = prf(pred, y)
        if strategy == "f1":
            v = F
        elif strategy == "youden":
            tn = int(np.sum(~pred & ~y))
            spec = tn / (tn + fp) if tn + fp else 0.0
            v = R + spec - 1
        elif strategy == "precision":
            if P >= min_precision and tp > 0:
                return float(t), f"precision {P:.3f} >= {min_precision}"
            continue
        if v > best_v + 1e-12 or (abs(v - best_v) <= 1e-12 and best_t is not None and t > best_t):
            best_t, best_v = float(t), v      # ties -> higher (more conservative) threshold
    if strategy == "precision":
        return np.inf, f"never reaches precision {min_precision}: disabled"
    return best_t, f"val {strategy}={best_v:.3f}"


def ece(p, y, bins=10):
    edges = np.linspace(0, 1, bins + 1); e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi) if hi < 1 else (p >= lo) & (p <= hi)
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)


def fit_calibrator(kind, p, y):
    if kind == "isotonic":
        from sklearn.isotonic import IsotonicRegression
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p, y)
        return iso.predict
    from sklearn.linear_model import LogisticRegression
    z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))
    lr = LogisticRegression(C=1e6).fit(z.reshape(-1, 1), y)
    return lambda q: lr.predict_proba(
        np.log(np.clip(q, 1e-6, 1 - 1e-6) / np.clip(1 - q, 1e-6, 1)).reshape(-1, 1))[:, 1]


def micro(pred, Y):
    P, R, F, tp, fp, fn = prf(pred.ravel(), Y.ravel())
    n = Y.shape[0]
    return {"precision": P, "recall": R, "micro_f1": F,
            "unsupported_pos_per_case": fp / n, "omitted_pos_per_case": fn / n}


def paired_bootstrap_micro_f1(pa, pb, Y, n_boot, seed):
    rng = np.random.default_rng(seed); n = Y.shape[0]; d = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        d.append(micro(pb[i], Y[i])["micro_f1"] - micro(pa[i], Y[i])["micro_f1"])
    d = np.asarray(d)
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-probs", required=True); ap.add_argument("--val-labels", required=True)
    ap.add_argument("--test-probs", required=True); ap.add_argument("--test-labels", required=True)
    ap.add_argument("--strategy", choices=["f1", "precision", "youden", "nih"], default="f1")
    ap.add_argument("--min-precision", type=float, default=0.5)
    ap.add_argument("--calibrate", choices=["none", "platt", "isotonic"], default="none")
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="recalibration_out")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    Pv, Yv = np.load(a.val_probs).astype(float), np.load(a.val_labels).astype(bool)
    Pt, Yt = np.load(a.test_probs).astype(float), np.load(a.test_labels).astype(bool)
    for name, arr in [("val-probs", Pv), ("val-labels", Yv), ("test-probs", Pt), ("test-labels", Yt)]:
        assert arr.ndim == 2 and arr.shape[1] == 7, f"{name} must be (n, 7) in {SEVEN_NIH} order"
    assert len(Pv) == len(Yv) and len(Pt) == len(Yt), "probability/label row counts differ"

    Pt_raw = Pt.copy()          # the NIH-threshold baseline always uses uncalibrated outputs
    calib_report = {}
    if a.calibrate != "none":
        Pv_c, Pt_c = Pv.copy(), Pt.copy()
        for j, lab in enumerate(SEVEN_NIH):
            if Yv[:, j].sum() == 0:
                continue
            f = fit_calibrator(a.calibrate, Pv[:, j], Yv[:, j].astype(int))
            Pv_c[:, j], Pt_c[:, j] = f(Pv[:, j]), f(Pt[:, j])
            calib_report[lab] = {
                "test_brier_before": float(np.mean((Pt[:, j] - Yt[:, j]) ** 2)),
                "test_brier_after": float(np.mean((Pt_c[:, j] - Yt[:, j]) ** 2)),
                "test_ece_before": ece(Pt[:, j], Yt[:, j]),
                "test_ece_after": ece(Pt_c[:, j], Yt[:, j]),
            }
        Pv, Pt = Pv_c, Pt_c

    nih = np.array(NIH_THRESHOLDS_7)
    new, notes = [], []
    print(f"\nValidation n={len(Yv)}, test n={len(Yt)}, strategy={a.strategy}, calibrate={a.calibrate}\n")
    for j, lab in enumerate(SEVEN_NIH):
        npos = int(Yv[:, j].sum())
        if a.strategy == "nih":
            t, why = nih[j], "archived NIH threshold"
        else:
            t, why = choose_threshold(Pv[:, j], Yv[:, j], a.strategy, a.min_precision)
        if 0 < npos < MIN_POS_WARN:
            why += f"; WARNING only {npos} validation positives"
        new.append(t); notes.append(why)
    new = np.array(new)

    pred_nih = Pt_raw >= nih[None, :]
    pred_new = Pt >= new[None, :]
    rows = []
    for j, lab in enumerate(SEVEN_NIH):
        a_ = prf(pred_nih[:, j], Yt[:, j]); b_ = prf(pred_new[:, j], Yt[:, j])
        rows.append({"finding": lab, "val_pos": int(Yv[:, j].sum()), "test_pos": int(Yt[:, j].sum()),
                     "theta_nih": float(nih[j]),
                     "theta_new": (None if np.isinf(new[j]) else float(new[j])),
                     "test_P_nih": a_[0], "test_R_nih": a_[1], "test_F1_nih": a_[2],
                     "test_P_new": b_[0], "test_R_new": b_[1], "test_F1_new": b_[2],
                     "note": notes[j]})

    m_nih, m_new = micro(pred_nih, Yt), micro(pred_new, Yt)
    lo, hi = paired_bootstrap_micro_f1(pred_nih, pred_new, Yt, a.bootstrap, a.seed)
    zero_nih = [r["finding"] for r in rows if r["test_F1_nih"] == 0]
    zero_new = [r["finding"] for r in rows if r["test_F1_new"] == 0]

    import csv
    with open(os.path.join(a.out_dir, "recalibration_per_label.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    summary = {"strategy": a.strategy, "min_precision": a.min_precision, "calibrate": a.calibrate,
               "n_val": int(len(Yv)), "n_test": int(len(Yt)),
               "test_micro_nih": m_nih, "test_micro_recalibrated": m_new,
               "delta_micro_f1": m_new["micro_f1"] - m_nih["micro_f1"],
               "delta_micro_f1_95ci": [lo, hi], "bootstrap": a.bootstrap,
               "zero_f1_nih": zero_nih, "zero_f1_recalibrated": zero_new,
               "calibration": calib_report}
    with open(os.path.join(a.out_dir, "recalibration_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # ---- console report
    print(f"{'finding':<14}{'valpos':>7}{'th_NIH':>8}{'th_new':>8}{'F1_NIH':>8}{'F1_new':>8}  note")
    for r in rows:
        tn = "off" if r["theta_new"] is None else f"{r['theta_new']:.2f}"
        print(f"{r['finding']:<14}{r['val_pos']:>7}{r['theta_nih']:>8.2f}{tn:>8}"
              f"{r['test_F1_nih']:>8.3f}{r['test_F1_new']:>8.3f}  {r['note']}")
    for tag, m in [("NIH thresholds", m_nih), ("recalibrated", m_new)]:
        print(f"\n{tag:<15} P={m['precision']:.3f} R={m['recall']:.3f} F1={m['micro_f1']:.3f} "
              f"FP/case={m['unsupported_pos_per_case']:.3f} FN/case={m['omitted_pos_per_case']:.3f}")
    print(f"\nDelta micro-F1 = {summary['delta_micro_f1']:+.3f}, 95% CI [{lo:+.3f}, {hi:+.3f}]")
    if abs(m_nih["micro_f1"] - 0.202) > 0.002:
        print("\n!! NIH-threshold micro-F1 is not 0.202: inputs differ from the archived run "
              "(Table IV). Do not report these numbers next to Table IV without explaining why.")

    # ---- LaTeX snippet with the numbers just computed
    def fmt(x): return "off" if x is None else f"{x:.2f}"
    trs = "\n".join(f"{r['finding']} & {r['theta_nih']:.2f} & {fmt(r['theta_new'])} & "
                    f"{r['test_F1_nih']:.3f} & {r['test_F1_new']:.3f} \\\\" for r in rows)
    sig = "excludes" if lo * hi > 0 else "includes"
    strat = {"f1": "maximise validation F1", "youden": "maximise Youden's J",
             "precision": f"reach validation precision $\\geq{a.min_precision}$ (labels that never do are disabled)",
             "nih": "retain the NIH thresholds"}[a.strategy]
    tex = f"""% Auto-generated by recalibrate_thresholds.py -- numbers below are from this run.
\\begin{{table}}[t]
\\caption{{Detector thresholds recalibrated on IU validation reports}}
\\label{{tab:recal}}
\\centering\\footnotesize
\\begin{{tabular}}{{@{{}}lcccc@{{}}}}
\\toprule
Finding & $\\theta_{{\\mathrm{{NIH}}}}$ & $\\theta_{{\\mathrm{{IU}}}}$ & F1 (NIH $\\theta$) & F1 (IU $\\theta$) \\\\
\\midrule
{trs}
\\midrule
Micro & & & {m_nih['micro_f1']:.3f} & {m_new['micro_f1']:.3f} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}

% Paragraph:
Thresholds were re-selected per label on the {len(Yv)} IU validation reports to {strat}, then frozen
and applied to the test split. Test micro-F1 changed from {m_nih['micro_f1']:.3f} to {m_new['micro_f1']:.3f}
(paired bootstrap 95\\% CI of the difference [{lo:+.3f}, {hi:+.3f}], which {sig} zero); unsupported-positive
predictions per case went from {m_nih['unsupported_pos_per_case']:.3f} to {m_new['unsupported_pos_per_case']:.3f}
and omitted labels per case from {m_nih['omitted_pos_per_case']:.3f} to {m_new['omitted_pos_per_case']:.3f}.
Findings with zero test F1: {', '.join(zero_nih) or 'none'} at NIH thresholds; {', '.join(zero_new) or 'none'} after recalibration.
"""
    with open(os.path.join(a.out_dir, "recalibration_snippet.tex"), "w") as f:
        f.write(tex)
    print(f"\nWrote {a.out_dir}/recalibration_per_label.csv, recalibration_summary.json, recalibration_snippet.tex")


if __name__ == "__main__":
    main()
