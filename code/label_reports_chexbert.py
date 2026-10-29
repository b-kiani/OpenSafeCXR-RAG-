#!/usr/bin/env python3
"""
label_reports_chexbert.py -- CheXbert labels for IU reference reports (7 shared findings).

Produces the proxy target-domain labels used by recalibrate_thresholds.py.
Run it once for the validation split and once for the test split.

    pip install "radeval>=2.2" numpy pandas
    python label_reports_chexbert.py --csv iu_val.csv  --out iu_val_labels.npy
    python label_reports_chexbert.py --csv iu_test.csv --out iu_test_labels.npy

Input CSV columns: uid,report  (rows in the same order as the detector-probability arrays)

Label convention: RadEval "rrg" mode, i.e. CheXbert "positive" AND "uncertain" -> 1,
"negative"/"blank" -> 0. Use the same convention that produced the paper's Table IV;
if that run treated uncertain as negative, pass --uncertain-negative.
"""
import argparse
import numpy as np
import pandas as pd

from common import SEVEN_NIH, NIH_TO_CHEXBERT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--uncertain-negative", action="store_true",
                    help="map CheXbert 'uncertain' to 0 instead of 1")
    a = ap.parse_args()

    from radeval.metrics.f1chexbert.f1chexbert import F1CheXbert
    scorer = F1CheXbert(device=a.device)

    df = pd.read_csv(a.csv)
    reports = df["report"].fillna("").astype(str).tolist()
    mode = "classification" if a.uncertain_negative else "rrg"
    rows = np.asarray(scorer.get_labels(reports, mode=mode))
    if a.uncertain_negative:
        rows = (rows == 1).astype(int)          # classification: 1 pos, -1 uncertain, 0 neg

    names = list(scorer.TARGET_NAMES)
    cols = [names.index(NIH_TO_CHEXBERT[l]) for l in SEVEN_NIH]
    labels = rows[:, cols].astype(np.int8)
    np.save(a.out, labels)
    np.save(a.out.replace(".npy", "_uids.npy"), df["uid"].to_numpy())
    print(f"wrote {a.out}: shape {labels.shape}; positives per finding:")
    for l, c in zip(SEVEN_NIH, labels.sum(0)):
        print(f"  {l:<14} {int(c)}")


if __name__ == "__main__":
    main()
